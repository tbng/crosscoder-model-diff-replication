import re
from dataclasses import dataclass
from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import einops

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

@dataclass
class CrossCoderConfig:
    """Class for storing configuration parameters for the CrossCoder"""

    d_in: int
    d_hidden: int | None = None
    dict_mult: int | None = None

    l1_coeff: float = 3e-4

    apply_b_dec_to_input: bool = False

    def __post_init__(self):
        assert (
            int(self.d_hidden is None) + int(self.dict_mult is None) == 1
        ), "Exactly one of d_hidden or dict_mult must be provided"
        if (self.d_hidden is None) and isinstance(self.dict_mult, int):
            self.d_hidden = self.d_in * self.dict_mult
        elif (self.dict_mult is None) and isinstance(self.d_hidden, int):
            #assert self.d_hidden % self.d_in == 0, "d_hidden must be a multiple of d_in"
            self.dict_mult = self.d_hidden // self.d_in


class CrossCoder(nn.Module):
    def __init__(self, cfg: CrossCoderConfig):
        super().__init__()
        self.cfg = cfg

        assert isinstance(cfg.d_hidden, int)

        # W_enc has shape (d_in, d_encoder), where d_encoder is a multiple of d_in (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(2, cfg.d_in, cfg.d_hidden))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, 2, cfg.d_in))
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(2, cfg.d_in))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor):
        # TODO: lots of this stuff is legacy SAE stuff that probably is wrong / unnecessary
        x_cent = x - self.b_dec * self.cfg.apply_b_dec_to_input
        x_enc = einops.einsum(
            x_cent,
            self.W_enc,
            "... n_layers d_model, n_layers d_model d_hidden -> ... d_hidden",
        )
        acts = F.relu(x_enc + self.b_enc)
        #x_reconstruct = acts @ self.W_dec + self.b_dec
        x_reconstruct = einops.einsum(
            acts,
            self.W_dec,
            "... d_hidden, d_hidden n_layers d_model -> ... n_layers d_model",
        )
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_layers d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()
        #l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_layers]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_layers -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    def __repr__(self) -> str:
        return f"CrossCoder(d_in={self.cfg.d_in}, dict_mult={self.cfg.dict_mult})"


# # ==============================================================
# # ! TRANSFORMERS
# # This returns the activations & resid_pre as well (optionally)
# # ==============================================================


class TransformerLensWrapper(nn.Module):
    """
    This class wraps around & extends the TransformerLens model, so that we can make sure things like the forward
    function have a standardized signature.
    """

    def __init__(self, model: HookedTransformer, hook_point: str):
        super().__init__()
        assert (
            hook_point in model.hook_dict
        ), f"Error: hook_point={hook_point!r} must be in model.hook_dict"
        self.model = model
        self.hook_point = hook_point

        # Get the layer (so we can do the early stopping in our forward pass)
        layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
        assert layer_match, f"Error: expecting hook_point to be 'blocks.{{layer}}.{{...}}', but got {hook_point!r}"
        self.hook_layer = int(layer_match.group(1))

        # Get the hook names for the residual stream (final) and residual stream (immediately after hook_point)
        self.hook_point_resid = utils.get_act_name("resid_post", self.hook_layer)
        self.hook_point_resid_final = utils.get_act_name(
            "resid_post", self.model.cfg.n_layers - 1
        )
        assert self.hook_point_resid in model.hook_dict
        assert self.hook_point_resid_final in model.hook_dict

    @overload
    def forward(
        self,
        tokens: Tensor,
        return_logits: Literal[True],
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        tokens: Tensor,
        return_logits: Literal[False],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        tokens: Int[Tensor, "batch seq"],
        attention_mask: Tensor,
        return_logits: bool = True,
    ):
        """
        Inputs:
            tokens: Int[Tensor, "batch seq"]
                The input tokens, shape (batch, seq)
            return_logits: bool
                If True, returns (logits, residual, activation)
                If False, returns (residual, activation)
        """

        # Run with hook functions to store the activations & final value of residual stream
        # If return_logits is False, then we compute the last residual stream value but not the logits
        output: Tensor = self.model.run_with_hooks(
            tokens,
            attention_mask=attention_mask,
            # stop_at_layer = (None if return_logits else self.hook_layer),
            fwd_hooks=[
                (self.hook_point, self.hook_fn_store_act),
                (self.hook_point_resid_final, self.hook_fn_store_act),
            ],
        )

        # The hook functions work by storing data in model's hook context, so we pop them back out
        activation: Tensor = self.model.hook_dict[self.hook_point].ctx.pop("activation")
        if self.hook_point_resid_final == self.hook_point:
            residual: Tensor = activation
        else:
            residual: Tensor = self.model.hook_dict[
                self.hook_point_resid_final
            ].ctx.pop("activation")

        if return_logits:
            return output, residual, activation
        return residual, activation

    def hook_fn_store_act(self, activation: torch.Tensor, hook: HookPoint):
        hook.ctx["activation"] = activation

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def W_U(self):
        return self.model.W_U

    @property
    def W_out(self):
        return self.model.W_out


def to_resid_dir(dir: Float[Tensor, "feats d_in"], model: TransformerLensWrapper):
    """
    Takes a direction (eg. in the post-ReLU MLP activations) and returns the corresponding dir in the residual stream.

    Args:
        dir:
            The direction in the activations, i.e. shape (feats, d_in) where d_in could be d_model, d_mlp, etc.
        model:
            The model, which should be a HookedTransformerWrapper or similar.
    """
    # If this SAE was trained on the residual stream or attn/mlp out, then we don't need to do anything
    if "resid" in model.hook_point or "_out" in model.hook_point:
        return dir

    # If it was trained on the MLP layer, then we apply the W_out map
    elif ("pre" in model.hook_point) or ("post" in model.hook_point):
        return dir @ model.W_out[model.hook_layer]

    # Others not yet supported
    else:
        raise NotImplementedError(
            "The hook your SAE was trained on isn't yet supported"
        )
