from functools import partial
import argparse
import copy

import einops
import torch
from transformer_lens import HookedTransformer

from buffer import Buffer
from crosscoder import CrossCoder
from utils import get_gsm8k_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='version_4', help='CrossCoder checkpoint version number')
parser.add_argument('--ckpt', type=int, default=6, help='CrossCoder checkpoint')

args = parser.parse_args()

torch.set_grad_enabled(False)  # important for memory saving

device = 'cuda:0'
DTYPE = torch.bfloat16

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device=device,
    dtype=DTYPE,
)

math_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    device=device,
    dtype=DTYPE,
)

# current best is version 4 checkpoint 6
cross_coder = CrossCoder.load(args.version, args.ckpt)
collect_layer = 14

train_questions, train_answers = get_gsm8k_dataset(split='train')

# dataset
eos = base_model.tokenizer.special_tokens_map['eos_token']
train_questions, train_answers = get_gsm8k_dataset(split='train')
merged_prompts = [q + f" {eos} " + a for q, a in zip(train_questions, train_answers)]
all_tokens_padded = base_model.tokenizer(merged_prompts, padding=True, return_tensors='pt')
all_tokens_padded.input_ids.shape


def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    cross_coder.W_enc.data[0, :, :] = cross_coder.W_enc.data[0, :, :] * base_scaling_factor
    cross_coder.W_enc.data[1, :, :] = cross_coder.W_enc.data[1, :, :] * chat_scaling_factor

    cross_coder.W_dec.data[:, 0, :] = cross_coder.W_dec.data[:, 0, :] / base_scaling_factor
    cross_coder.W_dec.data[:, 1, :] = cross_coder.W_dec.data[:, 1, :] / chat_scaling_factor

    cross_coder.b_dec.data[0, :] = cross_coder.b_dec.data[0, :] / base_scaling_factor
    cross_coder.b_dec.data[1, :] = cross_coder.b_dec.data[1, :] / chat_scaling_factor
    return cross_coder


def splice_act_hook(act, hook, spliced_act):
    act[:, 1:, :] = spliced_act  # Drop BOS
    return act


def zero_ablation_hook(act, hook):
    act[:] = 0
    return act


def get_ce_recovered_metrics(tokens, masks, model_A, model_B, cross_coder):
    # get clean loss
    ce_clean_A = model_A(tokens, attention_mask=masks, return_type="loss")
    ce_clean_B = model_B(tokens, attention_mask=masks, return_type="loss")

    # get zero abl loss
    ce_zero_abl_A = model_A.run_with_hooks(
        tokens,
        attention_mask=masks,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )
    ce_zero_abl_B = model_B.run_with_hooks(
        tokens,
        attention_mask=masks,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )

    # bunch of annoying set up for splicing
    _, cache_A = model_A.run_with_cache(
        tokens,
        attention_mask=masks,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
    )
    resid_act_A = cache_A[cross_coder.cfg["hook_point"]]

    _, cache_B = model_B.run_with_cache(
        tokens,
        attention_mask=masks,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
    )
    resid_act_B = cache_B[cross_coder.cfg["hook_point"]]

    cross_coder_input = torch.stack([resid_act_A, resid_act_B], dim=0)
    cross_coder_input = cross_coder_input[:, :, 1:, :]  # Drop BOS
    cross_coder_input = einops.rearrange(
        cross_coder_input,
        "n_models batch seq_len d_model -> (batch seq_len) n_models d_model",
    )

    cross_coder_output = cross_coder.decode(cross_coder.encode(cross_coder_input))
    cross_coder_output = einops.rearrange(
        cross_coder_output,
        "(batch seq_len) n_models d_model -> n_models batch seq_len d_model",
        batch=tokens.shape[0]
    )
    cross_coder_output_A = cross_coder_output[0]
    cross_coder_output_B = cross_coder_output[1]

    # get spliced loss
    ce_loss_spliced_A = model_A.run_with_hooks(
        tokens,
        attention_mask=masks,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_A))],
    )
    ce_loss_spliced_B = model_B.run_with_hooks(
        tokens,
        attention_mask=masks,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_B))],
    )

    # compute % CE recovered metric
    ce_recovered_A = 1 - ((ce_loss_spliced_A - ce_clean_A) / (ce_zero_abl_A - ce_clean_A))
    ce_recovered_B = 1 - ((ce_loss_spliced_B - ce_clean_B) / (ce_zero_abl_B - ce_clean_B))

    metrics = {
        "ce_loss_spliced_A": ce_loss_spliced_A.item(),
        "ce_loss_spliced_B": ce_loss_spliced_B.item(),
        "ce_clean_A": ce_clean_A.item(),
        "ce_clean_B": ce_clean_B.item(),
        "ce_zero_abl_A": ce_zero_abl_A.item(),
        "ce_zero_abl_B": ce_zero_abl_B.item(),
        "ce_diff_A": (ce_loss_spliced_A - ce_clean_A).item(),
        "ce_diff_B": (ce_loss_spliced_B - ce_clean_B).item(),
        "ce_recovered_A": ce_recovered_A.item(),
        "ce_recovered_B": ce_recovered_B.item(),
    }
    return metrics


# Estimating normalizing factor
folded_cross_coder = copy.deepcopy(cross_coder)

buff = Buffer(cross_coder.cfg, base_model, math_model, all_tokens_padded)
base_estimated_scaling_factor, math_estimated_scaling_factor = buff.normalisation_factor.detach().cpu().numpy()
print(base_estimated_scaling_factor, math_estimated_scaling_factor)
folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_estimated_scaling_factor, math_estimated_scaling_factor)
folded_cross_coder = folded_cross_coder.to(base_model.cfg.dtype)

num_cals = 20
perm_idx = torch.randperm(len(all_tokens_padded.input_ids))[:num_cals]
tokens = all_tokens_padded.input_ids[perm_idx]
masks = all_tokens_padded.attention_mask[perm_idx]
ce_metrics = get_ce_recovered_metrics(tokens, masks, base_model, math_model, folded_cross_coder)

for (k, v) in ce_metrics.items():
    print(f"{k}: {v}")
del tokens