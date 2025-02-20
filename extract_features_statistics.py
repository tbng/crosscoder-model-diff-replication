from datetime import datetime

import copy
import plotly.express as px
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_vis.data_config_classes import SaeVisConfig
from crosscoder import CrossCoder
from utils import get_gsm8k_dataset
from datasets import load_dataset
from sae_vis.data_storing_fns import SaeVisData

torch.set_grad_enabled(False)  # important for memory saving

device = 'cuda:0'


base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device=device,
    dtype=torch.bfloat16,
)

math_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    device=device,
    dtype=torch.bfloat16,
)


version = 'version_4'
ckpt = 6

cross_coder = CrossCoder.load(version, ckpt)
collect_layer = 14

train_questions, train_answers = get_gsm8k_dataset(split='train')


# Load dataset

eos = base_model.tokenizer.special_tokens_map['eos_token']
train_questions, train_answers = get_gsm8k_dataset(split='train')
merged_prompts = [
    f"Problem: {q} {eos} Answer: {a}" for q, a in zip(train_questions, train_answers)
]

data_prm800k = load_dataset(
    'json', data_files='data/prm800k/prm800k_train.jsonl')['train']
problem, solution, answer = data_prm800k['problem'], data_prm800k['solution'], data_prm800k['answer']
eos = base_model.tokenizer.special_tokens_map['eos_token']

merged_prompts_prm800k = [
    f"Problem: {q} {eos} Answer: {a} {eos} Final Answer: {fa}" for (q, a, fa) in zip(problem, solution, answer)
]

combined_prompts = merged_prompts + merged_prompts_prm800k

max_length = 256
all_tokens_padded = base_model.tokenizer(
    combined_prompts,
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=max_length
)

norms = cross_coder.W_dec.norm(dim=-1)
relative_norms = norms[:, 1] / norms.sum(dim=-1)

# Hard-coded normalizing factor to avoid accumulation in memory. These are the ones for version_10_3
base_estimated_scaling_factor, math_estimated_scaling_factor = 0.7367579, 0.38336924

folded_cross_coder = copy.deepcopy(cross_coder)

# we'll only fold the normalization scaling factors into W_enc, since we aren't splicing back into the model with the reconstructed


def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    cross_coder.W_enc.data[0, :, :] = cross_coder.W_enc.data[0,
                                                             :, :] * base_scaling_factor
    cross_coder.W_enc.data[1, :, :] = cross_coder.W_enc.data[1,
                                                             :, :] * chat_scaling_factor

    return cross_coder


def get_sae_vis_cross_coder(folded_cross_coder):

    from sae_vis.model_fns import CrossCoderConfig, CrossCoder

    encoder_cfg = CrossCoderConfig(d_in=base_model.cfg.d_model,
                                   d_hidden=cross_coder.cfg["dict_size"],
                                   apply_b_dec_to_input=False)
    sae_vis_cross_coder = CrossCoder(encoder_cfg)
    sae_vis_cross_coder.load_state_dict(folded_cross_coder.state_dict())
    sae_vis_cross_coder = sae_vis_cross_coder.to(device)
    sae_vis_cross_coder = sae_vis_cross_coder.to(torch.bfloat16)
    return sae_vis_cross_coder


folded_cross_coder = fold_activation_scaling_factor(
    folded_cross_coder, base_estimated_scaling_factor, math_estimated_scaling_factor)
sae_vis_cross_coder = get_sae_vis_cross_coder(folded_cross_coder)

distinct_latent_mask = (relative_norms > 0.95).cpu()
torch.sum(distinct_latent_mask)
distinct_feature_idx = torch.arange(cross_coder.cfg['dict_size'])[
    distinct_latent_mask]

sae_vis_config = SaeVisConfig(
    hook_point=folded_cross_coder.cfg["hook_point"],
    features=distinct_feature_idx,
    verbose=True,
    minibatch_size_tokens=4,
    minibatch_size_features=16,
)

print(len(distinct_feature_idx))


num_samples = 100
perm_idx = torch.randperm(len(all_tokens_padded.input_ids))[
    :num_samples].cpu().numpy()

sae_vis_data = SaeVisData.create(
    encoder=sae_vis_cross_coder,
    encoder_B=None,
    model_A=base_model,
    model_B=math_model,
    # in practice, better to use more data
    tokens=all_tokens_padded.input_ids[perm_idx],
    attention_mask=all_tokens_padded.attention_mask[perm_idx],
    cfg=sae_vis_config,
)

# Save visualization

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"crosscoder_{current_time}_{version}_{ckpt}_feature_vis_demo.html"
promptvis = f"crosscoder_{current_time}_{version}_{ckpt}_prompt_vis_demo.html"
sae_vis_data.save_feature_centric_vis(filename)

# sae_vis_data.save_prompt_centric_vis(combined_prompts[perm_idx], promptvis)

print('Done!')
