# %%
from transformer_lens import HookedTransformer
from utils import arg_parse_update_cfg, get_gsm8k_dataset
from trainer import Trainer
import torch
from datasets import load_dataset
# %%

device='cuda:0'

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device=device,
    dtype=torch.bfloat16,
)

chat_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B", 
    device=device,
    dtype=torch.bfloat16,
)

# %%

eos = base_model.tokenizer.special_tokens_map['eos_token']
train_questions, train_answers = get_gsm8k_dataset(split='train')
merged_prompts = [
    f"Problem: {q} {eos} Answer: {a}" for q, a in zip(train_questions, train_answers)
]

data_prm800k = load_dataset('json', data_files='data/prm800k/prm800k_train.jsonl')['train']
problem, solution, answer = data_prm800k['problem'], data_prm800k['solution'], data_prm800k['answer']
eos = base_model.tokenizer.special_tokens_map['eos_token']

merged_prompts_prm800k = [
    f"Problem: {q} {eos} Answer: {a} {eos} Final Answer: {fa}" for (q, a, fa) in zip(problem, solution, answer)
]

max_length = 2048
all_tokens = base_model.tokenizer(
    merged_prompts + merged_prompts_prm800k,
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=max_length
)

# %%
dict_scaler = 8

default_cfg = {
    "seed": 42,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": dict_scaler * base_model.cfg.d_model,
    "seq_len": 2048,
    "enc_dtype": "fp32",
    "model_name": "qwen2_5_1_5B",
    "device": f"cuda:0",
    "site": "resid_pre",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "crosscoder",
}

cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
# %%
