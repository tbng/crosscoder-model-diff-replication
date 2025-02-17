# %%
from transformer_lens import HookedTransformer
from utils import arg_parse_update_cfg, get_gsm8k_dataset
from trainer import Trainer
# %%
device = 'cuda:0'


base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device=device, 
)

chat_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    device=device, 
)

# %%
eos = base_model.tokenizer.special_tokens_map['eos_token']
train_questions, train_answers = get_gsm8k_dataset(split='train')
merged_prompts = [q + f" {eos} " + a for q, a in zip(train_questions, train_answers)]
all_tokens = base_model.tokenizer(merged_prompts, padding=True, return_tensors="pt")

# %%
default_cfg = {
    "seed": 42,
    "batch_size": 64,
    "buffer_mult": 32,
    "lr": 5e-5,
    "num_tokens": 40_000_00,
    "l1_coeff": 2.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**13,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "qwen2_5_1_5B",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 2,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "crosscoder",
    "wandb_entity": "binhnt",
}

cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
# %%
