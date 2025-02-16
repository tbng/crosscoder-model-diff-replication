# %%
from transformer_lens import HookedTransformer
from utils import arg_parse_update_cfg
from datasets import load_dataset
from trainer import Trainer
# %%
device = 'cuda:0'


def get_gsm8k_dataset(split='test'): # split can be train
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset[split]

    question = [f"{example['question']}\n" for example in test_set]
    answer = []
    # get numerical answer
    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '') 
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)
    return question, answer



base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device=device, 
)

chat_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    device=device, 
)

# %%
train_questions, train_answers = get_gsm8k_dataset(split='train')
all_tokens = base_model.tokenizer(train_questions, padding=True, return_tensors="pt")

# %%
default_cfg = {
    "seed": 42,
    "batch_size": 100,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "qwen2_5_1_5B",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4,
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
