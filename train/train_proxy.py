"""
Trains a proxy reward model on the gold-labeled data.

Single GPU:
    python -m train.train_proxy <proxy_size> <micro_bs> <eval_bs> <lr> [grad_ckpt]

Multi-GPU:
    torchrun --standalone --nproc_per_node=N \
        -m train.train_proxy <proxy_size> <micro_bs> <eval_bs> <lr> [grad_ckpt]
"""

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
import random
import wandb
import sys

proxy_family_name = "Qwen/Qwen2.5"
proxy_size = sys.argv[1]
proxy_name = f"{proxy_family_name}-{proxy_size}"

project_name = "overopt"

val_ratio = 0.1
warmup_steps = 100
lr = float(sys.argv[4])
beta2 = 0.95
num_epochs = 1
micro_batch_size = int(sys.argv[2])
effective_batch_size = 64
eval_batch_size = int(sys.argv[3])
grad_ckpt = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

per_step = micro_batch_size * world_size
assert effective_batch_size % per_step == 0, (
    f"effective batch {effective_batch_size} not divisible by "
    f"micro_bs {micro_batch_size} x world_size {world_size}"
)
grad_accum = effective_batch_size // per_step

max_length = 2048
eval_steps = 200

seed = 42

tokenizer = AutoTokenizer.from_pretrained(proxy_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForSequenceClassification.from_pretrained(proxy_name, num_labels=1, dtype=torch.float32)
model.config.pad_token_id = tokenizer.pad_token_id

def apply_qwen_chat_template(entry):
    """
    Applies the Qwen chat template to a single entry in the dataset.
    """
    return {
        "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": entry["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        "chosen": entry["chosen"].strip(),
        "rejected": entry["rejected"].strip(),
    }

rated_data = load_from_disk("data/datasets/train_proxy_rated").remove_columns(["chosen_score", "rejected_score"])
rated_data = rated_data.map(apply_qwen_chat_template)

prompts = sorted(set(rated_data["prompt"]))
random.Random(seed).shuffle(prompts)
val_prompts = set(prompts[: int(val_ratio * len(prompts))])

val_data   = rated_data.filter(lambda ex: ex["prompt"] in val_prompts)
train_data = rated_data.filter(lambda ex: ex["prompt"] not in val_prompts)

fsdp = []
fsdp_config = {}
if world_size > 1:
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"

    fsdp = ["full_shard", "auto_wrap"]
    fsdp_config = {
        "version": 2,
        "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
        "activation_checkpointing": grad_ckpt,
    }

args = RewardConfig(
    # Training
    learning_rate=lr,
    adam_beta2=beta2,
    warmup_steps=warmup_steps,
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=num_epochs,

    # Sharding / activation memory
    fsdp=fsdp,
    fsdp_config=fsdp_config,
    gradient_checkpointing=grad_ckpt and world_size == 1,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,

    # Logging
    output_dir=f"runs/proxy/{proxy_size}",
    report_to=["wandb"],
    run_name=f"proxy-{proxy_size}",

    # Misc
    max_length=max_length,
    eos_token="<|im_end|>",
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="no",
    seed=seed,
)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data
)

if rank == 0:
    wandb.init(
        project=project_name,
        name=f"proxy-{proxy_size}",
        group="proxy-qwen",
        config={
            "lr": lr,
            "effective_batch": effective_batch_size,
            "micro_batch": micro_batch_size,
            "world_size": world_size,
            "grad_ckpt": grad_ckpt,
        }
    )

trainer.train()
trainer.save_model(f"models/proxy/{proxy_size}")