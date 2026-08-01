"""
Trains a proxy reward model on the gold-labeled data.

Run with: 'python -m train.train_proxy <proxy_size> <micro_batch_size> <eval_batch_size>'
"""

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
import random
from utils.utils import apply_proxy_chat_template
import wandb
import sys

proxy_family_name = "EleutherAI/pythia"
proxy_size = sys.argv[1]
proxy_name = f"{proxy_family_name}-{proxy_size}"

project_name = "overopt"

val_ratio = 0.2
lr = 5e-5
micro_batch_size = int(sys.argv[2])
effective_batch_size = 64
eval_batch_size = int(sys.argv[3])

assert effective_batch_size % micro_batch_size == 0
grad_accum = effective_batch_size // micro_batch_size

max_length = 2048
eval_steps = 200
save_steps = 200

seed = 42

tokenizer = AutoTokenizer.from_pretrained(proxy_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForSequenceClassification.from_pretrained(proxy_name, num_labels=1, dtype=torch.float32)
model.config.pad_token_id = tokenizer.pad_token_id

rated_data = load_from_disk("data/datasets/train_proxy_rated")
rated_data = rated_data.map(apply_proxy_chat_template)

prompts = sorted(set(rated_data["prompt"]))
random.Random(seed).shuffle(prompts)
val_prompts = set(prompts[: int(val_ratio * len(prompts))])

val_data   = rated_data.filter(lambda ex: ex["prompt"] in val_prompts)
train_data = rated_data.filter(lambda ex: ex["prompt"] not in val_prompts)

args = RewardConfig(
    # Training
    learning_rate=lr,
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum,

    # Logging
    output_dir=f"runs/proxy/{proxy_size}",
    report_to=["wandb"],
    run_name=f"proxy-{proxy_size}",

    # Misc
    max_length=max_length,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_steps=save_steps,
    seed=seed,
)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data
)

wandb.init(
    project=project_name,
    name=f"proxy-{proxy_size}",
    group="proxy",
    config={"lr": lr, "effective_batch": effective_batch_size}
)

trainer.train()
trainer.save_model(f"models/proxy/{proxy_size}")