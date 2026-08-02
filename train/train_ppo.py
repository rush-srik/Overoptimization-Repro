"""
Train the policy using PPO with the proxy RM as the reward model.
"""

import torch
from copy import deepcopy
from trl.experimental import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_from_disk

device = "cuda" if torch.cuda.is_available() else "cpu"

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"

proxy_size = "14m"
seed = 42

val_ratio = 0.1
learning_rate = 5e-6
num_epochs = 1
micro_batch_size = 16
effective_batch_size = 64
eval_batch_size = 64

assert effective_batch_size % micro_batch_size == 0
grad_accum = effective_batch_size // micro_batch_size

max_response_length = 900

policy = AutoModelForCausalLM.from_pretrained(policy_name, device_map=device)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)
policy_tokenizer.padding_side = "left"

ref_model = deepcopy(policy)
value_model = AutoModelForSequenceClassification.from_pretrained(policy_name, num_labels=1, device_map=device)

proxy_rm = AutoModelForSequenceClassification.from_pretrained(f"models/proxy/{proxy_size}", device_map=device)

data = [entry["prompt"] for entry in load_from_disk("data/datasets/train_ppo")]

prompts = [policy_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in data]

ppo_data = [{"input_ids": policy_tokenizer(prompt)["input_ids"]} for prompt in prompts]

n = int(val_ratio * len(data))
train_data = ppo_data[n:]
val_data = ppo_data[:n]

config = PPOConfig(
    # Training
    kl_estimator="k3",
    kl_coef=0.0,
    learning_rate=learning_rate,
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=num_epochs,

    # Logging
    output_dir=f"runs/ppo/{proxy_size}/{seed}",
    report_to=["wandb"],
    seed=seed,

    # Misc
    response_length=max_response_length,
)

trainer = PPOTrainer(
    args=config,
    processing_class=policy_tokenizer,
    model=policy,
    ref_model=ref_model,
    value_model=value_model,
    reward_model=proxy_rm,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()
