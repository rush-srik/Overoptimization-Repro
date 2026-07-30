from datasets import load_from_disk
from trl import RewardTrainer, RewardConfig
import random
from utils.utils import apply_proxy_chat_template

proxy_family_name = "EleutherAI/pythia"
proxy_size = "14m"
proxy_name = f"{proxy_family_name}-{proxy_size}"

val_ratio = 0.2
seed = 42

rated_data = load_from_disk("data/datasets/train_proxy_rated")
rated_data = rated_data.map(apply_proxy_chat_template)

prompts = sorted(set(rated_data["prompt"]))
random.Random(seed).shuffle(prompts)
val_prompts = set(prompts[: int(val_ratio * len(prompts))])

val_data   = rated_data.filter(lambda ex: ex["prompt"] in val_prompts)
train_data = rated_data.filter(lambda ex: ex["prompt"] not in val_prompts)

args = RewardConfig(
    output_dir=f"runs/proxy/{proxy_size}",
    max_length=2048,
    eval_strategy="steps",
    eval_steps=100,
    report_to=["wandb"],
    run_name=f"proxy-{proxy_size}",
)

trainer = RewardTrainer(
    model=proxy_name,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()
