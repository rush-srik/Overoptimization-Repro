"""
Split the base dataset into proxy training data, PPO training data, and validation data.
Extract prompts only and filter out prompts greater than 'max_prompt_len.'
Writes 'train_proxy,' 'train_ppo,' and 'val' to disk.
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
proxy_name = "EleutherAI/pythia-70m" # For tokenizer
split_name = "prefs"
proxy_ratio = 0.2
max_prompt_len = 1024

tokenizer = AutoTokenizer.from_pretrained(proxy_name)

train_data = load_dataset(dataset_name, split=f"train_{split_name}")
val_data = load_dataset(dataset_name, split=f"test_{split_name}")

n = int(proxy_ratio*len(train_data))
proxy_data = train_data.select(range(n))
ppo_data = train_data.select(range(n, len(train_data)))

extracted_train_proxy = {
    "prompt": list(dict.fromkeys(entry["prompt"] for entry in proxy_data if len(tokenizer(entry["prompt"])["input_ids"]) <= max_prompt_len))
}
extracted_train_ppo = {
    "prompt": list(dict.fromkeys(entry["prompt"] for entry in ppo_data if len(tokenizer(entry["prompt"])["input_ids"]) <= max_prompt_len))
}

extracted_val = {
    "prompt": list(dict.fromkeys(entry["prompt"] for entry in val_data if len(tokenizer(entry["prompt"])["input_ids"]) <= max_prompt_len))
}

train_proxy_ds = Dataset.from_dict(extracted_train_proxy)
train_ppo_ds = Dataset.from_dict(extracted_train_ppo)
val_ds = Dataset.from_dict(extracted_val)

print(f"Proxy training dataset length: {len(train_proxy_ds)}")
print(f"PPO training dataset length: {len(train_ppo_ds)}")
print(f"Val dataset length: {len(val_ds)}")

train_proxy_ds.save_to_disk("data/datasets/train_proxy")
train_ppo_ds.save_to_disk("data/datasets/train_ppo")
val_ds.save_to_disk("data/datasets/val")