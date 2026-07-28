from datasets import load_dataset, Dataset

dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
split_name = "prefs"
proxy_ratio = 0.2

train_data = load_dataset(dataset_name, split=f"train_{split_name}")
val_data = load_dataset(dataset_name, split=f"test_{split_name}")

n = int(proxy_ratio*len(train_data))
extracted_train_proxy = {
    "prompts": list(dict.fromkeys(entry["prompt"] for entry in train_data.select(range(n))))
}
extracted_train_ppo = {
    "prompts": list(dict.fromkeys(entry["prompt"] for entry in train_data.select(range(n, len(train_data)))))
}

extracted_val = {
    "prompts": list(dict.fromkeys(entry["prompt"] for entry in val_data))
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