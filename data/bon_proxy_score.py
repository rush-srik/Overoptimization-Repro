"""
Score the BoN val generations with the proxy RM with specified 'proxy_size' 
and add its scores to the dataset.
Writes 'data/datasets/bon_scored_{proxy_size}' to disk.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

proxy_size = sys.argv[1]
scoring_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(f"models/proxy/{proxy_size}", dtype=torch.float32, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(f"models/proxy/{proxy_size}")

generations = load_from_disk("data/datasets/bon_completions")
input_texts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True
    ) + r.strip() + tokenizer.eos_token
    for p, r in zip(generations["prompt"], generations["response"])
]

scores = []
for start in range(0, len(input_texts), scoring_batch_size):
    batch = input_texts[start:start+scoring_batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        scores.extend(model(**inputs).logits.squeeze(-1).float().tolist())

    if start % (scoring_batch_size*10) == 0:
        print(f"[scoring] {start}/{len(input_texts)}", end="\r")

generations = generations.add_column(f"{proxy_size}_score", scores).save_to_disk(f"data/datasets/bon_scored_{proxy_size}")