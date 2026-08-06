"""
Score the BoN val generations with the gold RM and add gold scores to the dataset.
Writes 'data/datasets/bon_scored' to disk.
"""

import os
import shutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

gold_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

scoring_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

gold_rm = AutoModelForSequenceClassification.from_pretrained(
    gold_name,
    device_map=device,
    dtype=torch.bfloat16,
)
gold_tokenizer = AutoTokenizer.from_pretrained(gold_name)

generations = load_from_disk("data/datasets/bon_completions")
input_texts = [
    gold_tokenizer.apply_chat_template(
        [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
        tokenize=False
    )
    for p, r in zip(generations["prompt"], generations["response"])
]

scores = []
for start in range(0, len(input_texts), scoring_batch_size):
    batch = input_texts[start:start+scoring_batch_size]
    inputs = gold_tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(gold_rm.device)
    with torch.inference_mode():
        scores.extend(gold_rm(**inputs).logits.squeeze(-1).float().tolist())

    if start % (scoring_batch_size*10) == 0:
        print(f"[scoring] {start}/{len(input_texts)}", end="\r")

generations = generations.add_column("gold_score", scores).save_to_disk("data/datasets/bon_scored")
