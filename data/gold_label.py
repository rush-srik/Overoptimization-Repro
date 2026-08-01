"""
Use the gold RM to score the completions of the policy
and create a pairwise preference dataset.
Writes 'train_proxy_rated' to disk.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, Dataset
from itertools import combinations
from collections import defaultdict

gold_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
scoring_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

gold_rm = AutoModelForSequenceClassification.from_pretrained(
    gold_name,
    device_map=device,
    dtype=torch.bfloat16,
)
gold_tokenizer = AutoTokenizer.from_pretrained(gold_name)

completions = load_from_disk("data/datasets/train_proxy_completions")

prompts = completions["prompt"]
responses = completions["response"]
entry_idx = completions["prompt_idx"]

input_texts = [
    gold_tokenizer.apply_chat_template(
        [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
        tokenize=False
    )
    for p, r in zip(prompts, responses)
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

rows_by_prompt = defaultdict(list)
for row, idx in enumerate(entry_idx):
    rows_by_prompt[idx].append(row)

gold_pairs = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

for rows in rows_by_prompt.values():
    for a, b in combinations(rows, 2):
        if scores[a] == scores[b]: # Skip ties
            continue
        chosen, rejected = (a, b) if scores[a] > scores[b] else (b, a)
        gold_pairs["prompt"].append(prompts[a])
        gold_pairs["chosen"].append(responses[chosen])
        gold_pairs["rejected"].append(responses[rejected])


print(f"[done] {len(gold_pairs["prompt"])} preference pairs recorded.")

gold_dataset = Dataset.from_dict(gold_pairs)
gold_dataset.save_to_disk("data/datasets/train_proxy_rated")