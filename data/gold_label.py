"""
Generate completions to the proxy training data using the policy, then rate them
using the gold reward model and create a pairwise preference dataset for proxy RM training.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams
from itertools import combinations
from collections import defaultdict

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"
gold_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

K = 4 # completions per prompt
scoring_batch_size = 64
max_continuation_length = 900

device = "cuda" if torch.cuda.is_available() else "cpu"

sampling_params = SamplingParams(max_tokens=max_continuation_length, n=K, temperature=1.0)
policy = LLM(model=policy_name, gpu_memory_utilization=0.8)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)

gold_rm = AutoModelForSequenceClassification.from_pretrained(
    gold_name,
    device_map="auto",
    dtype=torch.bfloat16,
)
gold_tokenizer = AutoTokenizer.from_pretrained(gold_name)

data = [entry["prompt"] for entry in load_from_disk("data/datasets/train_proxy")]
prompts = [policy_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in data]

outputs = policy.generate(prompts, sampling_params)

entries = []

for out, prompt in zip(outputs, data):

    entry = {
        "prompt": prompt,
        "responses": []
    }

    for response in out.outputs:
        if response.finish_reason == "length":
            continue
        entry["responses"].append(response.text)

    entries.append(entry)

input_texts, entry_idx = [], []
for i, entry in enumerate(entries):
    for response in entry["responses"]:
        conv = [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": response}
        ]
        input_texts.append(gold_tokenizer.apply_chat_template(conv, tokenize=False))
        entry_idx.append(i)

scores = []
for start in range(0, len(input_texts), scoring_batch_size):
    batch = input_texts[start:start+scoring_batch_size]
    inputs = gold_tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(gold_rm.device)
    with torch.inference_mode():
        scores.extend(gold_rm(**inputs).logits.squeeze(-1).float().tolist())

    if start % (scoring_batch_size*10) == 0:
        print(f"[scoring] {start}/{len(input_texts)}", end="\r")

score_dict = defaultdict(list)
for idx, score in zip(entry_idx, scores):
    score_dict[idx].append(score)

gold_pairs = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

for idx in score_dict:
    for pair in combinations(range(len(score_dict[idx])), 2):
        gold_pairs["prompt"].append(entries[idx]["prompt"])

        chosen_idx, rejected_idx = sorted(pair, key=lambda p: score_dict[idx][p], reverse=True)
        gold_pairs["chosen"].append(entries[idx]["responses"][chosen_idx])
        gold_pairs["rejected"].append(entries[idx]["responses"][rejected_idx])

print(f"[done] {len(gold_pairs["prompt"])} preference pairs recorded.")

gold_dataset = Dataset.from_dict(gold_pairs)
gold_dataset.save_to_disk("data/datasets/train_proxy_rated")