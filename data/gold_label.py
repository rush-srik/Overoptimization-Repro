"""
Generate completions to the proxy training data using the policy, then rate them
using the gold reward model and create a pairwise preference dataset for proxy RM training.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams
import itertools

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"
gold_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

K = 4 # completions per prompt
max_continuation_length = 900

device = "cuda" if torch.cuda.is_available() else "cpu"

sampling_params = SamplingParams(max_tokens=max_continuation_length, n=K, temperature=1.0)
policy = LLM(model=policy_name, gpu_memory_utilization=0.8)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)

gold_rm = AutoModelForSequenceClassification.from_pretrained(
    gold_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    num_labels=1
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

gold_pairs = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

for i, entry in enumerate(entries):

    scores = []
    for response in entry["responses"]:
        conv = [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": response}
        ]
        text = gold_tokenizer.apply_chat_template(conv, tokenize=False)
        inputs = gold_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(gold_rm.device)

        with torch.no_grad():
            scores.append((response, gold_rm(**inputs).logits[0][0].item()))

    for pair in itertools.combinations(scores, 2):
        gold_pairs["prompt"].append(entry["prompt"])

        chosen_idx = max([0, 1], key=lambda idx: pair[idx][1])
        gold_pairs["chosen"].append(pair[chosen_idx][0])
        gold_pairs["rejected"].append(pair[1-chosen_idx][0])

    print(f"[scoring] {i}/{len(entries)}", end="\r")

print(f"[done] {len(gold_pairs)} preference pairs recorded.")

gold_dataset = Dataset.from_dict(gold_pairs)
gold_dataset.save_to_disk("data/datasets/train_proxy_rated")