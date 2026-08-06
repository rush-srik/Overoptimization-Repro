"""
Generate completions to the val prompts, 'N' per prompt.
Used for BoN.
"""

import torch
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk
from vllm import LLM, SamplingParams

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"
N = 64
n_prompts = 1280
max_continuation_length = 900
seed = 42

sampling_params = SamplingParams(max_tokens=max_continuation_length, n=N, temperature=1.0)
policy = LLM(model=policy_name, gpu_memory_utilization=0.8, seed=seed)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)

data = [entry["prompt"] for entry in load_from_disk("data/datasets/val").select(range(n_prompts))]
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
        entry["responses"].append(response.text)

    entries.append(entry)

# Flatten for ease of gold labeling
completions = {
    "prompt_idx": [],
    "prompt": [],
    "response": []
}
for i, entry in enumerate(entries):
    for response in entry["responses"]:
        completions["prompt_idx"].append(i)
        completions["prompt"].append(entry["prompt"])
        completions["response"].append(response)

Dataset.from_dict(completions).save_to_disk("data/datasets/bon_completions")
