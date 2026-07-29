"""
Generate completions to the proxy training data using the policy, then rate them
using the gold reward model and create a pairwise preference dataset for proxy RM training.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from vllm import LLM, SamplingParams

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"
gold_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

K = 4 # completions per prompt
max_continuation_length = 900

device = "cuda" if torch.cuda.is_available() else "cpu"

sampling_params = SamplingParams(max_tokens=max_continuation_length, n=K, temperature=1.0)
policy = LLM(model=policy_name)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)

gold_rm = AutoModelForSequenceClassification.from_pretrained(gold_name, num_labels=1)
gold_tokenizer = AutoTokenizer.from_pretrained(gold_name)

gold_labels = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

data = [entry["prompt"] for entry in load_from_disk("data/datasets/train_proxy")]
prompts = [policy_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in data]

outputs = policy.generate(prompts, sampling_params)


