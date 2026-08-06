"""
Score the PPO val generations with the gold RM and add gold scores to the datasets.
"""

import os
import shutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

gold_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
gen_dir = "runs/ppo"
proxy_sizes = ["0.5B", "1.5B", "3B"]
steps = [0, 1, 2, 4, 8, 16, 32, 64, 116]

scoring_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

gold_rm = AutoModelForSequenceClassification.from_pretrained(
    gold_name,
    device_map=device,
    dtype=torch.bfloat16,
)
gold_tokenizer = AutoTokenizer.from_pretrained(gold_name)

def apply_gold_template(prompts, responses):
    return [
        gold_tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False
        )
        for p, r in zip(prompts, responses)
    ]

for proxy_size in proxy_sizes:
    for step in steps:
        print(f"\n[scoring] {proxy_size} step {step}")

        data_path = f"{gen_dir}/{proxy_size}/gen/step-{step}"
        if not os.path.exists(data_path):
            print(f"[skipping] {data_path} doesn't exist")
            continue

        generations = load_from_disk(data_path)
        if "gold_score" in generations.column_names:
            print(f"[skipping] {data_path} already scored")
            continue

        input_texts = apply_gold_template(generations["prompt"], generations["response"])

        scores = []
        for start in range(0, len(input_texts), scoring_batch_size):
            batch = input_texts[start:start+scoring_batch_size]
            inputs = gold_tokenizer(batch, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(gold_rm.device)
            with torch.inference_mode():
                scores.extend(gold_rm(**inputs).logits.squeeze(-1).float().tolist())

            if start % (scoring_batch_size*10) == 0:
                print(f"[scoring] {start}/{len(input_texts)}", end="\r")

        generations = generations.add_column("gold_score", scores)
        out = data_path
        tmp = out + ".tmp"

        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        generations.save_to_disk(tmp)
        del generations
        shutil.copy2(f"{out}/meta.json", f"{tmp}/meta.json")
        shutil.rmtree(out)
        os.rename(tmp, out)
