"""
Select completions by proxy score and collect 
proxy vs. gold data for plotting.
Writes 'runs/bon/<proxy_size>/gen/n-<n>' to disk.
"""

import os
import json
import numpy as np
from scipy.special import gammaln
from datasets import load_from_disk, Dataset

proxy_sizes = ["0.5B-500", "0.5B-1k", "0.5B-4k", "0.5B-8k", "0.5B"]
n_points = [1, 2, 4, 8, 16, 32, 64, 128]
gold_path = "data/datasets/bon_scored"
out_dir = "runs/bon"

def subset_weights(N, n):
    """
    Probability that the sample at each rank is the argmax of a uniformly random
    n-subset of an N-sample pool: C(i-1, n-1) / C(N, n).
    """
    i = np.arange(1, N + 1)
    logw = (gammaln(i) - gammaln(np.maximum(i - n + 1, 1)) + np.log(n)
            - gammaln(N + 1) + gammaln(N - n + 1))
    return np.where(i >= n, np.exp(logw), 0.0)

gold = load_from_disk(gold_path)
prompt_idx = np.array(gold["prompt_idx"])

n_prompts = prompt_idx.max() + 1
N = len(prompt_idx) // n_prompts

assert np.array_equal(prompt_idx, np.repeat(np.arange(n_prompts), N)), \
    f"{gold_path} is not exactly N samples per prompt, in prompt order"

prompts = gold["prompt"][::N]
gold_scores = np.array(gold["gold_score"]).reshape(n_prompts, N)

print(f"[bon] {n_prompts} prompts x {N} samples")

for proxy_size in proxy_sizes:
    proxy_path = f"data/datasets/bon_scored_{proxy_size}"
    if not os.path.exists(proxy_path):
        print(f"[skipping] {proxy_path} doesn't exist")
        continue

    proxy = load_from_disk(proxy_path)
    assert np.array_equal(np.array(proxy["prompt_idx"]), prompt_idx), \
        f"{proxy_path} rows aren't aligned with {gold_path}"

    proxy_scores = np.array(proxy[f"{proxy_size}_score"]).reshape(n_prompts, N)

    order = np.argsort(proxy_scores, axis=1)
    proxy_sorted = np.take_along_axis(proxy_scores, order, axis=1)
    gold_sorted = np.take_along_axis(gold_scores, order, axis=1)

    for n in n_points:
        weights = subset_weights(N, n)
        gold_at_n = gold_sorted @ weights
        proxy_at_n = proxy_sorted @ weights
        kl = np.log(n) - (n - 1) / n

        out = f"{out_dir}/{proxy_size}/gen/n-{n}"
        Dataset.from_dict({
            "prompt": prompts,
            "gold_score": gold_at_n.tolist(),
            "proxy_score": proxy_at_n.tolist(),
        }).save_to_disk(out)

        with open(f"{out}/meta.json", "w") as f:
            json.dump({
                "n": n,
                "kl": float(kl),
                "gold_score_mean": float(gold_at_n.mean()),
                "proxy_score_mean": float(proxy_at_n.mean()),
                "gold_sd": float(gold_scores.std(1).mean()),
                "proxy_sd": float(proxy_scores.std(1).mean()),
                "proxy_size": proxy_size,
                "n_prompts": int(n_prompts),
            }, f, indent=2)

        print(f"[bon] {proxy_size} n={n} d={np.sqrt(kl):.3f} "
              f"gold={gold_at_n.mean():.4f} proxy={proxy_at_n.mean():.4f}")
