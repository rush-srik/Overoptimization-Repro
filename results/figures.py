"""
Fit a model to the gold scores to get a trend, make a table of alpha/beta
per proxy size, and generate a plot of the scores.
Writes 'coefficients.md' and 'gold_vs_kl.png' to disk.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datasets import load_from_disk

proxy_sizes = ["0.5B", "1.5B", "3B"]
steps = [0, 1, 2, 4, 8, 16, 32, 64, 116]
gen_dir = "runs/ppo"
out_dir = "results/figures"
n_bootstrap = 1000

rng = np.random.default_rng(0)

def trend(d, alpha, beta):
    return d * (alpha - beta * np.log(d))

gold, proxy, dists = {}, {}, {}
for proxy_size in proxy_sizes:
    g, p, kl = [], [], []
    for step in steps:
        data_path = f"{gen_dir}/{proxy_size}/gen/step-{step}"
        if not os.path.exists(data_path):
            print(f"[skipping] {data_path} doesn't exist")
            continue

        generations = load_from_disk(data_path)
        g.append(np.array(generations["gold_score"]))
        p.append(np.array(generations["proxy_score"]))
        with open(f"{data_path}/meta.json") as f:
            kl.append(json.load(f)["kl"])

    # Both RMs are only meaningful up to an offset, so pin them to 0 at the initial policy
    gold[proxy_size] = np.stack(g) - g[0].mean()
    proxy[proxy_size] = np.stack(p) - p[0].mean()
    dists[proxy_size] = np.sqrt(np.array(kl))

coefficients = {}
for proxy_size in proxy_sizes:
    d, scores = dists[proxy_size], gold[proxy_size]
    fit = d > 0 # Infinite slope initially

    alpha, beta = curve_fit(trend, d[fit], scores[fit].mean(1))[0]

    n_prompts = scores.shape[1]
    samples = []
    for _ in range(n_bootstrap):
        resampled = scores[:, rng.integers(0, n_prompts, n_prompts)].mean(1)
        try:
            samples.append(curve_fit(trend, d[fit], resampled[fit], p0=(alpha, beta))[0])
        except RuntimeError:
            continue
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)

    coefficients[proxy_size] = (alpha, beta, lo, hi)

os.makedirs(out_dir, exist_ok=True)

rows = ["| Proxy RM | alpha | beta |", "| --- | --- | --- |"]
for proxy_size in proxy_sizes:
    alpha, beta, lo, hi = coefficients[proxy_size]
    rows.append(
        f"| {proxy_size} "
        f"| {alpha:.4f} [{lo[0]:.4f}, {hi[0]:.4f}] "
        f"| {beta:.4f} [{lo[1]:.4f}, {hi[1]:.4f}] |"
    )
table = "\n".join(rows)

print(table)
with open(f"{out_dir}/coefficients.md", "w") as f:
    f.write(table + "\n")

colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(proxy_sizes)))

plt.figure(figsize=(7, 5))
for color, proxy_size in zip(colors, proxy_sizes):
    d = dists[proxy_size]
    alpha, beta, _, _ = coefficients[proxy_size]
    smooth = np.linspace(d[d > 0].min(), d.max(), 200)

    plt.plot(d, gold[proxy_size].mean(1), color=color, label=f"{proxy_size} gold")
    plt.plot(d, proxy[proxy_size].mean(1), color=color, ls="--", alpha=0.6,
             label=f"{proxy_size} proxy")
    plt.plot(smooth, trend(smooth, alpha, beta), color=color, lw=0.8, alpha=0.5)

plt.axhline(0, color="0.8", lw=0.8, zorder=0)
plt.xlabel(r"$\sqrt{D_{KL}(\pi \Vert \pi_{init})}$")
plt.ylabel("RM score")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{out_dir}/gold_vs_kl.png", dpi=200)
print(f"Wrote {out_dir}/gold_vs_kl.png")
