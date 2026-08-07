"""
Fit a model to the gold scores to get a trend, make a table of alpha/beta
per proxy size, and generate a plot of the scores.
Writes 'coefficients_<method>.md' and '<method>_data.png' to disk.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from datasets import load_from_disk

method = "bon"

methods = {
    "bon": {"gen_dir": "runs/bon", "point": "n",
            "points": [1, 2, 4, 8, 16, 32, 64, 128],
            "trend": lambda d, alpha, beta: d * (alpha - beta * d)},
    "ppo": {"gen_dir": "runs/ppo", "point": "step",
            "points": [0, 4, 10, 20, 30, 40, 50, 60],
            "trend": lambda d, alpha, beta: d * (alpha - beta * np.log(d))},
}

# We use a fixed proxy size and vary the amount of data.
proxy_sizes = ["0.5B-500", "0.5B-1k", "0.5B-4k", "0.5B-8k", "0.5B"]
gen_dir = methods[method]["gen_dir"]
points = methods[method]["points"]
point_label = methods[method]["point"]
trend = methods[method]["trend"]
out_dir = "results/figures"
n_bootstrap = 1000

rng = np.random.default_rng(0)

gold, proxy, dists = {}, {}, {}
for proxy_size in proxy_sizes:
    g, p, kl, sd = [], [], [], None
    for point in points:
        data_path = f"{gen_dir}/{proxy_size}/gen/{point_label}-{point}"
        if not os.path.exists(data_path):
            print(f"[skipping] {data_path} doesn't exist")
            continue

        generations = load_from_disk(data_path)
        g.append(np.array(generations["gold_score"]))
        p.append(np.array(generations["proxy_score"]))
        with open(f"{data_path}/meta.json") as f:
            meta = json.load(f)
        kl.append(meta["kl"])
        if sd is None:
            sd = (meta["gold_sd"], meta["proxy_sd"])

    if not g:
        continue

    # Standardize scores so they're comparable
    gold_sd, proxy_sd = sd
    gold[proxy_size] = (np.stack(g) - g[0].mean()) / gold_sd
    proxy[proxy_size] = (np.stack(p) - p[0].mean()) / proxy_sd
    dists[proxy_size] = np.sqrt(np.array(kl))

proxy_sizes = [s for s in proxy_sizes if s in gold]

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

rows = ["| Proxy RM | alpha | beta |"]
for proxy_size in proxy_sizes:
    alpha, beta, lo, hi = coefficients[proxy_size]
    rows.append(
        f"| {proxy_size} "
        f"| {alpha:.4f} [{lo[0]:.4f}, {hi[0]:.4f}] "
        f"| {beta:.4f} [{lo[1]:.4f}, {hi[1]:.4f}] |"
    )
table = "\n".join(rows)

print(table)
with open(f"{out_dir}/coefficients_{method}.md", "w") as f:
    f.write(table + "\n")

colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(proxy_sizes)))

def data_size(proxy_size):
    return "60k" if "-" not in proxy_size else proxy_size.split("-", 1)[1]

fig, ax = plt.subplots(figsize=(7, 5))
for color, proxy_size in zip(colors, proxy_sizes):
    d = dists[proxy_size]

    ax.plot(d, gold[proxy_size].mean(1), color=color, lw=1.8)
    ax.plot(d, proxy[proxy_size].mean(1), color=color, lw=1.8, ls="--", alpha=0.6)

size_legend = ax.legend(
    handles=[Line2D([], [], color=c, lw=1.8, label=data_size(s))
             for c, s in zip(colors, proxy_sizes)],
    title="Data Size", fontsize=8, title_fontsize=8,
    loc="upper left", bbox_to_anchor=(0.02, 0.98),
)
ax.add_artist(size_legend)  # a second legend() would replace the first

fig.canvas.draw()
box = size_legend.get_window_extent().transformed(ax.transAxes.inverted())
ax.legend(
    handles=[Line2D([], [], color="0.35", lw=1.8, label="Gold"),
             Line2D([], [], color="0.35", lw=1.8, ls="--", label="Proxy")],
    title="Reward", fontsize=8, title_fontsize=8,
    loc="upper left", bbox_to_anchor=(box.x1-0.02, 0.98),
)

ax.set_xlabel(r"sqrt(KL)")
ax.set_ylabel(r"RM score (standardized)")
fig.tight_layout()
fig.savefig(f"{out_dir}/{method}_data.png", dpi=200)
print(f"Wrote {out_dir}/{method}_data.png")
