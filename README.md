# Reproduction of Scaling Laws for Reward Model Overoptimization

This is a reproduction of [Gao et al., *Scaling Laws for Reward Model
Overoptimization*](https://arxiv.org/abs/2210.10760) (arXiv:2210.10760), using
open models and training proxies on different data sizes.

![Gold and proxy reward vs. sqrt(KL), for proxy RMs trained on 500 to 60k
comparisons](results/figures/bon_data.png)

## How it works

A "gold" reward model stands in for ground truth. Policy rollouts are scored
with it to form pairwise preferences, and proxy reward models are trained on
those preferences, with each one having seen a different amount of data.

Best-of-N (BoN) then uses the proxy reward to score several completions to a
given prompt and picks the highest-scoring one. The plot above shows proxy vs.
gold scores for that top completion, at each data size.

The x-axis is KL on a square-root scale, which grows with the number of
completions sampled and represents the distance from the base model's outputs to
the selected ones.

## Functional form

This reproduction also captures the functional form of the gold reward presented
in the paper:

$$R_{\text{bon}}(d) = d\,(\alpha_{\text{bon}} - \beta_{\text{bon}}\,d)$$

where $d$ is $\sqrt{\mathrm{KL}}$, and $\alpha_{\text{bon}}$ and
$\beta_{\text{bon}}$ are parameters that depend on the configuration.

## Results

The figure shows that proxy rewards continue to increase at a similar rate as
KL-distance increases, but a gap begins to form between the proxy and gold
scores, with gold scores eventually starting to fall (visible in proxy with 500
pairs); this is overoptimization.

For the data sizes tested, this reproduction finds the following coefficients:

| Data size | $\alpha_{\text{bon}}$ | $\beta_{\text{bon}}$ |
| --- | --- | --- |
| 500 | 0.386 | 0.136 |
| 1k | 0.461 | 0.123 |
| 4k | 0.616 | 0.121 |
| 8k | 0.741 | 0.118 |
| 60k | 0.913 | 0.122 |

The peak of the gold reward sits at $d=\frac{\alpha}{2\beta}$, so this table
shows that using more data delays overoptimization.

Note that these coefficients are in standardized units and are therefore not
comparable to those presented in the paper.

## Setup

| Role | Model |
| --- | --- |
| Policy | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Gold RM | [`Skywork/Skywork-Reward-V2-Llama-3.1-8B`](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B) |
| Proxy RM | [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| Feedback dataset | [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |

For BoN, completions were sampled from a dataset of 1,280 held-out prompts, and
the following sizes of n were used: `[1, 2, 4, 8, 16, 32, 64, 128]`

Like in the paper, KL was also computed analytically with the formula
$\text{KL}=\log n - \frac{n-1}{n}$.

## Limitations

1. This reproduction uses two different model families, Qwen and Skywork-Reward.
   The original paper used GPT-3 exclusively.
2. The policy is instruct-tuned, but the paper's was only SFT.
3. This reproduction does not test PPO yet, but I am working on adding that
   method.
