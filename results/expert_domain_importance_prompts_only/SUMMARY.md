# OLMoE Expert Importance Across Domains

## Experimental setup

- Checkpoint: allenai/OLMoE-1B-7B-0924
- Resolved model revision: 6d84c48581ece794365f2b8e9cfb043c68ade9c5
- Device: NVIDIA A40 (cuda, bfloat16)
- Seed: 42
- Maximum sequence length: 512 tokens
- Bootstrap replicates: 100
- Reference answers included: False

The model was evaluated without generation or weight updates. Routing utilization, selected gate mass, and the L2 magnitude of each weighted expert output were collected in the same backbone forward pass. Functional contribution is an activation-magnitude proxy, not a causal intervention.

## Dataset and token summary

| Domain | Dataset | Split | Examples | Tokens | Mean tokens/example | Substitution |
|---|---|---:|---:|---:|---:|---|
| General | Salesforce/wikitext (wikitext-103-raw-v1) | test | 100 | 9653 | 96.5 | no |
| Math | openai/gsm8k (main) | test | 100 | 6425 | 64.2 | no |
| Coding | google-research-datasets/mbpp (full) | test | 100 | 10841 | 108.4 | no |
| Reasoning | allenai/ai2_arc (ARC-Challenge) | test | 100 | 7868 | 78.7 | no |

## Main cross-domain correlations

| Metric | Mean layer-wise Spearman across domain pairs | Mean top-25% Jaccard across domain pairs |
|---|---:|---:|
| Routing frequency | 0.233 | 0.238 |
| Gate mass | 0.277 | 0.247 |
| Functional contribution | 0.323 | 0.277 |

## Lowest-correlation domain pairs

| Pair | Functional Spearman | Kendall tau | Top-25% Jaccard |
|---|---:|---:|---:|
| General vs Coding | -0.046 | -0.026 | 0.134 |
| General vs Reasoning | 0.263 | 0.189 | 0.219 |
| Math vs Coding | 0.294 | 0.215 | 0.283 |
| Coding vs Reasoning | 0.339 | 0.252 | 0.316 |
| General vs Math | 0.384 | 0.273 | 0.284 |
| Math vs Reasoning | 0.707 | 0.533 | 0.428 |

## Layers with the strongest domain dependence

| MoE layer | Mean functional Spearman across pairs | Lowest pair | Lowest Spearman |
|---:|---:|---|---:|
| 5 | 0.170 | General vs Coding | -0.205 |
| 13 | 0.226 | General vs Coding | -0.296 |
| 12 | 0.228 | General vs Coding | -0.182 |
| 10 | 0.246 | General vs Coding | -0.208 |
| 14 | 0.256 | General vs Coding | -0.193 |
| 4 | 0.303 | General vs Coding | -0.012 |
| 2 | 0.314 | General vs Coding | -0.000 |
| 0 | 0.319 | General vs Reasoning | 0.141 |

## Domain-specialized expert examples

Specialization is based on normalized functional contribution. The reported ratio uses a 1e-12 numerical floor; the absolute range and rank change should be used to guard against ratios driven by tiny denominators.

| Layer | Expert | Maximum domain | Minimum domain | Ratio | Normalized range | Rank range |
|---:|---:|---|---|---:|---:|---:|
| 13 | 52 | General | Coding | 11069.04 | 0.0928 | 63.0 |
| 1 | 25 | Coding | General | 50.71 | 0.1418 | 62.0 |
| 15 | 49 | General | Coding | ∞ (zero minimum) | 0.0424 | 62.0 |
| 11 | 27 | Coding | Reasoning | 158.70 | 0.1761 | 61.0 |
| 5 | 42 | Coding | Reasoning | 389.09 | 0.1394 | 61.0 |
| 13 | 25 | Coding | Math | 466.87 | 0.1121 | 61.0 |
| 14 | 6 | Coding | Reasoning | 339.97 | 0.1109 | 61.0 |
| 11 | 13 | Coding | Reasoning | 1003.79 | 0.1059 | 61.0 |
| 12 | 38 | Coding | Math | 442.06 | 0.1053 | 61.0 |
| 3 | 43 | Coding | Reasoning | 123.34 | 0.1024 | 61.0 |
| 11 | 48 | Reasoning | Coding | 1894.74 | 0.0808 | 61.0 |
| 5 | 2 | Coding | Math | 341.63 | 0.0764 | 61.0 |

## Routing frequency versus functional contribution

| Domain | Mean within-domain Spearman across layers |
|---|---:|
| General | 0.888 |
| Math | 0.897 |
| Coding | 0.943 |
| Reasoning | 0.911 |

Functional-contribution rankings were more stable across domains than routing frequency by 0.090 mean Spearman.

## Bootstrap uncertainty

Examples were resampled independently within each domain from the stored per-example expert vectors; the model was not rerun. Aggregate intervals are for the mean layer-wise Spearman in each domain pair.

| Domain pair | Observed | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| General vs Coding | -0.046 | -0.040 | [-0.089, 0.010] |
| General vs Reasoning | 0.263 | 0.256 | [0.186, 0.319] |
| Math vs Coding | 0.294 | 0.295 | [0.263, 0.322] |
| Coding vs Reasoning | 0.339 | 0.340 | [0.296, 0.378] |
| General vs Math | 0.384 | 0.380 | [0.329, 0.438] |
| Math vs Reasoning | 0.707 | 0.701 | [0.676, 0.729] |

## Limitations

- The functional metric is a weighted-output norm. It does not measure the loss increase caused by removing or perturbing an expert.
- Domain datasets differ in style, sequence length, and availability of reference answers. Token normalization reduces but does not remove these confounds.
- OLMoE's selected top-k weights are used exactly as implemented; when norm_topk_prob=false, selected weights do not sum to one.
- Bootstrap intervals capture example-sampling uncertainty for these corpora, not model, checkpoint, prompt-format, or dataset-choice uncertainty.
- This is one base checkpoint. Conclusions should be checked on another MoE model and with intervention-based expert ablations before guiding compression.

# Go / No-Go Assessment

**GO: strong support.** 75/96 layer-pair functional correlations were below 0.5; mean functional Spearman was 0.323, and mean top-25% Jaccard was 0.277. The observed disagreement is broad enough to justify testing distributionally robust allocation.

This result is evidence about distribution-conditioned routing utilization and an activation-magnitude proxy. It is not yet evidence that a specific mixed-precision allocation improves downstream quality. The next stage should preserve this diagnostic boundary: validate selected high-disagreement layers with expert ablation or masking before implementing quantization.
