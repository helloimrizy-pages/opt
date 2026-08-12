# OLMoE Expert Importance Across Domains

## Experimental setup

- Checkpoint: allenai/OLMoE-1B-7B-0924
- Resolved model revision: 6d84c48581ece794365f2b8e9cfb043c68ade9c5
- Device: NVIDIA A40 (cuda, bfloat16)
- Seed: 42
- Maximum sequence length: 512 tokens
- Bootstrap replicates: 100
- Reference answers included: True

The model was evaluated without generation or weight updates. Routing utilization, selected gate mass, and the L2 magnitude of each weighted expert output were collected in the same backbone forward pass. Functional contribution is an activation-magnitude proxy, not a causal intervention.

## Dataset and token summary

| Domain | Dataset | Split | Examples | Tokens | Mean tokens/example | Substitution |
|---|---|---:|---:|---:|---:|---|
| General | Salesforce/wikitext (wikitext-103-raw-v1) | test | 100 | 9653 | 96.5 | no |
| Math | openai/gsm8k (main) | test | 100 | 16392 | 163.9 | no |
| Coding | google-research-datasets/mbpp (full) | test | 100 | 18675 | 186.8 | no |
| Reasoning | allenai/ai2_arc (ARC-Challenge) | test | 100 | 7968 | 79.7 | no |

## Main cross-domain correlations

| Metric | Mean layer-wise Spearman across domain pairs | Mean top-25% Jaccard across domain pairs |
|---|---:|---:|
| Routing frequency | 0.227 | 0.259 |
| Gate mass | 0.273 | 0.268 |
| Functional contribution | 0.315 | 0.299 |

## Lowest-correlation domain pairs

| Pair | Functional Spearman | Kendall tau | Top-25% Jaccard |
|---|---:|---:|---:|
| General vs Coding | -0.079 | -0.047 | 0.131 |
| General vs Reasoning | 0.261 | 0.188 | 0.219 |
| General vs Math | 0.261 | 0.180 | 0.234 |
| Coding vs Reasoning | 0.314 | 0.231 | 0.332 |
| Math vs Coding | 0.507 | 0.370 | 0.461 |
| Math vs Reasoning | 0.623 | 0.465 | 0.416 |

## Layers with the strongest domain dependence

| MoE layer | Mean functional Spearman across pairs | Lowest pair | Lowest Spearman |
|---:|---:|---|---:|
| 12 | 0.200 | General vs Coding | -0.252 |
| 13 | 0.203 | General vs Coding | -0.336 |
| 5 | 0.209 | General vs Coding | -0.213 |
| 10 | 0.223 | General vs Coding | -0.237 |
| 14 | 0.245 | General vs Coding | -0.282 |
| 4 | 0.298 | General vs Coding | -0.040 |
| 0 | 0.305 | General vs Coding | 0.122 |
| 2 | 0.315 | General vs Coding | 0.022 |

## Domain-specialized expert examples

Specialization is based on normalized functional contribution. The reported ratio uses a 1e-12 numerical floor; the absolute range and rank change should be used to guard against ratios driven by tiny denominators.

| Layer | Expert | Maximum domain | Minimum domain | Ratio | Normalized range | Rank range |
|---:|---:|---|---|---:|---:|---:|
| 1 | 25 | Coding | General | 60.01 | 0.1683 | 63.0 |
| 3 | 43 | Coding | Reasoning | 147.67 | 0.1211 | 62.0 |
| 13 | 52 | General | Coding | 1473.76 | 0.0928 | 62.0 |
| 11 | 48 | Reasoning | Coding | 1224.07 | 0.0801 | 62.0 |
| 7 | 43 | General | Coding | 603.77 | 0.0566 | 62.0 |
| 15 | 49 | General | Coding | ∞ (zero minimum) | 0.0424 | 62.0 |
| 11 | 27 | Coding | Reasoning | 141.33 | 0.1983 | 61.0 |
| 6 | 52 | Coding | General | 198.23 | 0.1648 | 61.0 |
| 7 | 2 | Coding | General | 177.34 | 0.1563 | 61.0 |
| 5 | 42 | Coding | Reasoning | 430.85 | 0.1551 | 61.0 |
| 9 | 46 | Coding | Reasoning | 237.55 | 0.1338 | 61.0 |
| 14 | 6 | Coding | Reasoning | 372.53 | 0.1207 | 61.0 |

## Routing frequency versus functional contribution

| Domain | Mean within-domain Spearman across layers |
|---|---:|
| General | 0.888 |
| Math | 0.916 |
| Coding | 0.944 |
| Reasoning | 0.911 |

Functional-contribution rankings were more stable across domains than routing frequency by 0.088 mean Spearman.

## Bootstrap uncertainty

Examples were resampled independently within each domain from the stored per-example expert vectors; the model was not rerun. Aggregate intervals are for the mean layer-wise Spearman in each domain pair.

| Domain pair | Observed | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| General vs Coding | -0.079 | -0.073 | [-0.126, -0.024] |
| General vs Reasoning | 0.261 | 0.255 | [0.185, 0.318] |
| General vs Math | 0.261 | 0.267 | [0.202, 0.343] |
| Coding vs Reasoning | 0.314 | 0.315 | [0.277, 0.344] |
| Math vs Coding | 0.507 | 0.507 | [0.487, 0.527] |
| Math vs Reasoning | 0.623 | 0.618 | [0.587, 0.647] |

## Limitations

- The functional metric is a weighted-output norm. It does not measure the loss increase caused by removing or perturbing an expert.
- Domain datasets differ in style, sequence length, and availability of reference answers. Token normalization reduces but does not remove these confounds.
- OLMoE's selected top-k weights are used exactly as implemented; when norm_topk_prob=false, selected weights do not sum to one.
- Bootstrap intervals capture example-sampling uncertainty for these corpora, not model, checkpoint, prompt-format, or dataset-choice uncertainty.
- This is one base checkpoint. Conclusions should be checked on another MoE model and with intervention-based expert ablations before guiding compression.

# Go / No-Go Assessment

**GO: strong support.** 70/96 layer-pair functional correlations were below 0.5; mean functional Spearman was 0.315, and mean top-25% Jaccard was 0.299. The observed disagreement is broad enough to justify testing distributionally robust allocation.

This result is evidence about distribution-conditioned routing utilization and an activation-magnitude proxy. It is not yet evidence that a specific mixed-precision allocation improves downstream quality. The next stage should preserve this diagnostic boundary: validate selected high-disagreement layers with expert ablation or masking before implementing quantization.
