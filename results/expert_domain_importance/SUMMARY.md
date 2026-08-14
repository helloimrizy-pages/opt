# OLMoE Expert Importance Across Domains

## Experimental setup

- Checkpoint: allenai/OLMoE-1B-7B-0924
- Resolved model revision: 6d84c48581ece794365f2b8e9cfb043c68ade9c5
- Device: CPU (arm) (cpu, bfloat16)
- Seed: 42
- Maximum sequence length: 128 tokens
- Bootstrap replicates: 100
- Reference answers included: True

The model was evaluated without generation or weight updates. Routing utilization, selected gate mass, and the L2 magnitude of each weighted expert output were collected in the same backbone forward pass. Functional contribution is an activation-magnitude proxy, not a causal intervention.

## Dataset and token summary

| Domain | Dataset | Split | Examples | Tokens | Mean tokens/example | Substitution |
|---|---|---:|---:|---:|---:|---|
| General | Salesforce/wikitext (wikitext-103-raw-v1) | test | 5 | 482 | 96.4 | no |
| Math | openai/gsm8k (main) | test | 5 | 559 | 111.8 | no |
| Coding | google-research-datasets/mbpp (full) | test | 5 | 640 | 128.0 | no |
| Reasoning | allenai/ai2_arc (ARC-Challenge) | test | 5 | 432 | 86.4 | no |

**Length warning:** The mean token count reached the configured maximum for Coding. This indicates widespread truncation and should be revisited at the intended 512-token cap.

## Main cross-domain correlations

| Metric | Mean layer-wise Spearman across domain pairs | Mean top-25% Jaccard across domain pairs |
|---|---:|---:|
| Routing frequency | 0.322 | 0.326 |
| Gate mass | 0.350 | 0.323 |
| Functional contribution | 0.378 | 0.343 |

## Lowest-correlation domain pairs

| Pair | Functional Spearman | Kendall tau | Top-25% Jaccard |
|---|---:|---:|---:|
| General vs Coding | 0.028 | 0.025 | 0.201 |
| General vs Reasoning | 0.318 | 0.233 | 0.296 |
| General vs Math | 0.343 | 0.246 | 0.287 |
| Coding vs Reasoning | 0.388 | 0.291 | 0.378 |
| Math vs Coding | 0.492 | 0.360 | 0.405 |
| Math vs Reasoning | 0.696 | 0.519 | 0.493 |

## Layers with the strongest domain dependence

| MoE layer | Mean functional Spearman across pairs | Lowest pair | Lowest Spearman |
|---:|---:|---|---:|
| 0 | 0.286 | General vs Coding | -0.004 |
| 12 | 0.318 | General vs Coding | 0.056 |
| 10 | 0.320 | General vs Coding | -0.098 |
| 13 | 0.323 | General vs Coding | -0.047 |
| 5 | 0.326 | General vs Coding | 0.036 |
| 2 | 0.330 | General vs Coding | -0.030 |
| 4 | 0.344 | General vs Coding | -0.045 |
| 1 | 0.357 | General vs Coding | 0.099 |

## Domain-specialized expert examples

Specialization is based on normalized functional contribution. The reported ratio uses a 1e-12 numerical floor; the absolute range and rank change should be used to guard against ratios driven by tiny denominators.

| Layer | Expert | Maximum domain | Minimum domain | Ratio | Normalized range | Rank range |
|---:|---:|---|---|---:|---:|---:|
| 1 | 25 | Coding | General | 117.02 | 0.1661 | 63.0 |
| 11 | 27 | Coding | General | 266.88 | 0.1926 | 62.0 |
| 9 | 46 | Coding | General | ∞ (zero minimum) | 0.1308 | 62.0 |
| 14 | 6 | Coding | Reasoning | ∞ (zero minimum) | 0.1264 | 62.0 |
| 3 | 43 | Coding | Reasoning | 570.10 | 0.1238 | 62.0 |
| 12 | 38 | Coding | Reasoning | 2146.56 | 0.1178 | 62.0 |
| 8 | 3 | Coding | Reasoning | 1064.10 | 0.1057 | 62.0 |
| 0 | 21 | Coding | General | 162.14 | 0.0928 | 62.0 |
| 13 | 52 | General | Math | ∞ (zero minimum) | 0.0525 | 62.0 |
| 7 | 44 | General | Coding | ∞ (zero minimum) | 0.0431 | 62.0 |
| 10 | 40 | General | Coding | ∞ (zero minimum) | 0.0526 | 61.5 |
| 5 | 25 | General | Math | ∞ (zero minimum) | 0.0452 | 61.5 |

## Routing frequency versus functional contribution

| Domain | Mean within-domain Spearman across layers |
|---|---:|
| General | 0.886 |
| Math | 0.901 |
| Coding | 0.946 |
| Reasoning | 0.910 |

Functional-contribution rankings were more stable across domains than routing frequency by 0.056 mean Spearman.

## Bootstrap uncertainty

Examples were resampled independently within each domain from the stored per-example expert vectors; the model was not rerun. Aggregate intervals are for the mean layer-wise Spearman in each domain pair.

| Domain pair | Observed | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| General vs Coding | 0.028 | 0.046 | [-0.138, 0.295] |
| General vs Reasoning | 0.318 | 0.288 | [0.157, 0.441] |
| General vs Math | 0.343 | 0.309 | [0.154, 0.440] |
| Coding vs Reasoning | 0.388 | 0.376 | [0.289, 0.462] |
| Math vs Coding | 0.492 | 0.485 | [0.430, 0.540] |
| Math vs Reasoning | 0.696 | 0.653 | [0.567, 0.724] |

## Limitations

- The functional metric is a weighted-output norm. It does not measure the loss increase caused by removing or perturbing an expert.
- Domain datasets differ in style, sequence length, and availability of reference answers. Token normalization reduces but does not remove these confounds.
- At least one domain reached the mean sequence-length cap in this run; truncation can itself change the observed domain ranking.
- OLMoE's selected top-k weights are used exactly as implemented; when norm_topk_prob=false, selected weights do not sum to one.
- Bootstrap intervals capture example-sampling uncertainty for these corpora, not model, checkpoint, prompt-format, or dataset-choice uncertainty.
- This is one base checkpoint. Conclusions should be checked on another MoE model and with intervention-based expert ablations before guiding compression.

# Go / No-Go Assessment

**NO DECISION: preliminary strong support.** 70/96 layer-pair functional correlations were below 0.5; mean functional Spearman was 0.378, and mean top-25% Jaccard was 0.343. The observed disagreement is broad enough to justify testing distributionally robust allocation. However, the smallest domain has only 5 examples, below the prescribed 100-example quick baseline; the heuristic decision is withheld.

This result is evidence about distribution-conditioned routing utilization and an activation-magnitude proxy. It is not yet evidence that a specific mixed-precision allocation improves downstream quality. The next stage should preserve this diagnostic boundary: validate selected high-disagreement layers with expert ablation or masking before implementing quantization.

> This was a reduced 5-example/domain diagnostic, below the prescribed 100-example quick baseline. It validates the pipeline and provides a preliminary signal, not a go/no-go result.
