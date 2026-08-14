# OLMoE Expert Importance Across Domains

## Experimental setup

- Checkpoint: allenai/OLMoE-1B-7B-0924
- Resolved model revision: 6d84c48581ece794365f2b8e9cfb043c68ade9c5
- Device: NVIDIA A40 (cuda, bfloat16)
- Seed: 42
- Maximum sequence length: 512 tokens
- Bootstrap replicates: 100
- Reference answers included: False
- Prompt style: neutral_fixed_token_control
- Shared neutral prefix: `Input:\n`
- Measured source positions per example: 64
- Look-ahead next-token labels per example: 1
- Exact measured token budget per domain: 6400

The model was evaluated without generation or weight updates. Routing utilization, selected gate mass, and the L2 magnitude of each weighted expert output were collected in the same backbone forward pass. Functional contribution is an activation-magnitude proxy, not a causal intervention.

## Dataset and token summary

| Domain | Dataset | Split | Examples | Tokens | Mean tokens/example | Substitution |
|---|---|---:|---:|---:|---:|---|
| General | Salesforce/wikitext (wikitext-103-raw-v1) | test | 100 | 6400 | 64.0 | no |
| Math | openai/gsm8k (main) | test | 100 | 6400 | 64.0 | no |
| Coding | google-research-datasets/mbpp (full) | test | 100 | 6400 | 64.0 | no |
| Reasoning | allenai/ai2_arc (ARC-Challenge) | test | 100 | 6400 | 64.0 | no |

## Controlled-corpus eligibility

Only examples with enough content for all measured positions and the look-ahead label are eligible. This table makes that length-conditioned selection explicit.

| Domain | Candidate texts | Length-eligible | Selected | Selected original-token range |
|---|---:|---:|---:|---:|
| General | 1000 | 564 | 100 | 70–445 |
| Math | 1000 | 302 | 100 | 65–145 |
| Coding | 500 | 332 | 100 | 65–382 |
| Reasoning | 1000 | 424 | 100 | 65–132 |

## Main cross-domain correlations

| Metric | Mean layer-wise Spearman across domain pairs | Mean top-25% Jaccard across domain pairs |
|---|---:|---:|
| Routing frequency | 0.261 | 0.244 |
| Gate mass | 0.277 | 0.236 |
| Functional contribution | 0.318 | 0.261 |

## Lowest-correlation domain pairs

| Pair | Functional Spearman | Kendall tau | Top-25% Jaccard |
|---|---:|---:|---:|
| General vs Coding | 0.056 | 0.046 | 0.151 |
| Math vs Coding | 0.205 | 0.148 | 0.213 |
| General vs Reasoning | 0.260 | 0.189 | 0.216 |
| Coding vs Reasoning | 0.319 | 0.230 | 0.260 |
| General vs Math | 0.443 | 0.320 | 0.305 |
| Math vs Reasoning | 0.623 | 0.465 | 0.420 |

## Layers with the strongest domain dependence

| MoE layer | Mean functional Spearman across pairs | Lowest pair | Lowest Spearman |
|---:|---:|---|---:|
| 2 | 0.205 | Math vs Coding | -0.062 |
| 13 | 0.210 | General vs Coding | -0.206 |
| 14 | 0.220 | General vs Coding | -0.178 |
| 10 | 0.233 | General vs Coding | -0.094 |
| 12 | 0.240 | General vs Coding | -0.014 |
| 5 | 0.243 | General vs Coding | 0.001 |
| 9 | 0.309 | General vs Coding | 0.101 |
| 15 | 0.310 | General vs Coding | -0.085 |

## Domain-specialized expert examples

Specialization is based on normalized functional contribution. The reported ratio uses a 1e-12 numerical floor; the absolute range and rank change should be used to guard against ratios driven by tiny denominators.

| Layer | Expert | Maximum domain | Minimum domain | Ratio | Normalized range | Rank range |
|---:|---:|---|---|---:|---:|---:|
| 13 | 52 | General | Coding | 22278.67 | 0.0767 | 63.0 |
| 2 | 4 | Math | Coding | 137.95 | 0.0671 | 63.0 |
| 5 | 56 | Reasoning | Coding | 1011.12 | 0.0936 | 62.0 |
| 14 | 3 | General | Coding | 2627.73 | 0.0493 | 62.0 |
| 11 | 48 | Reasoning | Coding | 6783.03 | 0.1096 | 61.0 |
| 7 | 2 | Coding | General | 245.05 | 0.0954 | 61.0 |
| 15 | 49 | General | Coding | 6616.26 | 0.0444 | 61.0 |
| 15 | 59 | General | Coding | 1753.15 | 0.0378 | 61.0 |
| 5 | 2 | Coding | Math | 83.09 | 0.0795 | 60.0 |
| 14 | 58 | Coding | Reasoning | 50.86 | 0.0705 | 60.0 |
| 12 | 63 | Math | Coding | 13171.96 | 0.0684 | 60.0 |
| 5 | 44 | General | Math | 53.67 | 0.0519 | 60.0 |

## Routing frequency versus functional contribution

| Domain | Mean within-domain Spearman across layers |
|---|---:|
| General | 0.913 |
| Math | 0.921 |
| Coding | 0.956 |
| Reasoning | 0.931 |

Functional-contribution rankings were more stable across domains than routing frequency by 0.057 mean Spearman.

## Same-domain split-half reliability

Repeated disjoint half-splits estimate how reproducible each domain's ranking is at half the available sample size. The Spearman–Brown column projects the mean split-half value to the full sample size; it is a reliability diagnostic, not a cross-domain correction.

| Domain | Metric | Mean split-half Spearman | 95% interval | Spearman–Brown |
|---|---|---:|---:|---:|
| General | Routing frequency | 0.947 | [0.916, 0.967] | 0.973 |
| General | Gate mass | 0.945 | [0.911, 0.966] | 0.972 |
| General | Functional contribution | 0.937 | [0.903, 0.962] | 0.967 |
| Math | Routing frequency | 0.978 | [0.965, 0.984] | 0.989 |
| Math | Gate mass | 0.978 | [0.968, 0.984] | 0.989 |
| Math | Functional contribution | 0.979 | [0.972, 0.984] | 0.989 |
| Coding | Routing frequency | 0.987 | [0.981, 0.991] | 0.993 |
| Coding | Gate mass | 0.985 | [0.979, 0.990] | 0.993 |
| Coding | Functional contribution | 0.981 | [0.970, 0.988] | 0.991 |
| Reasoning | Routing frequency | 0.978 | [0.969, 0.984] | 0.989 |
| Reasoning | Gate mass | 0.979 | [0.970, 0.985] | 0.989 |
| Reasoning | Functional contribution | 0.977 | [0.970, 0.984] | 0.989 |

## Bootstrap uncertainty

Examples were resampled independently within each domain from the stored per-example expert vectors; the model was not rerun. Aggregate intervals are for the mean layer-wise Spearman in each domain pair.

| Domain pair | Observed | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| General vs Coding | 0.056 | 0.051 | [0.002, 0.099] |
| Math vs Coding | 0.205 | 0.201 | [0.174, 0.227] |
| General vs Reasoning | 0.260 | 0.256 | [0.194, 0.317] |
| Coding vs Reasoning | 0.319 | 0.316 | [0.272, 0.356] |
| General vs Math | 0.443 | 0.437 | [0.398, 0.483] |
| Math vs Reasoning | 0.623 | 0.620 | [0.595, 0.655] |

## Controlled expert-masking loss effects

For each pre-registered layer/expert pair, its selected gate coefficient was set to zero at the measured source positions only. Tokens were not rerouted, and model weights were not changed. Positive delta NLL means masking made next-token prediction worse.

| Layer/expert | Domain | Proxy rank | Routed-token fraction | Delta NLL (nats/token) | 95% paired-bootstrap CI |
|---|---|---:|---:|---:|---:|
| L11/E27 | General | 55.0 | 0.0586 | -0.000368 | [-0.001169, 0.000380] |
| L11/E27 | Math | 36.0 | 0.1556 | -0.001429 | [-0.002168, -0.000742] |
| L11/E27 | Coding | 1.0 | 0.9633 | 0.034239 | [0.028053, 0.040219] |
| L11/E27 | Reasoning | 47.0 | 0.0702 | -0.001299 | [-0.002122, -0.000454] |
| L10/E56 | General | 56.0 | 0.0755 | -0.002373 | [-0.003148, -0.001602] |
| L10/E56 | Math | 29.0 | 0.2642 | -0.001105 | [-0.002053, -0.000185] |
| L10/E56 | Coding | 1.0 | 0.9537 | 0.028811 | [0.023161, 0.034023] |
| L10/E56 | Reasoning | 45.0 | 0.1253 | -0.001960 | [-0.002937, -0.001050] |
| L1/E25 | General | 27.0 | 0.1314 | -0.008503 | [-0.012635, -0.005102] |
| L1/E25 | Math | 23.0 | 0.1509 | -0.002377 | [-0.005104, 0.000047] |
| L1/E25 | Coding | 2.0 | 0.8253 | 0.022161 | [0.016006, 0.028060] |
| L1/E25 | Reasoning | 15.0 | 0.1775 | -0.007505 | [-0.011337, -0.004367] |

## Proxy-versus-causal domain contrasts

For pre-registered targets, the high and low domains come from the prior prompt-only run rather than being selected from these masking outcomes. The contrast is high-domain minus low-domain mask delta NLL, with domains resampled independently. Controlled proxy extrema are shown as a replication check.

| Layer/expert | Tested high vs low | Controlled proxy high vs low | Loss-delta contrast | 95% CI | Across-domain proxy/loss Spearman | Direction aligned |
|---|---|---|---:|---:|---:|---|
| L11/E27 | Coding vs Reasoning | Coding vs General | 0.035539 | [0.028813, 0.041402] | 0.200 | yes |
| L10/E56 | Coding vs General | Coding vs General | 0.031183 | [0.025796, 0.036232] | 1.000 | yes |
| L1/E25 | Coding vs General | Coding vs General | 0.030664 | [0.023681, 0.037605] | 0.800 | yes |

## Limitations

- The functional metric remains a weighted-output norm. The masking experiment measures loss sensitivity only for the explicitly tested experts.
- Sequence lengths, token budgets, answer inclusion, and the wrapper prefix are controlled, but domain remains entangled with dataset choice and the content's natural surface form.
- The fixed-token design selects examples long enough to supply every measured position and its next-token label. Eligibility rates differ by dataset, so results describe length-matched subsets rather than each benchmark's complete distribution.
- OLMoE's selected top-k weights are used exactly as implemented; when norm_topk_prob=false, selected weights do not sum to one.
- Bootstrap intervals capture example-sampling uncertainty for these corpora, not model, checkpoint, prompt-format, or dataset-choice uncertainty.
- This is one base checkpoint. Conclusions should be checked on another MoE model before guiding compression.
- Selected-route zeroing does not reroute tokens to replacement experts and is not identical to quantizing, deleting, or globally disabling an expert.
- Only a small pre-registered expert set was intervened on; it does not characterize causal sensitivity for all 1,024 layer/expert positions.
- The three high-versus-low domain pairs were pre-registered from the prior prompt-only run. Across-domain proxy/loss Spearman values use only four domains and are descriptive; bootstrap intervals have no multiplicity adjustment.

# Go / No-Go Assessment

**GO: strong support.** 75/96 layer-pair functional correlations were below 0.5; mean functional Spearman was 0.318, and mean top-25% Jaccard was 0.261. The observed disagreement is broad enough to justify testing distributionally robust allocation.

The activation results are now accompanied by controlled selected-route masking for the pre-registered experts. They still do not establish that a specific mixed-precision allocation improves downstream quality.

# Controlled Causal-Validation Assessment

**GO FOR A LIMITED, REVERSIBLE COMPRESSION PILOT.** Mean functional split-half reliability was 0.969. 3/3 pre-registered experts showed both positive high-domain masking harm and a positive high-versus-low loss contrast, with both bootstrap intervals excluding zero. This supports testing a small reversible allocation pilot, not deployment-scale quantization.

> This was a quick-mode run. Treat the assessment as preliminary until the 500-example/domain run reproduces it.
