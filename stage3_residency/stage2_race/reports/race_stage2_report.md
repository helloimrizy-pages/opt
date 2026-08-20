RACE_STAGE2_NO_GO

# RACE Stage 2: Adaptive Multi-Horizon Future-Reuse Ranking

## A. Executive verdict

- Frozen primary RACE variant: `RACE_ONLINE` (`race_online_rankloss_eta0.1_init_uniform_scope_per_layer`).
- Frozen parameters: adviser pool of 9 causal advisers ['MARKOV_H1', 'MARKOV_H2', 'MARKOV_H4', 'MARKOV_H8', 'MARKOV_H16', 'MARKOV_H32', 'GATE_EWMA', 'LFU_DECAY', 'PERSISTENCE'], capped future-reuse horizon `H_max = 32` same-layer events, delayed Hedge learning rate `eta = 0.1`, initialization `uniform`, online loss `rank`, one weight vector per `per_layer` policy instance.
- Improvement over the frozen Stage 1 winner `markov_plus_ewma_h2_beta0.5_alpha0.95` across capacities 12–32: -1.98% to -1.06%.
- Original Stage 0 oracle gap closed: 3.17% to 6.38%, against 9.04% to 11.05% for the Stage 1 winner.
- Preregistered success criteria: The preregistered NO-GO rule fired because improvement over the frozen Stage 1 winner stayed below 5% at capacities [12, 16, 24, 32].

| Capacity | Stage 1 winner | RACE primary | Oracle | Improvement vs Stage 1 | Original oracle gap closed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 9,748,279 | 9,851,604 | 8,146,471 | -1.06% [-1.15, -0.96] | 3.17% [2.90, 3.44] |
| 16 | 7,822,852 | 7,945,937 | 5,904,787 | -1.57% [-1.71, -1.43] | 4.96% [4.50, 5.38] |
| 24 | 5,081,058 | 5,181,473 | 3,294,951 | -1.98% [-2.23, -1.74] | 6.05% [5.30, 6.76] |
| 32 | 3,159,525 | 3,203,094 | 1,785,846 | -1.38% [-1.75, -1.04] | 6.38% [5.42, 7.21] |

## B. Frozen prior evidence

Stage 0 and Stage 1 are unchanged. Stage 2 reads their sealed archives and recomputes none of their numbers.

- Stage 0 verdict `RACE_STAGE0_STRONG_GO`; trace logical hash `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`; archive manifest `9af9a053502709bff9a33017c4f5b80bc0faa2306bc41d980e7bd2d7274346d2`.
- Stage 1 verdict `RACE_STAGE1_STRONG_GO`; winner `markov_plus_ewma_h2_beta0.5_alpha0.95`; archive manifest `4539cab504052010c32f0b87571adc90884283ac060784ddef822c01768e5d1b`.
- Stage 1's decisive diagnostic remains intact: perfect next-use scoring with the same trivial eviction mechanism reproduces the offline oracle exactly at all fifty frozen conditions, so future-reuse ranking, not combinatorial eviction, is the bottleneck Stage 2 attacks.

Stage 1 oracle-gap closure on the same frozen suite (recomputed here only as a reference column, from the sealed Stage 1 rows):

| Capacity | Stage 0 simple | Stage 1 winner | Oracle | Stage 1 gap closed |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 12,849,725 | 12,849,725 | 12,849,725 | N/A |
| 12 | 9,907,430 | 9,748,279 | 8,146,471 | 9.04% |
| 16 | 8,052,402 | 7,822,852 | 5,904,787 | 10.69% |
| 24 | 5,303,005 | 5,081,058 | 3,294,951 | 11.05% |
| 32 | 3,299,634 | 3,159,525 | 1,785,846 | 9.26% |

## C. Algorithm

**Adviser pool.** Nine causal advisers, all reused unmodified from Stage 0/Stage 1: `MARKOV_H1, MARKOV_H2, MARKOV_H4, MARKOV_H8, MARKOV_H16, MARKOV_H32, GATE_EWMA, LFU_DECAY, PERSISTENCE`. The six Markov advisers score the current atomic request against calibration-fitted binary transition matrices at horizons [1, 2, 4, 8, 16, 32]; horizons 1–16 are bitwise identical to the frozen Stage 1 transition archive at its stored precision and horizon 32 is newly fitted on the identical calibration path with the identical frozen fitting function. `GATE_EWMA` uses the frozen Stage 1 alpha 0.95, `LFU_DECAY` the frozen Stage 0 alpha 0.95, and `PERSISTENCE` the previous same-layer request.

**Normalization.** At each decision event the advisers are scored only over the eligible eviction candidates — pre-event residents absent from the current atomic request — and each adviser's scores are replaced by average-rank percentiles `z` in `[0, 1]`, where `1.0` is that adviser's strongest retention recommendation. Exact adviser ties receive identical midranks, which keeps adviser indifference visible instead of manufacturing an ordering.

**Combination and eviction.** `S_e = sum_j w_j z_{j,e}` with `w` on the probability simplex. Candidates are ordered by `S` descending, then LRU recency descending, then expert ID ascending — the exact frozen Stage 1 ordering with `S` in place of the single predictor score — and the lowest-ranked candidates are evicted. No other eviction logic exists, and a Stage 2 variant that places all weight on one adviser reproduces the corresponding frozen Stage 1 single-predictor cost exactly.

**Delayed feedback.** The target is the capped next-use distance `d_tilde = min(d, 33)` in same-layer events. A decision at same-layer event `q` becomes a pending example; it is resolved only at event `q + 32`, once every follow-up event needed for its label has actually been observed, and the weight update is applied at that same event before that event's own decision. Decisions inside the final 32 events of a stream are never used for learning and their unresolved count is reported.

**Ranking loss.** For a resolved example, ordered candidate pairs with `d_tilde(a) < d_tilde(b)` are comparable; adviser `j` pays `0` when it ranks `a` above `b`, `1` when it inverts them and `0.5` on an exact score tie. The unweighted loss divides by the comparable-pair count; the cost-sensitive loss weights each pair by `|1/d_tilde(a) - 1/d_tilde(b)|` with the fixed potential `phi(d) = 1/d`. Both lie in `[0, 1]`. Examples with no comparable pair are skipped and counted.

**Multiplicative update.** `w_j <- w_j exp(-eta * loss_j)` renormalized to the simplex, with `eta = 0.1` selected on calibration alone from the frozen grid [0.1, 0.3, 1.0]. The state is carried in log space with per-step maximum subtraction, which is algebraically identical and cannot overflow or underflow.

## D. Causality audit

RACE never reads a future event at decision time. Operationally:

| Capacity | Examples generated | Resolved | Unresolved at stream end | Skipped (no comparable pair) | Applied updates | Mean delay | Max delay | Min update-minus-decision offset |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 2,678,247 | 2,673,313 | 4,934 (0.184%) | 3,691 | 2,669,622 | 32.00 | 32 | 32 |
| 16 | 2,541,357 | 2,536,678 | 4,679 (0.184%) | 713 | 2,535,965 | 32.00 | 32 | 32 |
| 24 | 2,146,990 | 2,143,019 | 3,971 (0.185%) | 389 | 2,142,630 | 32.00 | 32 | 32 |
| 32 | 1,650,597 | 1,647,682 | 2,915 (0.177%) | 255 | 1,647,427 | 32.00 | 32 | 32 |

- The minimum observed update-minus-decision offset equals `H_max = 32` same-layer events at every capacity, so no weight update ever preceded the observation of its own label.
- Every applied update carries recorded `decision_event_index`, `label_resolution_event_index` and `weight_update_event_index`; the simulator raises rather than updating if `update < decision + H_max`.
- The causal capped label built from the rolling 32-event window was cross-checked against an offline future-use computation in the pilot audit and matched on every checked example.
- The offline ranking observer is a one-way sink: enabling it leaves every simulated cost bit-identical, which is verified in both the pilot audit and the unit tests.
- Calibration used only the 80 frozen Stage 0 calibration sequences; evaluation used the disjoint 320-sequence split, and nothing was reselected after evaluation began.

## E. Main results

Frozen ten-workload suite, unit expert-transfer cost at lambda zero. Capacity 8 is degenerate (top-k = 8 leaves no eviction freedom) and contributes to no verdict count.

| Capacity | Stage 1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 |
| 12 | 9,748,279 | 9,999,390 | 9,919,727 | 9,851,604 | 9,833,069 | 8,146,471 |
| 16 | 7,822,852 | 8,120,618 | 8,005,697 | 7,945,937 | 7,939,359 | 5,904,787 |
| 24 | 5,081,058 | 5,333,296 | 5,230,074 | 5,181,473 | 5,245,722 | 3,294,951 |
| 32 | 3,159,525 | 3,335,492 | 3,233,051 | 3,203,094 | 3,323,031 | 1,785,846 |

Normalized by the Stage 0 strongest-simple cost at the same capacity:

| Capacity | Stage 1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 0.9839 | 1.0093 | 1.0012 | 0.9944 | 0.9925 | 0.8223 |
| 16 | 0.9715 | 1.0085 | 0.9942 | 0.9868 | 0.9860 | 0.7333 |
| 24 | 0.9581 | 1.0057 | 0.9862 | 0.9771 | 0.9892 | 0.6213 |
| 32 | 0.9575 | 1.0109 | 0.9798 | 0.9707 | 1.0071 | 0.5412 |

Success metrics for the frozen primary variant `RACE_ONLINE` (95% intervals are the conditional paired bootstrap described in section K):

| Capacity | Improvement vs Stage 1 | Original oracle gap closed | Stage 1 residual recovered |
| ---: | ---: | ---: | ---: |
| 12 | -1.06% [-1.15, -0.96] | 3.17% [2.90, 3.44] | -6.45% [-7.02, -5.83] |
| 16 | -1.57% [-1.71, -1.43] | 4.96% [4.50, 5.38] | -6.42% [-7.04, -5.84] |
| 24 | -1.98% [-2.23, -1.74] | 6.05% [5.30, 6.76] | -5.62% [-6.40, -4.96] |
| 32 | -1.38% [-1.75, -1.04] | 6.38% [5.42, 7.21] | -3.17% [-4.05, -2.38] |

By workload regime (no regime is averaged away):

| Regime | Capacity | Stage 0 simple | Stage 1 | RACE primary | Oracle | Improvement vs Stage 1 | Gap closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stationary | 12 | 2,305,638 | 2,264,721 | 2,279,645 | 1,896,388 | -0.66% | 6.35% |
| stationary | 16 | 1,872,290 | 1,814,668 | 1,836,079 | 1,375,283 | -1.18% | 7.29% |
| stationary | 24 | 1,230,577 | 1,175,318 | 1,201,464 | 768,001 | -2.22% | 6.29% |
| stationary | 32 | 763,662 | 728,401 | 742,036 | 416,358 | -1.87% | 6.23% |
| abrupt | 12 | 4,552,306 | 4,488,773 | 4,535,807 | 3,752,707 | -1.05% | 2.06% |
| abrupt | 16 | 3,693,108 | 3,598,607 | 3,657,652 | 2,715,747 | -1.64% | 3.63% |
| abrupt | 24 | 2,424,479 | 2,335,230 | 2,377,466 | 1,511,098 | -1.81% | 5.15% |
| abrupt | 32 | 1,502,762 | 1,449,860 | 1,465,212 | 816,437 | -1.06% | 5.47% |
| repeated | 12 | 727,303 | 716,788 | 726,356 | 598,832 | -1.33% | 0.74% |
| repeated | 16 | 592,679 | 577,674 | 587,541 | 436,094 | -1.71% | 3.28% |
| repeated | 24 | 390,600 | 377,218 | 382,562 | 244,364 | -1.42% | 5.50% |
| repeated | 32 | 243,124 | 235,345 | 236,395 | 132,133 | -0.45% | 6.06% |
| mixed | 12 | 2,322,183 | 2,277,997 | 2,309,796 | 1,898,544 | -1.40% | 2.92% |
| mixed | 16 | 1,894,325 | 1,831,903 | 1,864,665 | 1,377,663 | -1.79% | 5.74% |
| mixed | 24 | 1,257,349 | 1,193,292 | 1,219,981 | 771,488 | -2.24% | 7.69% |
| mixed | 32 | 790,086 | 745,919 | 759,451 | 420,918 | -1.81% | 8.30% |

### Stability and regressions

The preregistered stability flag marks any workload/capacity where `C_RACE > 1.03 * C_Stage1`. 4 of 50 workload/capacity cells are flagged.

| Workload | Regime | Capacity | Stage 1 | RACE | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| stationary_reasoning | stationary | 32 | 219,911 | 229,890 | 1.0454 |
| stationary_reasoning | stationary | 24 | 343,417 | 357,081 | 1.0398 |
| stationary_coding | stationary | 32 | 113,003 | 116,974 | 1.0351 |
| math_to_reasoning | abrupt | 32 | 401,800 | 414,090 | 1.0306 |

## F. Ablation

Uniform -> Static -> Online -> Cost, on the frozen ten-workload suite. A positive relative change means the later configuration is cheaper.

| Question | Comparison | Capacity | Before | After | Relative change |
| --- | --- | ---: | ---: | ---: | ---: |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 12 | 9,748,279 | 9,999,390 | -2.58% |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 16 | 7,822,852 | 8,120,618 | -3.81% |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 24 | 5,081,058 | 5,333,296 | -4.96% |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 32 | 3,159,525 | 3,335,492 | -5.57% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 12 | 9,999,390 | 9,919,727 | +0.80% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 16 | 8,120,618 | 8,005,697 | +1.42% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 24 | 5,333,296 | 5,230,074 | +1.94% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 32 | 3,335,492 | 3,233,051 | +3.07% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 12 | 9,919,727 | 9,851,604 | +0.69% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 16 | 8,005,697 | 7,945,937 | +0.75% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 24 | 5,230,074 | 5,181,473 | +0.93% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 32 | 3,233,051 | 3,203,094 | +0.93% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 12 | 9,851,604 | 9,833,069 | +0.19% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 16 | 7,945,937 | 7,939,359 | +0.08% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 24 | 5,181,473 | 5,245,722 | -1.24% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 32 | 3,203,094 | 3,323,031 | -3.74% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 12 | 9,999,390 | 9,944,175 | +0.55% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 16 | 8,120,618 | 8,038,921 | +1.01% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 24 | 5,333,296 | 5,254,771 | +1.47% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 32 | 3,335,492 | 3,276,729 | +1.76% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 12 | 9,851,604 | 9,738,992 | +1.14% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 16 | 7,945,937 | 7,811,937 | +1.69% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 24 | 5,181,473 | 5,066,820 | +2.21% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 32 | 3,203,094 | 3,137,922 | +2.03% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 12 | 9,748,279 | 9,738,992 | +0.10% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 16 | 7,822,852 | 7,811,937 | +0.14% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 24 | 5,081,058 | 5,066,820 | +0.28% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 32 | 3,159,525 | 3,137,922 | +0.68% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 12 | 9,919,727 | 9,912,736 | +0.07% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 16 | 8,005,697 | 7,992,796 | +0.16% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 24 | 5,230,074 | 5,222,621 | +0.14% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 32 | 3,233,051 | 3,234,006 | -0.03% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 12 | 9,851,604 | 9,887,855 | -0.37% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 16 | 7,945,937 | 8,009,648 | -0.80% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 24 | 5,181,473 | 5,274,423 | -1.79% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 32 | 3,203,094 | 3,302,906 | -3.12% |

## G. Ranking diagnostics

Measured on the actual eviction candidate sets. Pairwise ordering accuracy is `P(S_a > S_b | d_a < d_b)` with half credit for exact score ties; oracle-consistent eviction means at least one evicted expert attained the maximum true next-use distance among the candidates, and oracle-optimal means the evicted set matched a farthest-future-optimal choice.

| Variant | Capacity | Eviction events | Ordering accuracy (capped) | Ordering accuracy (true) | Oracle-consistent | Oracle-optimal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RACE_COST | 12 | 2,676,993 | 65.25% | 65.16% | 70.40% | 20.37% |
| RACE_COST | 16 | 2,538,092 | 62.91% | 62.84% | 50.32% | 12.31% |
| RACE_COST | 24 | 2,153,915 | 61.31% | 61.22% | 27.54% | 8.07% |
| RACE_COST | 32 | 1,683,135 | 61.09% | 60.98% | 16.98% | 6.51% |
| RACE_COST_EXTENDED | 12 | 2,673,754 | 66.50% | 66.43% | 71.95% | 21.98% |
| RACE_COST_EXTENDED | 16 | 2,530,301 | 64.31% | 64.25% | 53.67% | 14.22% |
| RACE_COST_EXTENDED | 24 | 2,125,445 | 62.64% | 62.57% | 31.63% | 9.84% |
| RACE_COST_EXTENDED | 32 | 1,633,640 | 62.25% | 62.16% | 19.98% | 8.04% |
| RACE_ONLINE | 12 | 2,678,247 | 65.45% | 65.36% | 71.80% | 20.98% |
| RACE_ONLINE | 16 | 2,541,357 | 63.18% | 63.10% | 54.00% | 13.71% |
| RACE_ONLINE | 24 | 2,146,990 | 61.75% | 61.66% | 32.48% | 9.85% |
| RACE_ONLINE | 32 | 1,650,597 | 61.60% | 61.49% | 21.47% | 8.47% |
| RACE_ONLINE_EXTENDED | 12 | 2,673,900 | 66.60% | 66.53% | 72.30% | 22.11% |
| RACE_ONLINE_EXTENDED | 16 | 2,529,905 | 64.40% | 64.34% | 54.24% | 14.41% |
| RACE_ONLINE_EXTENDED | 24 | 2,123,823 | 62.73% | 62.66% | 32.51% | 10.15% |
| RACE_ONLINE_EXTENDED | 32 | 1,627,933 | 62.36% | 62.26% | 20.94% | 8.43% |
| RACE_ONLINE_GLOBAL | 12 | 2,679,809 | 65.06% | 64.97% | 71.56% | 20.82% |
| RACE_ONLINE_GLOBAL | 16 | 2,543,193 | 62.54% | 62.45% | 53.46% | 13.56% |
| RACE_ONLINE_GLOBAL | 24 | 2,153,591 | 60.77% | 60.69% | 31.21% | 9.38% |
| RACE_ONLINE_GLOBAL | 32 | 1,673,093 | 60.65% | 60.53% | 20.28% | 7.83% |
| RACE_STATIC | 12 | 2,670,957 | 63.41% | 63.38% | 72.45% | 16.14% |
| RACE_STATIC | 16 | 2,536,018 | 61.98% | 61.95% | 51.63% | 9.36% |
| RACE_STATIC | 24 | 2,161,696 | 61.20% | 61.17% | 28.50% | 7.28% |
| RACE_STATIC | 32 | 1,671,634 | 61.18% | 61.13% | 18.55% | 7.04% |
| RACE_STATIC_EXTENDED | 12 | 2,670,961 | 63.53% | 63.50% | 72.57% | 16.25% |
| RACE_STATIC_EXTENDED | 16 | 2,535,452 | 62.06% | 62.04% | 51.84% | 9.47% |
| RACE_STATIC_EXTENDED | 24 | 2,160,516 | 61.26% | 61.23% | 28.62% | 7.32% |
| RACE_STATIC_EXTENDED | 32 | 1,670,277 | 61.23% | 61.19% | 18.64% | 7.08% |
| RACE_STATIC_PERLAYER | 12 | 2,671,483 | 63.58% | 63.55% | 72.51% | 16.26% |
| RACE_STATIC_PERLAYER | 16 | 2,536,645 | 62.11% | 62.09% | 51.78% | 9.46% |
| RACE_STATIC_PERLAYER | 24 | 2,160,752 | 61.33% | 61.30% | 28.67% | 7.37% |
| RACE_STATIC_PERLAYER | 32 | 1,669,522 | 61.30% | 61.25% | 18.49% | 7.03% |
| RACE_UNIFORM | 12 | 2,686,629 | 63.18% | 63.17% | 70.97% | 17.09% |
| RACE_UNIFORM | 16 | 2,560,990 | 61.41% | 61.41% | 52.09% | 11.17% |
| RACE_UNIFORM | 24 | 2,185,257 | 60.26% | 60.26% | 30.04% | 8.26% |
| RACE_UNIFORM | 32 | 1,712,711 | 60.26% | 60.24% | 18.51% | 7.07% |
| RACE_UNIFORM_EXTENDED | 12 | 2,683,379 | 63.83% | 63.81% | 71.23% | 17.86% |
| RACE_UNIFORM_EXTENDED | 16 | 2,553,920 | 62.00% | 61.99% | 52.46% | 11.72% |
| RACE_UNIFORM_EXTENDED | 24 | 2,172,730 | 60.77% | 60.75% | 30.39% | 8.61% |
| RACE_UNIFORM_EXTENDED | 32 | 1,693,250 | 60.75% | 60.72% | 18.90% | 7.28% |

Association between ranking quality and realized transfer cost across all evaluated variants and non-degenerate capacities:

- `pairwise_ordering_accuracy_capped` vs improvement over Stage 1: Spearman 0.804 (descriptive p=4.12e-10, 40 configuration points).
- `oracle_consistent_eviction_rate` vs improvement over Stage 1: Spearman 0.453 (descriptive p=0.00333, 40 configuration points).

Descriptive configuration-level association across variants and capacities; the configurations are not independent experiments.

## H. Horizon behavior

Mean deployed weight assigned to each Markov horizon by the frozen primary variant, averaged over layers and workloads:

| Capacity | Spare slots | H1 | H2 | H4 | H8 | H16 | H32 | Markov mass | Weighted mean horizon |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 4 | 0.0502 | 0.1062 | 0.0430 | 0.0149 | 0.0027 | 0.0019 | 0.2188 | 2.999 |
| 16 | 8 | 0.0666 | 0.1578 | 0.0687 | 0.0230 | 0.0030 | 0.0020 | 0.3211 | 2.969 |
| 24 | 16 | 0.0577 | 0.1823 | 0.1298 | 0.0378 | 0.0045 | 0.0027 | 0.4147 | 3.379 |
| 32 | 24 | 0.0415 | 0.1630 | 0.1489 | 0.0522 | 0.0074 | 0.0042 | 0.4172 | 3.912 |

## I. Weight adaptation

Mixed-interleaved workload, per-layer adviser weights of the frozen primary variant:

| Capacity | Layers | Mean effective advisers | End effective advisers | Most common dominant adviser | Mean dominant weight |
| ---: | ---: | ---: | ---: | --- | ---: |
| 12 | 16 | 1.779 | 1.201 | LFU_DECAY (13/16) | 0.8244 |
| 16 | 16 | 2.463 | 1.825 | LFU_DECAY (10/16) | 0.6723 |
| 24 | 16 | 2.918 | 2.110 | LFU_DECAY (9/16) | 0.5489 |
| 32 | 16 | 3.192 | 2.187 | MARKOV_H4 (6/16) | 0.5338 |

Empirical adviser regret (this is accounting, not a theoretical guarantee): cumulative resolved mixture loss of RACE minus the cumulative loss of the best single fixed adviser in hindsight.

| Capacity | Resolved examples | Mixture rank loss | Best fixed adviser | Best fixed loss | Empirical regret | Regret per example |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 12 | 2,673,313 | 923,718.1 | LFU_DECAY | 922,397.1 | 1,321.0 | +0.00049 |
| 16 | 2,536,678 | 942,309.5 | LFU_DECAY | 941,779.5 | 530.0 | +0.00021 |
| 24 | 2,143,019 | 839,935.2 | LFU_DECAY | 840,029.8 | -94.6 | -0.00004 |
| 32 | 1,647,682 | 652,719.7 | LFU_DECAY | 652,333.6 | 386.1 | +0.00023 |

Figure 3 plots adviser-weight trajectories for layers [0, 1], chosen by the preregistered rule (largest and median per-layer miss reduction of the frozen primary variant relative to RACE_UNIFORM on mixed_interleaved at capacity 32).

## J. Oracle residual

| Capacity | RACE cost | Oracle cost | Residual headroom vs Stage 0 simple | Remaining fraction of the Stage 1 residual |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 9,851,604 | 8,146,471 | 17.21% | 106.45% |
| 16 | 7,945,937 | 5,904,787 | 25.35% | 106.42% |
| 24 | 5,181,473 | 3,294,951 | 35.57% | 105.62% |
| 32 | 3,203,094 | 1,785,846 | 42.95% | 103.17% |

## K. Limitations

- These are trace simulations of expert residency, misses, admissions and transfers. No end-to-end latency improvement and no hardware speedup is claimed or measured.
- The trace is one fixed OLMoE-1B-7B-0924 decode trace on four fixed domains; nothing here generalizes automatically to other models, batch sizes or serving stacks.
- The online learning target is capped at 32 same-layer events. Reuse structure beyond that horizon is invisible to the learner by construction.
- RACE combines existing simple causal advisers. It trains no neural network, uses no reinforcement learning, performs no speculative prefetch and does not change OLMoE routing.
- Plain multiplicative weights is not a tracking algorithm: once an adviser's cumulative loss advantage becomes large its competitors receive numerically negligible weight, so the measured adaptation should be read as concentration, not as unlimited regime tracking.
- Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen stateful workload path. The cache trajectory and the online-learning trajectory are not regenerated under resampled orderings, so these are conditional intervals, not full uncertainty over possible workload orderings.
- The offline oracle and the perfect-score policy are non-causal diagnostics, not deployable methods.

## L. Next recommendation

Do not respond by building a neural predictor. The preregistered NO-GO rule fired and the diagnostic sections above localize why. Online adaptation changed cost relative to the calibration-learned static weights by +0.82% on average across the four non-degenerate capacities. Calibration-learned static weights changed cost relative to uniform weights by +1.80% on average. The labeled adviser-diversity ablation, which adds the frozen Stage 1 winner itself to the pool, changed cost relative to the Stage 1 winner by +0.30% on average; this isolates how much of the primary-pool result is a pool-expressiveness limit rather than a weighting limit.

### Diagnostic analysis

The preregistered response to this outcome is diagnosis, not added model capacity. Each question below is answered from the measurements in this run.

**Is adviser diversity too low?** Partly, and in a specific way. Adding exactly one adviser — the frozen Stage 1 winner itself, which is a raw-scale blend of two pool members — moves the online variant by +1.77% on average and turns the comparison against Stage 1 from negative into +0.30%. The primary pool cannot express that blend because percentile normalization is applied to each adviser separately and discards the magnitude information the raw-scale mixture uses. So the binding constraint is the representation the specification mandates, not the number of advisers: a tenth adviser only reaches parity with Stage 1, it does not unlock the oracle gap.

**Is the needed future information absent from causal routing history?** Largely yes. On the actual eviction candidate sets the combined score orders only 61.6%–65.4% of comparable pairs correctly, against 100% for the perfect-score policy that reproduces the oracle. The oracle-consistent eviction rate falls from 71.8% at the smallest non-degenerate capacity to 21.5% at the largest, i.e. the ranking is weakest exactly where the residual oracle headroom is largest. Every adviser in the pool is a low-order statistic of the same causal routing history, so their errors are correlated and re-weighting them cannot manufacture information none of them carries.

**Is the useful horizon longer than 32?** No — the measurement points the other way. The two longest Markov advisers together hold only 0.45%–1.16% of the deployed weight, and the weight-averaged Markov horizon stays between 2.97 and 3.91 same-layer events. The effective horizon does grow with spare residency (section H), which supports the Stage 1 lookahead finding, but it grows within a short range. Raising `H_max` above 32 is therefore not the indicated next step.

**Are the current gate and routing features insufficient?** This is where the evidence points. The six calibration-fitted Markov transition advisers together hold only 21.9%–41.7% of the deployed weight; the rest goes to decayed-frequency and gate-EWMA recency, which are pure history statistics with no routing content. A learner that is free to weight conditional next-request probabilities against plain recency mostly chooses recency, which says the transition features add little beyond what recency already encodes on this trace.

**Is the delayed adaptation too slow?** No. Across the four non-degenerate capacities the run applied 8,995,644 delayed weight updates at a fixed 32-event delay, and the empirical per-example adviser regret against the best fixed adviser in hindsight lies between -0.00004 and +0.00049. The online learner is already tracking the best single adviser essentially exactly; the loss it is minimizing simply does not have much left to give. Consistently with that, calibration selected the smallest learning rate in the frozen grid.

Taken together the indicated next direction is a **richer causal feature or scoring representation that preserves magnitude**, evaluated against the same frozen Stage 0/Stage 1 references, rather than a better weighting scheme over the present rank-normalized pool. Two concrete, still-non-neural candidates follow directly from the measurements above: (i) combine advisers on a calibrated common numeric scale instead of within-event percentiles, since the one representational change measured here recovered the entire primary-pool deficit; and (ii) target the loss at the retention boundary rather than at all comparable pairs, since only inversions that cross the eviction cutoff can change a miss. Neither requires a neural predictor, and neither should be adopted without its own preregistration.

## Reproducibility

- Stage 2 preregistration hash: `0cb2ce8ab260094934310edc188dc925811ccefb3ca11912a3aae8f361f09aef`
- Stage 2 frozen config file: `2ace8993dbe545080364223423768451b2d367b0221c6c93b2eb37f94ec1b2a9`
- Stage 2 evaluation manifest: `bcb65350561d28213b76c6671ebb3058f80b69f709b091a7d745b7f0b1fd7cb8`
- Stage 0 trace logical hash: `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`
- Stage 0 archive manifest: `9af9a053502709bff9a33017c4f5b80bc0faa2306bc41d980e7bd2d7274346d2`
- Stage 1 archive manifest: `4539cab504052010c32f0b87571adc90884283ac060784ddef822c01768e5d1b`
- Stage 1 frozen config: `c595b061a415a9874d83f167e1e0fb7873ec8850ace8acc1a5db17f814351b83`
- Stage 1 winner: `markov_plus_ewma_h2_beta0.5_alpha0.95`
- Stage 2 source commit at freeze: `f9236cb35541f07c96a8e03b6a36eacc1f21a92c`
- Stage 2 source bundle hash: `3ff7dc3cf0ff15e242231535b7ffc021e5b0858ef04b1595f5688454f43b3ae6`
- Stage 2 transition models: `d62a14dcb5fbc5b819ac955636a4260c5ea2c676d2d285ce20e2e9682e922752`
- Selected eta: `0.1`; initialization: `uniform`; primary loss: `rank`; weight scope: `per_layer`; H_max: `32`.

## Core answer

Can adaptive causal multi-horizon future-reuse ranking recover a substantial fraction of the expert-residency oracle gap that simple fixed prediction leaves behind? On this frozen evidence, the answer is: the frozen primary variant `RACE_ONLINE` changed transfer cost against the Stage 1 winner by -1.98% to -1.06% across capacities 12–32 and closed 3.17% to 6.38% of the original Stage 0 oracle gap, against 9.04% to 11.05% for simple fixed prediction. The preregistered verdict is RACE_STAGE2_NO_GO.
