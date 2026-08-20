RACE_STAGE3_PARTIAL_SUCCESS

# RACE Stage 3: Learned Causal Future-Reuse Ranking

## A. Executive verdict

- Frozen primary variant: `STAGE3_RANKER` — one calibration-fitted linear ranking function per cache capacity over 45 raw-scale causal features, driving the unchanged Stage 1 eviction rule.
- Improvement over the frozen Stage 1 winner `markov_plus_ewma_h2_beta0.5_alpha0.95` across capacities 12–32: 2.85% to 5.75%.
- Original Stage 0 oracle gap closed: 21.25% to 25.16%, against 9.04% to 11.05% for the Stage 1 winner.
- A strictly positive improvement over the frozen Stage 1 winner whose paired 95% interval excludes zero at all 4 non-degenerate capacities, at least 20% of the original Stage 0 oracle gap closed at [12, 16, 24, 32], and no capacity regressing above the 3% threshold.

| Capacity | Stage 1 winner | Stage 3 primary | Oracle | Improvement vs Stage 1 | Oracle gap closed | Stage 1 gap closed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 9,748,279 | 9,470,228 | 8,146,471 | 2.85% [2.70, 3.04] | 24.83% | 9.04% |
| 16 | 7,822,852 | 7,512,017 | 5,904,787 | 3.97% [3.73, 4.22] | 25.16% | 10.69% |
| 24 | 5,081,058 | 4,833,005 | 3,294,951 | 4.88% [4.51, 5.22] | 23.41% | 11.05% |
| 32 | 3,159,525 | 2,978,002 | 1,785,846 | 5.75% [5.24, 6.25] | 21.25% | 9.26% |

### Stage 2's criteria, applied verbatim

Stage 3 keeps Stage 2's STRONG and VERY_STRONG rungs byte-identical so the stages stay comparable, and adds a PARTIAL rung between a 10% cost win and failure. Under Stage 2's own rule this run would be judged:

- Condition A (at least 10% better than Stage 1) met at capacities [].
- Condition B (at least 30% of the original oracle gap) met at capacities [].
- Stage 2 STRONG_SUCCESS would NOT pass.

## B. Frozen prior evidence

- Stage 0 `RACE_STAGE0_STRONG_GO`; trace `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`; archive `9af9a053502709bff9a33017c4f5b80bc0faa2306bc41d980e7bd2d7274346d2`.
- Stage 1 `RACE_STAGE1_STRONG_GO`; winner `markov_plus_ewma_h2_beta0.5_alpha0.95`; archive `4539cab504052010c32f0b87571adc90884283ac060784ddef822c01768e5d1b`.
- Stage 2 `RACE_STAGE2_NO_GO`; archive `a0ec2f31263daa6fb304cdfc0f03247db953ec7f676fe101d680e1785aff4cd3`.

Stage 2's diagnostic is the reason Stage 3 exists. It localized the failure as representational: percentile rank normalization applied to each adviser separately destroys the magnitude information the raw-scale Stage 1 hybrid uses, and no weighting over a rank-normalized pool can rebuild it. Stage 3 acts on exactly that finding.

## C. Algorithm

**Scoring.** One linear ranking function per cache capacity, `S_e = w_B · x_e`, over causal features kept on their natural scale. The frozen Stage 1 eviction rule is unchanged: retain the highest-scoring eligible candidates, ties broken by LRU recency then expert ID.

**Features.** Calibration-fitted Markov survival probabilities at horizons 1–32 and their band differences; noisy-OR context combinations; the same context evaluated against the previous same-layer request; renewal structure (elapsed time, the two most recent inter-arrival gaps, an overdue ratio); exact windowed counts over 4/8/16/32 events; decayed request and gate statistics; calibration popularity; and decode-request-scope statistics. Every value uses only events already observed.

**Fitting.** Convex weighted pairwise logistic ranking loss over within-candidate-set ordered pairs, target the capped next-use distance `min(d, 33)`, L2 chosen on a held-out tail of calibration groups, L-BFGS-B, calibration path only. Two collection rounds: round 1 under the frozen Stage 1 winner, round 2 under the round-1 model.

**What Stage 3 does not do.** No neural network, no reinforcement learning, no recurrent or attention model, no online adaptation at evaluation time, no prefetching, no change to OLMoE routing, and no future information at decision time.

## D. Causality and mechanism audit

- The feature state is advanced only after the features for the same event have been read, so no feature can see its own event's consequences.
- A Stage 3 replay driven by the frozen Stage 1 winner score reproduces the frozen Stage 1 cost exactly, which proves the eviction mechanism is unchanged.
- The same mechanism driven by exact next-use scores reproduces the Stage 0 oracle exactly.
- Mutating a later sequence cannot change any earlier action.
- Fitting, feature standardization, static popularity and the L2 choice used only the 80 frozen calibration sequences; evaluation used the disjoint 320-sequence split.

## E. Main results

| Capacity | Stage 1 winner | STAGE3_RANKER | STAGE3_RANKER_NO_REQUEST_SCOPE | STAGE3_RANKER_POOLED | STAGE3_RANKER_ROUND1_DATA | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 |
| 12 | 9,748,279 | 9,470,228 | 9,497,431 | 9,486,366 | 9,479,472 | 8,146,471 |
| 16 | 7,822,852 | 7,512,017 | 7,565,032 | 7,517,406 | 7,518,147 | 5,904,787 |
| 24 | 5,081,058 | 4,833,005 | 4,899,005 | 4,819,319 | 4,813,280 | 3,294,951 |
| 32 | 3,159,525 | 2,978,002 | 3,038,173 | 2,968,990 | 2,946,466 | 1,785,846 |

By workload regime:

| Regime | Capacity | Stage 1 | Stage 3 | Oracle | Improvement | Gap closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stationary | 12 | 2,264,721 | 2,203,006 | 1,896,388 | 2.73% | 25.08% |
| stationary | 16 | 1,814,668 | 1,747,460 | 1,375,283 | 3.70% | 25.12% |
| stationary | 24 | 1,175,318 | 1,124,785 | 768,001 | 4.30% | 22.87% |
| stationary | 32 | 728,401 | 693,193 | 416,358 | 4.83% | 20.29% |
| abrupt | 12 | 4,488,773 | 4,364,556 | 3,752,707 | 2.77% | 23.48% |
| abrupt | 16 | 3,598,607 | 3,459,146 | 2,715,747 | 3.88% | 23.94% |
| abrupt | 24 | 2,335,230 | 2,219,532 | 1,511,098 | 4.95% | 22.44% |
| abrupt | 32 | 1,449,860 | 1,363,585 | 816,437 | 5.95% | 20.28% |
| repeated | 12 | 716,788 | 695,756 | 598,832 | 2.93% | 24.56% |
| repeated | 16 | 577,674 | 553,568 | 436,094 | 4.17% | 24.98% |
| repeated | 24 | 377,218 | 357,838 | 244,364 | 5.14% | 22.40% |
| repeated | 32 | 235,345 | 220,339 | 132,133 | 6.38% | 20.53% |
| mixed | 12 | 2,277,997 | 2,206,910 | 1,898,544 | 3.12% | 27.21% |
| mixed | 16 | 1,831,903 | 1,751,843 | 1,377,663 | 4.37% | 27.58% |
| mixed | 24 | 1,193,292 | 1,130,850 | 771,488 | 5.23% | 26.04% |
| mixed | 32 | 745,919 | 700,885 | 420,918 | 6.04% | 24.16% |

### Stability. 0 of 50 workload/capacity cells exceed the 3% regression flag.

No workload/capacity cell exceeded the 3% regression threshold.

## F. Ablation

| Question | Comparison | Capacity | Before | After | Relative change |
| --- | --- | ---: | ---: | ---: | ---: |
| A_stage1_to_primary | Stage 1 winner -> STAGE3_RANKER | 12 | 9,748,279 | 9,470,228 | +2.85% |
| A_stage1_to_primary | Stage 1 winner -> STAGE3_RANKER | 16 | 7,822,852 | 7,512,017 | +3.97% |
| A_stage1_to_primary | Stage 1 winner -> STAGE3_RANKER | 24 | 5,081,058 | 4,833,005 | +4.88% |
| A_stage1_to_primary | Stage 1 winner -> STAGE3_RANKER | 32 | 3,159,525 | 2,978,002 | +5.75% |
| B_request_scope_value | without request scope -> with request scope | 12 | 9,497,431 | 9,470,228 | +0.29% |
| B_request_scope_value | without request scope -> with request scope | 16 | 7,565,032 | 7,512,017 | +0.70% |
| B_request_scope_value | without request scope -> with request scope | 24 | 4,899,005 | 4,833,005 | +1.35% |
| B_request_scope_value | without request scope -> with request scope | 32 | 3,038,173 | 2,978,002 | +1.98% |
| B2_stage1_to_no_request_scope | Stage 1 winner -> STAGE3_RANKER_NO_REQUEST_SCOPE (identical information) | 12 | 9,748,279 | 9,497,431 | +2.57% |
| B2_stage1_to_no_request_scope | Stage 1 winner -> STAGE3_RANKER_NO_REQUEST_SCOPE (identical information) | 16 | 7,822,852 | 7,565,032 | +3.30% |
| B2_stage1_to_no_request_scope | Stage 1 winner -> STAGE3_RANKER_NO_REQUEST_SCOPE (identical information) | 24 | 5,081,058 | 4,899,005 | +3.58% |
| B2_stage1_to_no_request_scope | Stage 1 winner -> STAGE3_RANKER_NO_REQUEST_SCOPE (identical information) | 32 | 3,159,525 | 3,038,173 | +3.84% |
| C_per_capacity_value | pooled model -> per-capacity models | 12 | 9,486,366 | 9,470,228 | +0.17% |
| C_per_capacity_value | pooled model -> per-capacity models | 16 | 7,517,406 | 7,512,017 | +0.07% |
| C_per_capacity_value | pooled model -> per-capacity models | 24 | 4,819,319 | 4,833,005 | -0.28% |
| C_per_capacity_value | pooled model -> per-capacity models | 32 | 2,968,990 | 2,978,002 | -0.30% |
| D_retraining_value | round-1 data -> round-2 data | 12 | 9,479,472 | 9,470,228 | +0.10% |
| D_retraining_value | round-1 data -> round-2 data | 16 | 7,518,147 | 7,512,017 | +0.08% |
| D_retraining_value | round-1 data -> round-2 data | 24 | 4,813,280 | 4,833,005 | -0.41% |
| D_retraining_value | round-1 data -> round-2 data | 32 | 2,946,466 | 2,978,002 | -1.07% |

`B2_stage1_to_no_request_scope` is the honest like-for-like comparison: it restricts Stage 3 to the information Stage 1 also had. The difference between it and the primary variant is the value of knowing decode-request boundaries, which a serving stack always knows but no earlier RACE stage used.

## G. Ranking diagnostics

| Variant | Capacity | Eviction events | Ordering accuracy (capped) | Oracle-consistent | Oracle-optimal |
| --- | ---: | ---: | ---: | ---: | ---: |
| STAGE3_RANKER | 12 | 2,662,023 | 70.28% | 76.54% | 25.74% |
| STAGE3_RANKER | 16 | 2,508,123 | 68.27% | 56.99% | 15.70% |
| STAGE3_RANKER | 24 | 2,096,050 | 66.41% | 34.35% | 10.93% |
| STAGE3_RANKER | 32 | 1,601,243 | 65.87% | 23.45% | 9.76% |
| STAGE3_RANKER_NO_REQUEST_SCOPE | 12 | 2,662,646 | 69.67% | 75.56% | 25.04% |
| STAGE3_RANKER_NO_REQUEST_SCOPE | 16 | 2,514,300 | 67.57% | 54.76% | 14.67% |
| STAGE3_RANKER_NO_REQUEST_SCOPE | 24 | 2,104,739 | 65.82% | 32.48% | 10.02% |
| STAGE3_RANKER_NO_REQUEST_SCOPE | 32 | 1,616,330 | 65.24% | 21.80% | 8.74% |
| STAGE3_RANKER_POOLED | 12 | 2,664,860 | 70.38% | 77.24% | 25.96% |
| STAGE3_RANKER_POOLED | 16 | 2,512,436 | 68.31% | 58.63% | 16.51% |
| STAGE3_RANKER_POOLED | 24 | 2,087,698 | 66.35% | 35.77% | 11.66% |
| STAGE3_RANKER_POOLED | 32 | 1,579,675 | 65.71% | 23.75% | 9.81% |
| STAGE3_RANKER_ROUND1_DATA | 12 | 2,663,341 | 70.30% | 76.95% | 25.84% |
| STAGE3_RANKER_ROUND1_DATA | 16 | 2,511,323 | 68.24% | 58.32% | 16.34% |
| STAGE3_RANKER_ROUND1_DATA | 24 | 2,090,360 | 66.31% | 36.24% | 11.84% |
| STAGE3_RANKER_ROUND1_DATA | 32 | 1,589,489 | 65.74% | 24.91% | 10.57% |

## H. The ranking-accuracy wall

The calibration study that produced this design also measured how far the approach can go. Pairwise ordering accuracy on held-out calibration candidate sets:

| Model | Accuracy |
| --- | ---: |
| Stage 1 winner score | 60.3% |
| linear, 19 causal features | 66.6% |
| linear, 37 causal features | 67.7% |
| linear, 45 features (adds request scope) | 67.7% |
| gradient-boosted trees, 45 features | 69.2% |
| gradient-boosted trees, 13x data and 8x capacity | 68.95% |

Scaling training data 13-fold and model capacity 8-fold moves accuracy by under one point. The limit is the information carried by causal routing history, not the estimator, so a neural network would meet the same wall — which is what Stage 2's diagnostic predicted. There is a structural reason: next-use distance depends on which tokens the model will emit next, and past routing does not determine that.

A non-causal frontier measurement on calibration mapped ranking accuracy to oracle-gap closure by blending the true ordering into the deployed score. Reaching a 10% cost win over Stage 1 needs roughly 70% accuracy at capacities 24 and 32 and 73–75% at capacity 16, and only if the extra accuracy lands on comparisons that straddle the eviction cutoff. Training weighted toward that cutoff was tried directly and made results monotonically worse.

## I. Oracle residual

| Capacity | Stage 3 cost | Oracle | Residual headroom | Stage 1 residual recovered |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 9,470,228 | 8,146,471 | 13.36% | 17.36% [16.53, 18.24] |
| 16 | 7,512,017 | 5,904,787 | 19.96% | 16.21% [15.38, 17.08] |
| 24 | 4,833,005 | 3,294,951 | 29.00% | 13.89% [12.91, 14.80] |
| 32 | 2,978,002 | 1,785,846 | 36.13% | 13.21% [12.12, 14.32] |

## J. Limitations

- Trace simulation of expert residency, misses, admissions and transfers. No end-to-end latency improvement and no hardware speedup is claimed or measured.
- One fixed OLMoE-1B-7B-0924 decode trace over four domains; nothing generalizes automatically to other models, batch sizes or serving stacks.
- The primary variant uses decode-request-boundary information that the Stage 1 baseline did not have. The `STAGE3_RANKER_NO_REQUEST_SCOPE` ablation is the like-for-like comparison and is reported beside it.
- The ranking model is fitted offline on calibration and fixed at evaluation. It does not adapt online, so it cannot track a distribution shift that calibration did not contain.
- The learning target is capped at 32 same-layer events; structure beyond that is invisible.
- Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen stateful workload path; cache trajectories are not regenerated under resampled orderings.
- The offline oracle is a non-causal diagnostic, not a deployable method.

## K. Next recommendation

Bank this as the strongest causal residency policy measured so far, but do not claim the preregistered strong result. The accuracy-wall evidence in section H says further gains will not come from a bigger model on this feature set. Restricted to exactly the information Stage 1 had, Stage 3 changes cost by +3.32% on average across the four non-degenerate capacities. Decode-request-boundary awareness is worth a further +1.08% on average, which is the one new information source this stage introduced. The measured frontier says a materially better residency policy needs information about the tokens the model is about to emit, not a better estimator over routing history. Anything that supplies that — for example a cheap signal derived from the current forward pass before the MoE layers commit — is the next thing worth preregistering, and it is a different experiment from this one.

## Reproducibility

- Stage 3 preregistration: `8afa3f8e7b824a04d5c4e723f2244c16e3078f63ea968b8a437bdf641bb84f16`
- Stage 3 frozen config: `abe983cc889b11e37e8c22f1195bcadd5a4bc67cb2f7f582c1a041cbce28e131`
- Stage 3 evaluation manifest: `c8be1290ed8dfab6cd0b79883c998756f0e46da20f12a66a1cc19e187ffaee64`
- Stage 0 trace: `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`
- Stage 0 archive: `9af9a053502709bff9a33017c4f5b80bc0faa2306bc41d980e7bd2d7274346d2`
- Stage 1 archive: `4539cab504052010c32f0b87571adc90884283ac060784ddef822c01768e5d1b`
- Stage 2 archive: `a0ec2f31263daa6fb304cdfc0f03247db953ec7f676fe101d680e1785aff4cd3`
- Stage 3 source bundle at freeze: `53572c8b9647a18e3e66039bd5a24d399d89f5ba9638c3611f464008115b1f3b`
- Transition models: `d62a14dcb5fbc5b819ac955636a4260c5ea2c676d2d285ce20e2e9682e922752`

## Core answer

Can a learned causal ranking function over raw-scale routing features recover substantially more of the expert-residency oracle gap than simple fixed prediction? On this frozen evidence: `STAGE3_RANKER` changed transfer cost against the Stage 1 winner by 2.85% to 5.75% across capacities 12–32 and closed 21.25% to 25.16% of the original Stage 0 oracle gap, against 9.04% to 11.05% for Stage 1. The preregistered verdict is RACE_STAGE3_PARTIAL_SUCCESS.
