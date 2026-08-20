# RACE Stage 0: Expert Residency Oracle-Headroom Study

## Executive result

**PILOT_ONLY_NO_STAGE0_DECISION**

Pilot/smoke runs validate mechanics only; the full frozen trace is required.

This is a mechanics/runtime pilot and is not the final scientific dataset. No GO/NO-GO conclusion is authorized from it.

The primary quantity is equal-cost expert misses at `lambda=0`. The strongest simple policy is selected separately for each workload/capacity from LRU, LFU, LFU-decay with the globally calibrated alpha `0.9`, and Static Hotset. Random is excluded from that selection.

## Exact reason

- stationary: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).
- abrupt: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).
- repeated: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).
- mixed: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).

## Baseline tables

### Table A — Stationary

| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary | 8 | 75326 | 75326 | 75326 | 75326 | 75326 | 0.00% [0.00%, 0.00%] |
| stationary | 12 | 62617 | 60058 | 59740 | 62743 | 48240 | 18.80% [17.85%, 19.52%] |
| stationary | 16 | 52838 | 49915 | 49809 | 53659 | 35570 | 27.60% [25.99%, 28.86%] |
| stationary | 24 | 37341 | 34690 | 35295 | 39584 | 21330 | 37.16% [34.71%, 39.42%] |
| stationary | 32 | 25879 | 23367 | 24068 | 27608 | 13277 | 41.55% [37.64%, 45.53%] |

### Table B — Abrupt shifts

| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| abrupt | 8 | 150199 | 150199 | 150199 | 150199 | 150199 | 0.00% [0.00%, 0.00%] |
| abrupt | 12 | 124154 | 123946 | 118822 | 126146 | 95496 | 19.63% [18.91%, 20.37%] |
| abrupt | 16 | 104630 | 105576 | 99048 | 108979 | 69891 | 29.44% [28.17%, 30.52%] |
| abrupt | 24 | 73940 | 76292 | 69874 | 81537 | 41033 | 41.20% [39.32%, 42.65%] |
| abrupt | 32 | 50241 | 53515 | 47196 | 57887 | 24568 | 47.61% [44.56%, 50.20%] |

### Table C — Repeated and mixed workloads

| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeated | 8 | 23949 | 23949 | 23949 | 23949 | 23949 | 0.00% [0.00%, 0.00%] |
| repeated | 12 | 20381 | 20657 | 19545 | 20453 | 15750 | 19.42% [18.19%, 20.65%] |
| repeated | 16 | 17444 | 18215 | 16646 | 18028 | 11751 | 29.41% [27.11%, 31.57%] |
| repeated | 24 | 12662 | 14244 | 12057 | 13815 | 7133 | 40.84% [37.09%, 44.04%] |
| repeated | 32 | 8771 | 10574 | 8546 | 10173 | 4482 | 47.55% [40.87%, 52.10%] |
| mixed | 8 | 75409 | 75409 | 75409 | 75409 | 75409 | 0.00% [0.00%, 0.00%] |
| mixed | 12 | 62733 | 62731 | 60799 | 62814 | 48225 | 20.68% [19.96%, 21.38%] |
| mixed | 16 | 52999 | 53612 | 51122 | 53928 | 35508 | 30.54% [29.52%, 31.51%] |
| mixed | 24 | 37521 | 39422 | 36647 | 39987 | 21307 | 41.86% [40.20%, 43.19%] |
| mixed | 32 | 26495 | 27928 | 25485 | 28334 | 13052 | 48.79% [45.97%, 50.64%] |

Costs above are raw expert misses summed over every workload in the named regime. The plotted normalized cost is misses divided by requested experts.

## Paired bootstrap confidence intervals

The independent unit is a source prompt/decode sequence. Policies use identical fixed traces; workload resampling is stratified by segment/domain, and regime resampling clusters every repeated appearance of the same source prompt. Each replicate reselects the strongest simple policy, making the comparison conservative with respect to simple-baseline selection without treating reused prompts as independent evidence.

| Regime | Cache | Best-simple cost | Oracle cost | Absolute gap | Headroom (95% CI) | Mean/median sequence gap | Paired effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary | 8 | 75326 | 75326 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| stationary | 12 | 59406 | 48240 | 11166 | 18.80% [17.85%, 19.52%] | 348.94 / 359.50 | 3.749 |
| stationary | 16 | 49132 | 35570 | 13562 | 27.60% [25.99%, 28.86%] | 423.81 / 467.50 | 2.961 |
| stationary | 24 | 33944 | 21330 | 12614 | 37.16% [34.71%, 39.42%] | 394.19 / 443.50 | 2.365 |
| stationary | 32 | 22714 | 13277 | 9437 | 41.55% [37.64%, 45.53%] | 294.91 / 310.00 | 2.012 |
| abrupt | 8 | 150199 | 150199 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| abrupt | 12 | 118822 | 95496 | 23326 | 19.63% [18.91%, 20.37%] | 364.47 / 368.50 | 4.753 |
| abrupt | 16 | 99048 | 69891 | 29157 | 29.44% [28.17%, 30.52%] | 455.58 / 476.00 | 3.824 |
| abrupt | 24 | 69785 | 41033 | 28752 | 41.20% [39.32%, 42.65%] | 449.25 / 474.50 | 2.957 |
| abrupt | 32 | 46897 | 24568 | 22329 | 47.61% [44.56%, 50.20%] | 348.89 / 361.00 | 2.437 |
| repeated | 8 | 23949 | 23949 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| repeated | 12 | 19545 | 15750 | 3795 | 19.42% [18.19%, 20.65%] | 379.50 / 405.50 | 5.804 |
| repeated | 16 | 16646 | 11751 | 4895 | 29.41% [27.11%, 31.57%] | 489.50 / 518.50 | 4.905 |
| repeated | 24 | 12057 | 7133 | 4924 | 40.84% [37.09%, 44.04%] | 492.40 / 541.00 | 4.375 |
| repeated | 32 | 8546 | 4482 | 4064 | 47.55% [40.87%, 52.10%] | 406.40 / 433.50 | 3.709 |
| mixed | 8 | 75409 | 75409 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| mixed | 12 | 60799 | 48225 | 12574 | 20.68% [19.96%, 21.38%] | 392.94 / 382.50 | 5.710 |
| mixed | 16 | 51122 | 35508 | 15614 | 30.54% [29.52%, 31.51%] | 487.94 / 487.00 | 4.608 |
| mixed | 24 | 36647 | 21307 | 15340 | 41.86% [40.20%, 43.19%] | 479.38 / 484.00 | 3.834 |
| mixed | 32 | 25485 | 13052 | 12433 | 48.79% [45.97%, 50.64%] | 388.53 / 393.00 | 3.213 |

## Oracle validation

The scalable generalized farthest-in-future policy matched exact dynamic programming on **6690 exhaustive** and **500 random** tiny set-valued cases. Maximum objective difference was `0.0` across lambdas `[0.0, 0.25, 0.5, 1.0]`.

The proof obligation is intentionally limited to equal-size/equal-cost experts. The trace reports whether all OLMoE experts have equal parameter bytes; the scalable oracle must not be relabeled optimal for heterogeneous weights.

## Routing and locality diagnostics

| Workload | Entropy | Top-10 share | Gini | Consecutive Jaccard | Reuse <=10 | Adjacent-segment JS | Domain JS |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_general | 0.955 | 0.322 | 0.324 | 0.280 | 0.820 | NA | NA |
| stationary_coding | 0.789 | 0.628 | 0.672 | 0.360 | 0.885 | NA | NA |
| stationary_math | 0.895 | 0.447 | 0.489 | 0.292 | 0.847 | NA | NA |
| stationary_reasoning | 0.921 | 0.403 | 0.437 | 0.274 | 0.801 | NA | NA |
| general_to_coding | 0.933 | 0.402 | 0.400 | 0.320 | 0.845 | 0.362 | 0.362 |
| coding_to_general | 0.933 | 0.402 | 0.400 | 0.319 | 0.845 | 0.362 | 0.362 |
| general_to_math | 0.951 | 0.347 | 0.341 | 0.286 | 0.824 | 0.154 | 0.154 |
| math_to_reasoning | 0.926 | 0.394 | 0.419 | 0.283 | 0.816 | 0.110 | 0.110 |
| repeated_domain_cycle | 0.959 | 0.326 | 0.309 | 0.295 | 0.828 | 0.294 | 0.280 |
| mixed_interleaved | 0.947 | 0.367 | 0.353 | 0.300 | 0.817 | NA | 0.220 |

Associations with headroom are descriptive: workloads are fixed and cache-budget conditions are not independent replications.

## Cost interpretation and limitations

- These results are trace simulations of expert residency. They are not measured end-to-end inference latency or a runtime speedup.
- The primary unit cost counts one transfer per missing expert. Byte-weighted cost uses actual expert parameter bytes from the loaded checkpoint. If those sizes are equal, byte cost is exactly proportional to miss count.
- Atomic mandatory admission makes admissions equal misses. The reported lambda grid therefore rescales costs and cannot change rankings; it must not be interpreted as an independent latency model.
- Sequence-level bootstrap intervals condition on these prompts, this model revision, this deterministic decode, and the frozen workload construction.
- Static Hotset and LFU-decay use only the disjoint calibration sequences. All evaluation caches start empty, so initial compulsory transfers are counted.
- No defensible CUDA host-device measurement was available in this environment; hardware-weighted cost remains unavailable rather than fabricated.

## Figures

- `figure1_normalized_cost.png`
- `figure2_oracle_headroom.png`
- `figure3_locality_shift_vs_headroom.png`

## Reproducibility

- Source commit: `48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`
- Model: `allenai/OLMoE-1B-7B-0924` at `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- Trace hash: `d1eeecb420d6762e35be158a6dcfb70e1987354a6dcec159f046311137127b0e`
- Frozen evaluation hash: `3b04ac06e44005d856c9b63916a69f310c6539cc13cfcbd79d2c3c1b51ce773e`
- Decode seed: `42`; bootstrap seed: `20260819`
- Cache capacities per layer: `[8, 12, 16, 24, 32]`

Exact commands and file layout are in `stage3_residency/README.md` and the generated trace/evaluation manifests.

## Next action

Run and audit the frozen full GPU trace/evaluation before deciding.
