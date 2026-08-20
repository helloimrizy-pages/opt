# RACE Stage 0: Expert Residency Oracle-Headroom Study

## Executive result

**RACE_STAGE0_STRONG_GO**

At least one workload regime has >=15% oracle headroom with a paired CI lower bound >=5% at 3/5 or more cache budgets.

The primary quantity is equal-cost expert misses at `lambda=0`. The strongest simple policy is selected separately for each workload/capacity from LRU, LFU, LFU-decay with the globally calibrated alpha `0.95`, and Static Hotset. Random is excluded from that selection.

## Exact reason

- stationary: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).
- abrupt: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).
- repeated: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).
- mixed: STRONG-support budgets [12, 16, 24, 32] (4/5); >=5% positive-CI budgets [12, 16, 24, 32] (4/5).

## Baseline tables

### Table A — Stationary

| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary | 8 | 2992620 | 2992620 | 2992620 | 2992620 | 2992620 | 0.00% [0.00%, 0.00%] |
| stationary | 12 | 2484448 | 2379423 | 2305638 | 2460697 | 1896388 | 17.75% [17.50%, 18.02%] |
| stationary | 16 | 2080803 | 1985804 | 1872290 | 2104284 | 1375283 | 26.55% [26.19%, 26.90%] |
| stationary | 24 | 1432525 | 1373685 | 1230577 | 1536241 | 768001 | 37.59% [37.13%, 37.98%] |
| stationary | 32 | 924365 | 906157 | 763662 | 1067598 | 416358 | 45.48% [45.08%, 45.85%] |

### Table B — Abrupt shifts

| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| abrupt | 8 | 5926143 | 5926143 | 5926143 | 5926143 | 5926143 | 0.00% [0.00%, 0.00%] |
| abrupt | 12 | 4920496 | 4846449 | 4552306 | 4911736 | 3752707 | 17.56% [17.29%, 17.85%] |
| abrupt | 16 | 4124376 | 4133604 | 3693108 | 4227794 | 2715747 | 26.46% [26.01%, 26.85%] |
| abrupt | 24 | 2834117 | 2987061 | 2424479 | 3132030 | 1511098 | 37.67% [37.14%, 38.23%] |
| abrupt | 32 | 1822877 | 2055581 | 1502762 | 2207867 | 816437 | 45.67% [45.21%, 46.11%] |

### Table C — Repeated and mixed workloads

| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeated | 8 | 936595 | 936595 | 936595 | 936595 | 936595 | 0.00% [0.00%, 0.00%] |
| repeated | 12 | 782170 | 783081 | 727303 | 783004 | 598832 | 17.66% [17.17%, 18.18%] |
| repeated | 16 | 659101 | 677328 | 592679 | 677752 | 436094 | 26.42% [25.69%, 27.13%] |
| repeated | 24 | 456465 | 506514 | 390600 | 507670 | 244364 | 37.44% [36.57%, 38.30%] |
| repeated | 32 | 294197 | 358992 | 243124 | 359071 | 132133 | 45.65% [44.80%, 46.41%] |
| mixed | 8 | 2994367 | 2994367 | 2994367 | 2994367 | 2994367 | 0.00% [0.00%, 0.00%] |
| mixed | 12 | 2486751 | 2462189 | 2322183 | 2462671 | 1898544 | 18.24% [17.98%, 18.51%] |
| mixed | 16 | 2083867 | 2106884 | 1894325 | 2106565 | 1377663 | 27.27% [26.91%, 27.64%] |
| mixed | 24 | 1436643 | 1544893 | 1257349 | 1539628 | 771488 | 38.64% [38.22%, 39.04%] |
| mixed | 32 | 930611 | 1071717 | 790086 | 1071934 | 420918 | 46.73% [46.39%, 47.07%] |

Costs above are raw expert misses summed over every workload in the named regime. The plotted normalized cost is misses divided by requested experts.

## Paired bootstrap confidence intervals

The independent unit is a source prompt/decode sequence. Policies use identical fixed traces; workload resampling is stratified by segment/domain, and regime resampling clusters every repeated appearance of the same source prompt. Each replicate reselects the strongest simple policy, making the comparison conservative with respect to simple-baseline selection without treating reused prompts as independent evidence.

| Regime | Cache | Best-simple cost | Oracle cost | Absolute gap | Headroom (95% CI) | Mean/median sequence gap | Paired effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary | 8 | 2992620 | 2992620 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| stationary | 12 | 2305638 | 1896388 | 409250 | 17.75% [17.50%, 18.02%] | 1278.91 / 1307.00 | 4.477 |
| stationary | 16 | 1872290 | 1375283 | 497007 | 26.55% [26.19%, 26.90%] | 1553.15 / 1566.50 | 3.792 |
| stationary | 24 | 1230577 | 768001 | 462576 | 37.59% [37.13%, 37.98%] | 1445.55 / 1437.50 | 3.103 |
| stationary | 32 | 763662 | 416358 | 347304 | 45.48% [45.08%, 45.85%] | 1085.33 / 1078.50 | 2.601 |
| abrupt | 8 | 5926143 | 5926143 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| abrupt | 12 | 4552306 | 3752707 | 799599 | 17.56% [17.29%, 17.85%] | 1249.37 / 1286.00 | 4.229 |
| abrupt | 16 | 3693108 | 2715747 | 977361 | 26.46% [26.01%, 26.85%] | 1527.13 / 1540.00 | 3.539 |
| abrupt | 24 | 2424479 | 1511098 | 913381 | 37.67% [37.14%, 38.23%] | 1427.16 / 1395.00 | 2.889 |
| abrupt | 32 | 1502762 | 816437 | 686325 | 45.67% [45.21%, 46.11%] | 1072.38 / 1070.00 | 2.429 |
| repeated | 8 | 936595 | 936595 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| repeated | 12 | 727303 | 598832 | 128471 | 17.66% [17.17%, 18.18%] | 1284.71 / 1367.50 | 3.908 |
| repeated | 16 | 592679 | 436094 | 156585 | 26.42% [25.69%, 27.13%] | 1565.85 / 1633.50 | 3.297 |
| repeated | 24 | 390600 | 244364 | 146236 | 37.44% [36.57%, 38.30%] | 1462.36 / 1531.50 | 2.684 |
| repeated | 32 | 243124 | 132133 | 110991 | 45.65% [44.80%, 46.41%] | 1109.91 / 1123.00 | 2.336 |
| mixed | 8 | 2994367 | 2994367 | 0 | 0.00% [0.00%, 0.00%] | 0.00 / 0.00 | 0.000 |
| mixed | 12 | 2322183 | 1898544 | 423639 | 18.24% [17.98%, 18.51%] | 1323.87 / 1361.50 | 4.615 |
| mixed | 16 | 1894325 | 1377663 | 516662 | 27.27% [26.91%, 27.64%] | 1614.57 / 1633.50 | 3.937 |
| mixed | 24 | 1257349 | 771488 | 485861 | 38.64% [38.22%, 39.04%] | 1518.32 / 1500.00 | 3.231 |
| mixed | 32 | 790086 | 420918 | 369168 | 46.73% [46.39%, 47.07%] | 1153.65 / 1149.00 | 2.705 |

## Oracle validation

The scalable generalized farthest-in-future policy matched exact dynamic programming on **6690 exhaustive** and **500 random** tiny set-valued cases. Maximum objective difference was `0.0` across lambdas `[0.0, 0.25, 0.5, 1.0]`.

The proof obligation is intentionally limited to equal-size/equal-cost experts. The trace reports whether all OLMoE experts have equal parameter bytes; the scalable oracle must not be relabeled optimal for heterogeneous weights.

## Routing and locality diagnostics

| Workload | Entropy | Top-10 share | Gini | Consecutive Jaccard | Reuse <=10 | Adjacent-segment JS | Domain JS |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_general | 0.968 | 0.298 | 0.278 | 0.319 | 0.838 | NA | NA |
| stationary_coding | 0.841 | 0.553 | 0.596 | 0.335 | 0.861 | NA | NA |
| stationary_math | 0.915 | 0.427 | 0.439 | 0.292 | 0.834 | NA | NA |
| stationary_reasoning | 0.925 | 0.402 | 0.421 | 0.268 | 0.814 | NA | NA |
| general_to_coding | 0.938 | 0.391 | 0.382 | 0.327 | 0.849 | 0.200 | 0.200 |
| coding_to_general | 0.938 | 0.391 | 0.382 | 0.327 | 0.849 | 0.200 | 0.200 |
| general_to_math | 0.958 | 0.336 | 0.312 | 0.306 | 0.836 | 0.100 | 0.100 |
| math_to_reasoning | 0.932 | 0.389 | 0.398 | 0.280 | 0.824 | 0.073 | 0.073 |
| repeated_domain_cycle | 0.958 | 0.344 | 0.316 | 0.306 | 0.836 | 0.132 | 0.137 |
| mixed_interleaved | 0.948 | 0.368 | 0.350 | 0.303 | 0.836 | NA | 0.134 |

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
- Trace hash: `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`
- Frozen evaluation hash: `7ce228983b6547d61341757234e77ca7f59a4d0ba53b1e04b64243e9b2ea0971`
- Decode seed: `42`; bootstrap seed: `20260819`
- Cache capacities per layer: `[8, 12, 16, 24, 32]`

Exact commands and file layout are in `stage3_residency/README.md` and the generated trace/evaluation manifests.

## Next action

Proceed to design RACE. This result does not claim RACE will work.
