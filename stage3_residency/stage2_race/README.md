# RACE Stage 2 — Adaptive Multi-Horizon Future-Reuse Ranking

Final verdict: **`RACE_STAGE2_NO_GO`**.

This stage implements and evaluates the first actual RACE algorithm. It asks one
question:

> Can an online causal algorithm combine multiple imperfect future-reuse signals and
> adapt their importance over time so that expert eviction decisions recover
> substantially more of the offline-oracle advantage than the frozen Stage 1
> predictor?

On the frozen Stage 0 trace and the frozen Stage 1 evaluation paths, the answer is no.
The frozen primary variant `RACE_ONLINE` cost **1.06%–1.98% more** expert transfers than
the frozen Stage 1 winner at capacities 12–32 and closed **3.17%–6.38%** of the original
Stage 0 oracle gap, against **9.04%–11.05%** for Stage 1. The full report, all tables,
figures and the diagnostic analysis are in `reports/race_stage2_report.md`.

Stage 2 does not implement RL, train a neural network, use Transformer/RNN/LSTM/MLP
predictors, prefetch speculatively, touch the KV cache, quantize anything, or change
OLMoE routing. It does not alter the Stage 0 oracle or any Stage 1 result.

## What was frozen before anything ran

`configs/stage2_preregistered.json`
(`0cb2ce8ab260094934310edc188dc925811ccefb3ca11912a3aae8f361f09aef`, sealed by the
adjacent `.sha256`) fixes the adviser pool, the rank normalization, the eviction rule,
`H_max`, the delayed-feedback protocol, both loss definitions, the static-weight
optimizer, the learning-rate grid, the weight scope, the reset rule, the metric
definitions, every success threshold, the regression flag, the diagnostics and the
claim scope. Nothing in it was edited after calibration began.

Stage 2 verifies rather than trusts its inputs. `reports/stage2_dependency_manifest.json`
re-checks eleven Stage 0/Stage 1 files against their recorded SHA-256 values before any
simulation, including the Stage 1 archive manifest
`4539cab504052010c32f0b87571adc90884283ac060784ddef822c01768e5d1b`.

## Algorithm

```
frozen causal routing signals
        ↓  nine advisers: MARKOV_H{1,2,4,8,16,32}, GATE_EWMA, LFU_DECAY, PERSISTENCE
midrank percentile normalization over the current eviction candidate set
        ↓  z in [0,1], 1.0 = retain most strongly, exact adviser ties preserved
delayed Hedge adviser weighting, w_j <- w_j exp(-eta * loss_j)
        ↓  S_e = sum_j w_j z_{j,e}
the UNCHANGED Stage 1 eviction rule: S desc, LRU recency desc, expert ID asc
```

The Markov horizons 1–16 are bitwise identical to the frozen Stage 1 transition archive
at its stored precision; horizon 32 is newly fitted on the identical calibration path
with the identical frozen fitting function. `GATE_EWMA` reuses Stage 1's frozen
alpha 0.95 and `LFU_DECAY` reuses Stage 0's frozen alpha 0.95.

A Stage 2 variant that places all adviser weight on one pool member reproduces the
corresponding frozen Stage 1 single-predictor cost **exactly**, and the same mechanism
driven by exact next-use scores reproduces the Stage 0 offline oracle exactly. Both are
enforced by tests and by the pilot audit, and together they prove the eviction mechanism
was not changed.

## Causality

The learning target is the capped next-use distance `min(d, 33)` in same-layer events.
A decision at same-layer event `q` becomes a pending example and is resolved only at
event `q + 32`, once every follow-up event its label depends on has actually been
observed. The measured minimum update-minus-decision offset is exactly 32 at every
capacity, the mean and maximum feedback delay are both exactly 32, and 0.18% of examples
remain unresolved at stream end and are discarded rather than labelled. The offline
ranking observer is a one-way sink: enabling it leaves every simulated cost
bit-identical.

## Commands

```bash
stage3_residency/stage2_race/scripts/run_tests.sh                       # 18 + 19 + 61 tests
stage3_residency/stage2_race/scripts/build_dependency_manifest.py       # verify Stage 0/1
stage3_residency/stage2_race/scripts/fit_models.py                      # calibration-only Markov fit
stage3_residency/stage2_race/scripts/run_pilot.py                       # 9-check causal audit
stage3_residency/stage2_race/scripts/run_calibration.py --workers 12    # select and freeze
stage3_residency/stage2_race/scripts/run_evaluation.py  --workers 12    # one frozen pass
stage3_residency/stage2_race/scripts/analyze_stage2.py                  # tables, figures, report
stage3_residency/stage2_race/scripts/freeze_archive.py                  # seal
```

`RACE_STAGE2_PYTHON` and `RACE_STAGE2_WORKERS` drive `scripts/run_full.sh`, which runs
the whole chain in order.

## Calibration outcome

Calibration used only the 80 frozen Stage 0 calibration sequences. It selected
`eta = 0.1`, uniform initialization, the unweighted rank loss, and therefore
`RACE_ONLINE` as the frozen primary variant. The frozen configuration is
`results/calibration/stage2_frozen_config.json`
(`2ace8993dbe545080364223423768451b2d367b0221c6c93b2eb37f94ec1b2a9`).

## Headline result

| Capacity | Stage 1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 9,748,279 | 9,999,390 | 9,919,727 | 9,851,604 | 9,833,069 | 8,146,471 |
| 16 | 7,822,852 | 8,120,618 | 8,005,697 | 7,945,937 | 7,939,359 | 5,904,787 |
| 24 | 5,081,058 | 5,333,296 | 5,230,074 | 5,181,473 | 5,245,722 | 3,294,951 |
| 32 | 3,159,525 | 3,335,492 | 3,233,051 | 3,203,094 | 3,323,031 | 1,785,846 |

The ablation chain is monotone and explains the outcome completely. Averaged over the
four non-degenerate capacities: uniform rank aggregation costs 4.23% more than the
Stage 1 winner; calibration-learned static weights recover 1.80%; online adaptation
recovers a further 0.82%; adding the frozen Stage 1 winner itself as a tenth adviser
recovers a further 1.77%, which finally reaches 0.30% better than Stage 1. Every step
is small and the total barely reaches parity, far below the preregistered 10%
STRONG_SUCCESS threshold.

The reason is representational, not adaptive. The Stage 1 winner is a *raw-scale* blend
of two pool members; percentile-normalizing those members separately destroys the
magnitude information the blend uses, and no weighting over the normalized pool can
rebuild it. Meanwhile the online learner does its own job essentially perfectly:
empirical per-example adviser regret against the best fixed adviser in hindsight lies
between -0.00004 and +0.00049 over 8,995,644 delayed updates.

## Ablations, all evaluated once on the frozen paths

`RACE_UNIFORM`, `RACE_STATIC`, `RACE_ONLINE`, `RACE_COST` are the primary comparison.
`RACE_STATIC_PERLAYER`, `RACE_ONLINE_GLOBAL`, and the four `*_EXTENDED` variants (which
add the frozen Stage 1 winner as a tenth adviser) are clearly labeled ablations and
never enter the verdict.

## Scope and limitations

Results are trace simulations of expert residency, misses, admissions and transfers.
**No end-to-end latency improvement and no hardware speedup is claimed or measured.**
Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen
stateful workload path; cache and online-learning trajectories are not regenerated under
resampled orderings. `reports/race_stage2_theory_notes.md` proves a delayed-Hedge regret
bound for the exact implemented update and states explicitly that no regret theorem is
claimed for the combined ranking loss or for transfer cost.

`reports/post_evaluation_reporting_fixes.json` records two reporting-only changes made
after the frozen evaluation had completed and been hashed, with every evaluation
artifact hash unchanged and the verdict unchanged.
