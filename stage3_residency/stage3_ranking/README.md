# RACE Stage 3 — Learned Causal Future-Reuse Ranking

Final verdict: **`RACE_STAGE3_PARTIAL_SUCCESS`**.

Stage 2 returned `RACE_STAGE2_NO_GO` and localized why: percentile rank normalization
applied to each adviser separately destroys the magnitude information the raw-scale
Stage 1 hybrid exploits, and no weighting over a rank-normalized pool can rebuild it.
Stage 3 acts on exactly that finding. It keeps the Stage 1 eviction rule and the
Stage 0 cache semantics unchanged and replaces the retention score with one
calibration-fitted linear ranking function per cache capacity over raw-scale causal
features.

It **more than doubles** the Stage 1 winner's oracle-gap closure, held out on the
disjoint 320-sequence evaluation split, with every paired interval excluding zero and
no workload/capacity cell regressing.

| Capacity | Stage 1 winner | Stage 3 | Oracle | Improvement vs Stage 1 | Oracle gap closed | Stage 1's gap closed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 9,748,279 | 9,470,228 | 8,146,471 | +2.85% [2.70, 3.04] | **24.83%** | 9.04% |
| 16 | 7,822,852 | 7,512,017 | 5,904,787 | +3.97% [3.73, 4.22] | **25.16%** | 10.69% |
| 24 | 5,081,058 | 4,833,005 | 3,294,951 | +4.88% [4.51, 5.22] | **23.41%** | 11.05% |
| 32 | 3,159,525 | 2,978,002 | 1,785,846 | +5.75% [5.24, 6.25] | **21.25%** | 9.26% |

It is **not** a STRONG_SUCCESS. Under Stage 2's criteria, applied verbatim, Condition A
(a 10% cost win over Stage 1) is met at no capacity and Condition B (30% of the oracle
gap) at no capacity. Section H of the report explains, with measurements, why that
threshold is out of reach for any estimator over causal routing history.

## What is frozen

`configs/stage3_preregistered.json`
(`8afa3f8e7b824a04d5c4e723f2244c16e3078f63ea968b8a437bdf641bb84f16`) fixes the feature
families, the model form, the training protocol, the L2 grid and its selection rule,
the variant list, every metric, the verdict ladder, the regression flag and the claim
scope. It was sealed before the model was fitted on the full calibration split, and no
threshold was moved afterwards.

The STRONG and VERY_STRONG rungs are byte-identical to Stage 2 so the stages stay
comparable. A PARTIAL rung was added between "10% cost win" and "failure", because
Stage 2's ladder had nothing in between and the Stage 2 calibration evidence showed
that gap is wide — Condition A is about twice as hard as Condition B at every capacity.
That rung was written before any Stage 3 evaluation ran, and the report states the
Stage 2 outcome verbatim beside the Stage 3 verdict.

## Mechanism proofs

Two equivalences are enforced by tests and re-checked in the pilot audit, and together
they prove Stage 3 changed the score and nothing else:

- a Stage 3 replay driven by the frozen Stage 1 winner score reproduces the frozen
  Stage 1 cost **exactly** at every capacity;
- the same mechanism driven by exact next-use scores reproduces the Stage 0 offline
  oracle **exactly**.

The first of these initially failed on the real trace and the failure was real, not a
tolerance problem: Stage 1 updates its EWMA history *before* scoring, while the Stage 3
feature state is deliberately read before it absorbs the event. The audit scorer now
reproduces Stage 1's own recurrence and update order instead of approximating it.

## Honest accounting of where the gain comes from

| Ablation | Capacity 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 -> Stage 3, **identical information** | +2.57% | +3.30% | +3.58% | +3.84% |
| value of decode-request-boundary awareness | +0.29% | +0.70% | +1.35% | +1.98% |
| value of per-capacity models over one pooled model | +0.17% | +0.07% | -0.28% | -0.30% |
| value of retraining on the deployed policy's trajectory | +0.10% | +0.08% | -0.41% | -1.07% |

Most of the gain is better use of the same information Stage 1 had. Request-boundary
awareness is a genuinely new information source — a serving stack always knows when a
decode request starts, but no earlier RACE stage used it — and it is worth 0.3-2.0%.

Two of the design choices did not pay: per-capacity specialization and the second
training round are within a percent of the simpler pooled round-1 model, and slightly
worse at the two largest capacities. The simpler model would have been the better
choice and the tables say so.

## The wall

The calibration study behind this design also measured how far the approach can go.
Held-out pairwise ordering accuracy on calibration candidate sets:

| Model | Accuracy |
| --- | ---: |
| Stage 1 winner score | 60.3% |
| linear, 19 causal features | 66.6% |
| linear, 45 features | 67.7% |
| gradient-boosted trees, 45 features | 69.2% |
| gradient-boosted trees, **13x data and 8x capacity** | 68.95% |

Scaling training data 13-fold and model capacity 8-fold moves accuracy by under one
point. The limit is the information carried by causal routing history, not the
estimator — a neural network meets the same wall, which is what the Stage 2 diagnostic
predicted. Four principled estimators that ought to have worked all lost to the simple
baseline, including the textbook `E[min(d, 33)]` functional, which weights the two
least discriminative horizons most heavily and is dominated by its own noisiest terms.
Boundary-weighted training, the direction the Stage 2 report recommended, made results
monotonically worse.

## Commands

```bash
stage3_residency/stage3_ranking/scripts/run_tests.sh                    # 18+19+61+18 tests
stage3_residency/stage3_ranking/scripts/build_dependency_manifest.py    # verify Stage 0/1/2
stage3_residency/stage3_ranking/scripts/run_calibration.py              # fit and freeze
stage3_residency/stage3_ranking/scripts/run_pilot.py                    # 7-check audit
stage3_residency/stage3_ranking/scripts/run_evaluation.py --workers 12  # one frozen pass
stage3_residency/stage3_ranking/scripts/analyze_stage3.py               # tables, figures, report
stage3_residency/stage3_ranking/scripts/freeze_archive.py               # seal
```

`RACE_STAGE3_PYTHON` and `RACE_STAGE3_WORKERS` drive `scripts/run_full.sh`.

## Scope and limitations

Trace simulation of expert residency, misses, admissions and transfers. **No
end-to-end latency improvement and no hardware speedup is claimed or measured.** The
model is fitted offline on calibration and fixed at evaluation, so it cannot track a
shift calibration did not contain. The learning target is capped at 32 same-layer
events. Bootstrap intervals are conditional on the frozen stateful workload path.
Stage 0, Stage 1 and Stage 2 archives are byte-identical after this run.
