# RACE Stage 3 exploration notes — how the design was found

This is the calibration-only record behind `race_stage3_report.md`. It exists so that
the report's claims — especially section H, the ranking-accuracy wall — are backed by
numbers and by runnable code rather than by prose.

**Reproducing any of this.** From the repository root:

```bash
stage3_residency/stage3_ranking/exploration/run_exploration.sh wall.py
stage3_residency/stage3_ranking/exploration/run_exploration.sh frontier.py
```

Seed policies come from the committed Stage 3 calibration selection rather than from
scratch files, and `exploration/committed.py` asserts that the exploration and frozen
feature vectors are elementwise identical before reusing them.

**Status.** Supporting material, outside the sealed archive. The frozen path is sealed
by `final_archive_manifest.json`; this document and `../exploration/` are not part of
it. Nothing here influenced any threshold: the Stage 3 preregistration was sealed
before the model was fitted on the full calibration split, and every number below comes
from calibration data.

**Data discipline.** The 80 frozen Stage 0 calibration sequences were split again:
`calA` = sequences 0–39 (fit), `calB` = sequences 40–79 (measure). No evaluation
sequence was read at any point during exploration. Reference costs on `calB`:

| Capacity | strongest simple (lfu_decay) | Stage 1 winner | offline oracle |
| ---: | ---: | ---: | ---: |
| 12 | 297,137 | 290,256 | 242,559 |
| 16 | 242,208 | 232,415 | 175,831 |
| 24 | 163,604 | 153,844 | 100,454 |
| 32 | 105,187 | 97,319 | 55,739 |

Stage 1 closes 12.6 / 14.8 / 15.5 / 15.9% of the oracle gap here (mean 14.68%). Gap
closure on `calB` runs higher than on the evaluation split, so `calB` was used for
*ranking* designs against each other, never for predicting the evaluation number.

---

## 1. What the target actually looks like

Measured over 1,167,764 candidate observations at capacity 24 on `calB`
(`exploration/diag.py`). Capped next-use distance among the experts the Stage 1 policy
actually had to choose between:

- quantiles 1/5/10/25/50/75/90/95/99 = 1, 1, 1, 2, **5**, 11, 24, 33, 33
- mean 8.77; only 7.18% saturate at the cap of 33
- about 9.8 distinct distances per candidate set of ~24

Per-horizon discriminative power, on the same candidate sets:

| Horizon | mean survival `S_h` | pairwise ordering accuracy alone |
| --- | ---: | ---: |
| H1 | 0.161 | 59.51% |
| H2 | 0.278 | 60.30% |
| H4 | 0.450 | 60.33% |
| H8 | 0.665 | 59.17% |
| H16 | 0.840 | 57.41% |
| H32 | 0.928 | 55.98% |

This single table explains the first four failures below: the long horizons are nearly
saturated, so they carry almost no ordering information and a great deal of estimation
noise.

---

## 2. Four principled estimators that lost (`exploration/run1.py`)

Oracle gap closed on `calB`, mean over capacities 12–32. Negative means worse than the
strongest simple policy.

| Estimator | mean gap closed |
| --- | ---: |
| `E[min(d,33)]` from the six Markov horizons | **−18.86%** |
| the same, combining request rows by noisy-OR | −4.28% |
| per-expert inter-arrival hazard model | −5.71% |
| convex blends of the first and third (β = 0.25 / 0.5 / 0.75) | −4.37 / −6.06 / −10.06% |
| *(Stage 1 winner, for reference)* | *+14.68%* |

The first one is the instructive failure and worth stating precisely. Writing
`E[min(d,33)] = 33 − Σ_{h=1..32} S_h` and interpolating the survival curve linearly
between the six fitted horizons gives fixed coefficients

```
E[min(d,33)] = 33 − ( 1·S₁ + 1.5·S₂ + 3·S₄ + 6·S₈ + 12·S₁₆ + 8.5·S₃₂ )
```

which sum to 32, so the identity is exact. But the weights are largest exactly on H16
and H32 — the two least discriminative, noisiest terms in the table above. The
correct-in-expectation estimator is dominated by its own noise. Its measured pairwise
ordering accuracy is 58.20%, worse than plain H2 alone.

**Lesson kept:** being unbiased for squared error on the distance is not the same as
being good at ordering, and here the two objectives point in opposite directions.

---

## 3. What did work: pairwise ranking on raw-scale features

Held-out pairwise ordering accuracy on `calA` groups, then deployed on `calB`
(`exploration/fit.py`, `run2.py`, `run3.py`, `run4.py`).

| Step | accuracy | mean gap closed on calB | vs Stage 1 (B=12/16/24/32) |
| --- | ---: | ---: | --- |
| Stage 1 winner score | 60.33% | 14.68% | — |
| pairwise-logistic, 19 causal features | 66.61% | 25.81% | +2.38 / +3.50 / +4.36 / +4.56% |
| least squares on capped distance, same features | 65.37% | 18.00% | — |
| expanded to 37 features | 66.79% | 26.37% | +2.57 / +3.63 / +4.43 / +4.86% |
| plus decode-request-boundary features (45) | 67.66% | 28.94% | +2.87 / +4.23 / +5.68 / +6.67% |
| per-capacity, retrained on own trajectory | — | 29.66% | +3.16 / +4.37 / +5.63 / +7.16% |

Two things to note. Fitting the *ranking* objective rather than least squares on the
distance is worth about 7 points of gap closure at equal feature set. And the jump from
the Stage 1 score to the first learned model is by far the largest single step; every
later refinement is small.

Stability under a swapped split (`exploration/reverse.py`):

| Direction | mean gap closed | vs Stage 1 (B=12/16/24/32) |
| --- | ---: | --- |
| fit calA → test calB | 29.73% | +3.12 / +4.36 / +5.80 / +7.22% |
| fit calB → test calA | 29.43% | +3.09 / +4.62 / +6.09 / +7.84% |

---

## 4. Two ideas that sounded right and were wrong

**Truncating the target to the slot-residence horizon** (`exploration/run5.py`). At
capacity `B` a retained expert survives roughly `B/8` events before eviction pressure
returns, so predicting out to 33 looked like the wrong target. Training on
`min(d, T+1)`:

| T | 2 | 3 | 4 | 6 | 8 | 12 | 33 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean gap closed | 17.23% | 23.58% | 25.64% | 27.64% | 28.63% | 29.32% | **29.73%** |

Monotone in `T`. The full capped target is best; the hypothesis was wrong.

**Boundary-weighted training** (`exploration/run6.py`). Only comparisons that straddle
the retention cutoff can change an eviction, so weighting training pairs by proximity
to the true cutoff looked like the obvious win — and it is the direction the Stage 2
report explicitly recommended. Weighting pairs by `exp(−distance-from-cutoff / band)`:

| band | all pairs | 8.0 | 4.0 | 2.0 | 1.0 | 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean gap closed | **29.80%** | 29.52% | 27.01% | 22.90% | 16.15% | 13.26% |

Monotonically worse as the band narrows. **The Stage 2 report's recommendation (ii) was
wrong** and is corrected here. The plausible reason is that narrowing the band discards
most of the training signal while the *predicted* cutoff still has to be right globally,
but that explanation was not itself tested.

---

## 5. The wall (`exploration/ceiling.py`, `exploration/wall.py`)

Is the ~67% plateau a limit of the feature set or of the linear form? Held-out pairwise
ordering accuracy at capacity 24:

| training rows | linear | GBT (31 leaves, 250 iters) | GBT (127 leaves, 800 iters) |
| ---: | ---: | ---: | ---: |
| 85,447 | 66.45% | 68.08% | 67.35% |
| 370,729 | 66.92% | 68.49% | 68.64% |
| 1,115,190 | 66.94% | 68.59% | **68.95%** |

Reproduced from the committed `exploration/wall.py` after the seeds were anchored to
the committed calibration artifacts:

| training rows | linear | GBT (31, 250) | GBT (127, 800) |
| ---: | ---: | ---: | ---: |
| 85,438 | 66.49% | 67.99% | 67.31% |
| 371,688 | 66.92% | 68.45% | 68.59% |
| 1,116,209 | 66.95% | 68.59% | **68.95%** |

The reproduction differs from the original in the third significant figure because the
seed policy is now the committed round-1 pooled model, fitted on all 80 calibration
sequences, rather than the original scratch model fitted on `calA` alone. A different
seed policy visits slightly different candidate sets. The conclusion is unchanged.

Thirteen times the data and eight times the model capacity buy **under one accuracy
point**. Gradient-boosted trees beat the linear model by about 1.5 points and then stop
too. The plateau is a property of the information, not of the estimator — which is why
the report states that a neural network would meet the same wall, and why no neural
model was built.

The structural reason: next-use distance depends on which tokens the model is about to
emit, and those are produced by a forward pass that has not happened yet. Past routing
constrains the current context but not the next token.

---

## 6. The achievability frontier (`exploration/frontier.py`)

A non-causal diagnostic, never a policy. Inside each candidate set the deployed causal
score is blended with the true next-use ordering at mixing weight λ, which traces the
map from ranking accuracy to oracle-gap closure. On `calB`:

| λ | B=12 | B=16 | B=24 | B=32 |
| ---: | --- | --- | --- | --- |
| 0.0 | 70.4% → 29.2% | 68.6% → 30.0% | 66.8% → 29.6% | 66.2% → 30.1% |
| 0.2 | 72.9% → 38.1% | 71.7% → 44.3% | 70.4% → **52.3%** | 70.0% → **59.3%** |
| 0.4 | 82.1% → 65.1% | 80.6% → **71.1%** | 78.7% → 75.1% | 78.2% → 77.6% |
| 0.5 | 87.1% → **78.3%** | 85.7% → 81.6% | 83.8% → 82.0% | 83.1% → 83.4% |
| 1.0 | 100% → 100% | 100% → 100% | 100% → 99.9% | 100% → 99.3% |

Bold marks the first row meeting a 10% cost win over Stage 1, which on `calB` requires
65.8 / 49.8 / 39.8 / 35.6% gap closure at capacities 12 / 16 / 24 / 32.

Read it carefully: +3.6 *blended* accuracy points buy +23 gap points at capacity 24,
whereas the +6.5 real accuracy points earned in section 3 bought only +11. Blended
accuracy is worth roughly six times more per point because it lands precisely on the
cutoff-straddling comparisons. That is what motivated the boundary-weighted training in
section 4 — and section 4 is why that motivation did not survive contact with a
measurement.

---

## 7. Why Condition A is the binding constraint

Stage 2's Condition A asks for a 10% cost cut against Stage 1; Condition B asks for 30%
of the original Stage 0 oracle gap. They are not equally hard, because Stage 1 already
sits within 1.6–4.2% of the strongest simple policy:

| Capacity | gap closure needed for Condition A | for Condition B | Stage 1 today | Stage 3 achieved |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 64.4% | 30% | 9.04% | 24.83% |
| 16 | 47.1% | 30% | 10.69% | 25.16% |
| 24 | 36.4% | 30% | 11.05% | 23.41% |
| 32 | 30.1% | 30% | 9.26% | 21.25% |

Condition A is about twice as demanding as Condition B at every capacity. That
asymmetry, measured on calibration before the Stage 3 evaluation ran, is why the Stage 3
ladder added a PARTIAL rung between "10% cost win" and "failure" rather than reporting a
2.1–2.8x improvement over Stage 1 as a bare failure. Stage 2's criteria are still
reported verbatim in the Stage 3 report so the two stages stay comparable.

---

## 8. What was not tried

Stated so the record is not mistaken for an exhaustive search:

- clustering the routing state so the context model sees expert interactions rather than
  a bag-of-experts average of transition rows;
- conditioning on token identity or on token clusters;
- cross-layer context from the earlier layers of the same token;
- fixed-share or any other tracking variant of the online weighting from Stage 2;
- any signal derived from the current forward pass before the MoE layers commit, which
  is the one direction the frontier in section 6 suggests could actually clear the wall,
  and which is a different experiment needing its own preregistration.
