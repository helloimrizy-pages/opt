RACE_STAGE1_STRONG_GO

# RACE Stage 1: Simple Prediction Headroom Test

## A. Executive verdict

The globally calibration-selected causal policy was `markov_plus_ewma_h2_beta0.5_alpha0.95`. Across the frozen ten-workload suite it closed 9.04%–11.05% of the validated Stage 0 oracle gap at capacities 12–32, leaving 16.17%–41.63% of Stage 0 baseline cost between the predictor and oracle. At 3/4 or more non-degenerate capacities, the globally calibration-selected causal predictor closed under 50% of the Stage 0 oracle gap and left at least 10% of Stage 0 baseline cost between itself and the oracle.

## B. Frozen Stage 0 reference

- Verdict: `RACE_STAGE0_STRONG_GO`
- Source/base commit: `48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`
- Actual Stage 0 runtime commit: `0f70c61131b877dd9c297663886d563d9e27f55b`
- Trace logical hash: `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`
- Cache model: 16 independent layer caches, 64 experts per layer, atomic top-8 requests, mandatory admission, and no prefetch.

Validated Stage 0 oracle headroom (%):

| Capacity | Stationary | Abrupt | Repeated | Mixed |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.00% | 0.00% | 0.00% | 0.00% |
| 12 | 17.75% | 17.56% | 17.66% | 18.24% |
| 16 | 26.55% | 26.46% | 26.42% | 27.27% |
| 24 | 37.59% | 37.67% | 37.44% | 38.64% |
| 32 | 45.48% | 45.67% | 45.65% | 46.73% |

## C. Predictor implementations

All causal methods use one eviction mechanism: retain old candidates by prediction score, then LRU recency, then expert ID. Persistence uses the previous same-layer request; LastGate uses the last observed requested-expert gate weight; GateEWMA decays absent experts toward zero; Markov1 and MarkovH use fixed calibration-only binary transition probabilities; and Hybrid combines selected MarkovH with request-indicator EWMA. No method prefetches.

## D. Calibration procedure

The transition models and all selections used only the 80 frozen calibration sequences (`c74926f84f4e95e73c5934ca3b306f011b33f65b6f79613f6c7bcbdbb07a727b`). Evaluation uses the disjoint 320-sequence split. Gate alpha, Markov horizon, hybrid beta, and then one verdict predictor were each selected once by calibration misses summed over capacities 12–32; nothing was reselected by evaluation workload or capacity.

Selected family configurations: `{'persistence': {'method': 'persistence'}, 'last_gate': {'method': 'last_gate'}, 'gate_ewma': {'method': 'gate_ewma', 'alpha': 0.95}, 'markov_1': {'method': 'markov_1', 'horizon': 1}, 'markov_h': {'method': 'markov_h', 'horizon': 2}, 'markov_plus_ewma': {'method': 'markov_plus_ewma', 'horizon': 2, 'beta': 0.5, 'history_alpha': 0.95}}`.

## E. Main results

The complete required tables are in `tables/required_tables.md`; exact per-workload values are in `tables/causal_by_workload.csv`. Table 3 below is the decision-driving frozen-suite aggregate.

| Capacity | Baseline cost | Predictor cost | Oracle cost | Improvement | Gap closed (95% CI) | Residual (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12849725 | 12849725 | 12849725 | 0.00% | N/A — zero Stage 0 oracle gap | 0.00% [0.00, 0.00] |
| 12 | 9907430 | 9748279 | 8146471 | 1.61% | 9.04% [8.43, 9.68] | 16.17% [15.92, 16.42] |
| 16 | 8052402 | 7822852 | 5904787 | 2.85% | 10.69% [10.04, 11.35] | 23.82% [23.45, 24.16] |
| 24 | 5303005 | 5081058 | 3294951 | 4.19% | 11.05% [10.17, 11.87] | 33.68% [33.20, 34.16] |
| 32 | 3299634 | 3159525 | 1785846 | 4.25% | 9.26% [8.14, 10.42] | 41.63% [41.03, 42.22] |

Capacity 8 is reported only as a degenerate sanity condition and is excluded from every decision count.

## F. Oracle-gap closure

The frozen decision counted gap closure below 50% at capacities [12, 16, 24, 32] and residual headroom of at least 10% at capacities [12, 16, 24, 32]. The NO-GO gap-closure trigger held at []; the low-residual trigger held at [].

Figure 2 plots gap closure against spare residency `S=B-8`, including the 50% and 75% thresholds.

## G. Lookahead analysis

- Capacity 12: H=1 → 32.44%, H=2 → 73.65%, H=4 → 97.34%, H=8 → 99.88%, H=16 → 99.94%, H=32 → 99.98%.
- Capacity 16: H=1 → 7.84%, H=2 → 43.18%, H=4 → 83.76%, H=8 → 99.14%, H=16 → 99.89%, H=32 → 99.94%.
- Capacity 24: H=1 → -12.32%, H=2 → 11.74%, H=4 → 48.97%, H=8 → 88.56%, H=16 → 99.13%, H=32 → 99.78%.
- Capacity 32: H=1 → -19.49%, H=2 → 0.84%, H=4 → 32.22%, H=8 → 71.39%, H=16 → 95.36%, H=32 → 99.06%.

These policies are non-causal diagnostics. Each first action is exact for its visible finite horizon under the frozen unit-cost semantics; the full perfect-score rule matches the Stage 0 exact oracle cost on every full condition and on enumerated/random tiny traces.

## H. Predictor-quality analysis

Prediction quality was measured on the frozen mixed workload. The descriptive Spearman association between next-event average precision and mean residency gap closure was 0.881 (p=3.15e-05). This configuration-level association is diagnostic, not an independent-sample hypothesis test.

## I. Residual opportunity

The perfect-score policy uses the same simple retention mechanism and reproduces the full oracle cost, so under this equal-cost Stage 0 model the decision mechanism itself contributes no measurable residual when supplied exact next-use scores. The selected causal-to-lookahead differences therefore diagnose prediction/horizon error rather than speculative transfer timing. Residual size grows or shrinks with spare capacity as shown in Figure 2 and the exact CSVs.

## J. Limitations

- Results are trace simulations of expert residency, misses, admissions, and transfers; no end-to-end latency improvement or hardware speedup is claimed.
- Causal predictors are deliberately simple and fixed; this stage does not test learned neural forecasting.
- The full oracle and limited-lookahead policies are non-causal diagnostic comparators, not deployable methods.
- Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen workload ordering; stateful cache trajectories are not regenerated under reordered bootstrap workloads.
- Workload-suite aggregation sums the ten preregistered paths, so source prompts recurring across regimes receive the same clustered bootstrap multiplicity but contribute once per frozen occurrence.
- The absent raw pilot trace remains a Stage 0 archival limitation; the validated full trace and replay artifacts used here remain intact.

## K. Next action

Proceed to RACE algorithm design.

## Reproducibility hashes

- Stage 1 preregistration: `cea89fba235aef8338774d85da0687ac06b61d269f16fd6238c89a3b60ef9524`
- Stage 1 frozen config file: `c595b061a415a9874d83f167e1e0fb7873ec8850ace8acc1a5db17f814351b83`
- Transition model: `b8804d3ce5f87a6a129bdc9aa895a941c2343e5246659d506fcb01805d6b68a7`
- Evaluation manifest file: `056544558870a78c956e69c0a4d1c34391d745cfdac581375fcf9a05fecf7920`

## Core answer

After giving simple causal routing prediction a strong and fair chance, the selected `markov_plus_ewma_h2_beta0.5_alpha0.95` policy leaves 16.17%–41.63% of Stage 0 simple-baseline cost as unexploited oracle headroom across capacities 12–32.
