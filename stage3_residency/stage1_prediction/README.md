# RACE Stage 1 — Simple Prediction Headroom Test

This isolated stage asks whether fixed, inexpensive causal routing predictors plus
one common retention rule capture most of the validated Stage 0 offline-oracle
advantage. It does **not** implement RACE, prefetch experts, retrain the router, or
modify model weights.

The experiment references the exact Stage 0 full trace, calibration split,
workloads, simple-baseline results, and offline oracle. Stage 0 remains sealed by
`../reports/final_archive_manifest.json`; Stage 1 does not duplicate or alter the
trace archive.

## Frozen semantics

Residency is independent per MoE layer. Every event is one atomic top-8 set: all
hits and misses are computed against the same pre-event cache, all misses are
mandatory admissions, and the post-event cache contains the complete current
request plus selected old residents. Prediction changes only which old residents
are retained. It never creates a speculative admission.

Candidate ranking is shared by all causal predictors:

1. prediction score descending;
2. LRU recency descending;
3. expert ID ascending.

The primary cost is the Stage 0 unit expert-miss/transfer count at lambda zero.
Capacity 8 remains a sanity check; its zero oracle-gap denominator is reported as
`N/A` and it is excluded from the verdict.

## Preregistration and reproducibility

The outcome-sensitive grids, selection rule, suite aggregation, and verdict
thresholds were frozen in
`configs/stage1_preregistered.json` before calibration or evaluation. Its adjacent
SHA-256 sidecar is the authoritative preregistration seal.

Commands:

```bash
stage3_residency/stage1_prediction/scripts/run_tests.sh
stage3_residency/stage1_prediction/scripts/run_calibration.sh
stage3_residency/stage1_prediction/scripts/run_stage1_full.sh
stage3_residency/stage1_prediction/scripts/analyze_stage1.sh
stage3_residency/stage1_prediction/scripts/run_full.sh
```

Each command resolves paths relative to the repository and records its effective
configuration. `run_full.sh` performs tests, calibration/freeze, evaluation,
analysis, final audit, and archive sealing in that order.

## Causality boundary

Transition matrices and selected hyperparameters use only the 80 frozen Stage 0
calibration sequences. During evaluation, causal scores use the current and past
events in the same layer/cache stream plus immutable calibration artifacts. The
lookahead and perfect-score policies are explicitly non-causal diagnostics.

Bootstrap intervals reweight saved per-sequence contributions conditional on the
frozen workload ordering; stateful cache trajectories are not regenerated under
reordered bootstrap workloads.

Results concern simulated expert residency/miss counts; no end-to-end latency or
hardware speedup is claimed.

## Validated full result

Final verdict: **`RACE_STAGE1_STRONG_GO`**.

Calibration selected Gate-EWMA alpha 0.95, Markov-H horizon 2, Hybrid beta 0.50,
and the global verdict policy
`markov_plus_ewma_h2_beta0.5_alpha0.95`. The decision-driving aggregate is:

| Capacity | Stage0 best | Selected causal | Oracle | Gap closed | Residual headroom |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12,849,725 | 12,849,725 | 12,849,725 | N/A | 0.00% |
| 12 | 9,907,430 | 9,748,279 | 8,146,471 | 9.04% | 16.17% |
| 16 | 8,052,402 | 7,822,852 | 5,904,787 | 10.69% | 23.82% |
| 24 | 5,303,005 | 5,081,058 | 3,294,951 | 11.05% | 33.68% |
| 32 | 3,299,634 | 3,159,525 | 1,785,846 | 9.26% | 41.63% |

Both STRONG-GO conditions pass at 4/4 non-degenerate capacities. The full run has
1,050 condition rows, 144,900 per-sequence rows, and 14 prediction-quality rows.
All sanity checks pass. Exact finite-lookahead validation covered 37,052 enumerated
cases, 300 random cases, and 873,444 action checks with zero difference from the
tiny exact solver.

Frozen hashes:

- preregistration: `cea89fba235aef8338774d85da0687ac06b61d269f16fd6238c89a3b60ef9524`;
- Stage 1 frozen config file: `c595b061a415a9874d83f167e1e0fb7873ec8850ace8acc1a5db17f814351b83`;
- transition model: `b8804d3ce5f87a6a129bdc9aa895a941c2343e5246659d506fcb01805d6b68a7`;
- full results: `77ee9a4e1b34d3c6deb7bec30d2292d53ca19cfb30985600d5b2e6aead2beb00`;
- per-sequence results: `5b4fdc8089be00d8c54c2eb70fc4604be281659ad02316afc349f47c6a5360c2`.

The first report-generation attempt failed on a reporting-only key mismatch after
the full results were already complete and hashed. The simulation was not changed
or rerun. `reports/post_evaluation_reporting_fixes.json` records the exact repair,
unchanged result hashes, and a reconstruction of the pre-fix source-bundle hash.
