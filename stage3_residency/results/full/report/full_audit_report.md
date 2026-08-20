# RACE Stage 0 full audit report

Overall: **PASS**

- PASS — oracle_validation_passed
- PASS — all_five_budgets_each_regime
- PASS — oracle_never_exceeds_best_simple
- PASS — headroom_formula
- PASS — required_figures_exist
- PASS — report_exists

Trace hash: `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`
Frozen config hash: `7ce228983b6547d61341757234e77ca7f59a4d0ba53b1e04b64243e9b2ea0971`
Reported decision state: `RACE_STAGE0_STRONG_GO`

Source/base commit recorded by the preregistration:
`48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`.

Actual Stage 0 runtime commit recorded by the trace:
`0f70c61131b877dd9c297663886d563d9e27f55b`.

The pilot frozen/evaluation/audit artifacts are archived, but the raw pilot trace
directory is absent. This is an archival limitation, not a failure of the
validated full run; the complete full raw trace is present and independently
validated.

Bootstrap intervals are conditional on the frozen workload ordering and reweight
per-sequence contributions; stateful cache trajectories are not regenerated under
reordered bootstrap workloads.

Results concern simulated expert residency/miss counts; no end-to-end latency
improvement is claimed.

This audit covers artifact mechanics, raw-trace integrity, decision-driving policy
replay, calibration replay, and recomputed tables. The report and critical
artifact hashes are sealed in
`stage3_residency/reports/final_archive_manifest.json`.
