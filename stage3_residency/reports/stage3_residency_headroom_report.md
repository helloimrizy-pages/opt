# RACE Stage 0: Expert Residency Oracle-Headroom Study

## Executive result

**`RACE_STAGE0_STRONG_GO`**

The validated full run finds substantial algorithmic headroom between the
strongest eligible simple expert-residency policy and the future-aware offline
optimum. This result authorizes a separate RACE design stage; it does not claim
that RACE itself will work.

## Exact reason

The oracle reduced primary unit expert-miss cost by at least 15% at capacities
12, 16, 24, and 32 in every stationary, abrupt-shift, repeated-shift, and mixed
workload regime. This is STRONG-GO support at 4/5 preregistered cache budgets in
all four regimes. All passing 95% bootstrap interval lower bounds exceed 17%.

Capacity 8 has zero headroom because OLMoE top-k is eight and mandatory atomic
admission leaves no post-request residency choice at that capacity.

| Regime | C=8 | C=12 | C=16 | C=24 | C=32 |
|---|---:|---:|---:|---:|---:|
| stationary | 0.00% | 17.75% | 26.55% | 37.59% | 45.48% |
| abrupt | 0.00% | 17.56% | 26.46% | 37.67% | 45.67% |
| repeated | 0.00% | 17.66% | 26.42% | 37.44% | 45.65% |
| mixed | 0.00% | 18.24% | 27.27% | 38.64% | 46.73% |

The complete baseline tables, workload-level confidence intervals, routing
diagnostics, limitations, and figures are in the frozen detailed report at
`stage3_residency/results/full/report/stage3_residency_headroom_report.md`.

## Validated evidence

- Model: `allenai/OLMoE-1B-7B-0924`, revision
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- Hardware/precision: NVIDIA A40, bfloat16.
- Decode: greedy, seed 42, up to 128 new tokens, prompt prefill excluded.
- Trace: 400 prompts, 51,112 generated tokens, 817,792 atomic token/layer
  events, and 6,542,336 requested experts.
- Evaluation: 600 conditions, 4,800 result rows, and 82,800 per-sequence rows.
- Calibration: first 20 prompts/domain only; evaluation uses the remaining 80
  prompts/domain. LFU-decay `alpha=0.95` and Static Hotset scores reproduce
  exactly from the disjoint calibration trace.
- Oracle: generalized farthest-future matches exact dynamic programming on 6,690
  exhaustive and 500 fixed-seed random tiny atomic cases at every preregistered
  lambda, with maximum cost difference 0.0.
- Raw audit: every one of the 400 prompt-chunk hashes validates; their ordered
  arrays reproduce the aggregate trace byte-for-byte; the decision-driving
  LFU-decay and oracle conditions replay exactly from raw requests for every
  workload and capacity.

All frozen sanity checks pass: event accounting, capacity invariants, oracle
dominance, oracle cache monotonicity, top-k capacity equivalence, unlimited- and
zero-cache limits, deterministic evaluation replay, calibration/evaluation
separation, byte-cost proportionality, per-sequence aggregation, and random-seed
coverage.

## Provenance

The two commit fields have distinct roles and must not be conflated:

- Source/base commit recorded by the preregistration:
  `48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`.
- Actual Stage 0 runtime commit recorded by the trace:
  `0f70c61131b877dd9c297663886d563d9e27f55b`.

Frozen scientific identifiers:

- Stage 0 source bundle:
  `a88d593a1ab686138b5c62e1be23b7d08200169c5f62826af272c11a5eb287d4`.
- Preregistered full configuration:
  `17017dd4c3019e1ea625d21a7102afefdaa2c03129381f3f403c0184cc6576fc`.
- Full logical trace:
  `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`.
- Frozen evaluation configuration:
  `7ce228983b6547d61341757234e77ca7f59a4d0ba53b1e04b64243e9b2ea0971`.
- Static Hotset scores:
  `f26bef3216f1ed5c5ae6124d993a9d4441443aba6e7842f106e4aacb1eb634e6`.

Exact file-level hashes for the report and critical artifacts are frozen in
`stage3_residency/reports/final_archive_manifest.json`; its own checksum is in
`stage3_residency/reports/final_archive_manifest.sha256`.

## Archival and statistical limitations

The pilot frozen configuration, evaluation, and audit outputs remain archived,
but the raw `stage3_residency/traces/pilot/` directory is absent. Consequently,
the pilot cannot be independently replayed from the current checkout. This is an
archival limitation, not a failure of the validated full run: the complete full
raw trace is archived, logically hashed, reconstructed from all prompt chunks,
and independently replayed.

Bootstrap intervals are conditional on the frozen workload ordering and reweight
per-sequence contributions; stateful cache trajectories are not regenerated under
reordered bootstrap workloads.

The result is conditional on one checkpoint, one deterministic decode panel,
sequential prompt workloads, and independent per-layer caches. Abrupt-shift
headroom is similar to stationary headroom, so the result demonstrates general
future-aware residency headroom rather than a uniquely shift-driven effect.

Results concern simulated expert residency/miss counts; no end-to-end latency
improvement is claimed.

No defensible host-device expert-transfer benchmark was collected. Because all
experts have equal parameter size, the byte-weighted result is exactly
proportional to the unit-miss result and is not an independent latency model.

## Next action

Proceed to design RACE as a separately authorized stage. Do not claim that RACE
will work or that dynamic scheduling improves inference latency until an online
method and end-to-end runtime experiment actually demonstrate those outcomes.
