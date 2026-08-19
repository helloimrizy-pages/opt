# RACE Stage 0: Expert Residency Oracle-Headroom Study

## Executive status

**PENDING_REAL_DECODE_TRACE — NO STAGE 0 DECISION YET**

The simulator, exact solver, scalable-oracle validation, policies, workload
construction, statistics, sanity checks, report generation, and tests are
implemented.  The current host has no CUDA or MPS device.  Existing repository
artifacts contain teacher-forced per-example aggregates rather than generated-token
atomic top-8 requests, so they are scientifically unsuitable for this study.
The combined repository suite passes 244/244 tests (226 prior plus 18 new).
The validated implementation bundle and frozen config hashes are recorded in
`stage3_residency/reports/implementation_manifest.json`.

No oracle-headroom number is reported and none of
`RACE_STAGE0_STRONG_GO`, `RACE_STAGE0_WEAK_GO`, or `RACE_STAGE0_NO_GO` has been
issued.  Run the exact pilot command in `stage3_residency/README.md` on the pinned
CUDA/BF16 checkpoint, audit it, and only then run the frozen full configuration.

The dependency-light oracle audit is complete: generalized farthest-in-future
matched the exact state-space solver on 6,690 exhaustive and 500 fixed-seed random
tiny atomic traces at every preregistered lambda, with maximum cost difference
0.0.  This validates the oracle mechanism, not the missing real-trace result.

A real-checkpoint CPU smoke also passed on four prompts (one/domain) and two decode
tokens each: 8 generated tokens, 128 atomic layer events, and 1,024 requested
experts. Every generated token had all 16 top-8 layer requests; event accounting,
oracle dominance, cache monotonicity, equal 12,582,912-byte expert sizes, and the
unlimited-cache compulsory-load check passed. Its trace hash is
`0d37c56ec1e98eef8bc844d111298bc0b8bf8720dc8379bdfbe8379ae4dd6a2d`.
This deliberately tiny smoke is not used for a headroom estimate or decision.

No CUDA transfer calibration was possible locally. The configured lab host
requires interactive Tailscale reauthentication, so no remote job was launched.
Hardware-weighted latency remains unavailable rather than inferred from nominal
bandwidth.

This status is not evidence for or against RACE and makes no runtime-speedup,
latency, or end-to-end inference claim.
