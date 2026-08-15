# Stage 2C: Fragility-weighted robust specialist preservation

Generated: 2026-08-15T14:42:49.276784+00:00

Stage 2C balances predicted residual domain vulnerability
(ResidualRisk_d = q_norm[d] * (1 - Coverage_d)) instead of raw
specialist coverage. Fragility comes only from domain-level uniform
base-precision calibration NLL; no expert-level delta-NLL surrogate is
used, and the frozen Stage 2A SURROGATE_NO_GO and Stage 2B
ROBUST_PRESERVATION_NO_GO decisions are preserved unchanged.

Storage numbers are exact projected format bytes for QDQ-simulated
formats; no runtime speedup, latency, or measured-memory claim is made.

## Seed-45 development split

- Seed: 45, 50 examples/domain, 64 measured positions/example
- general: prior pool fully excluded = True; disjointness verified = True
- math: prior pool fully excluded = False; disjointness verified = True
- coding: prior pool fully excluded = False; disjointness verified = True
- reasoning: prior pool fully excluded = False; disjointness verified = True

## Independent audit

- Passed: True
- Checks: 61 passed, 0 failed

