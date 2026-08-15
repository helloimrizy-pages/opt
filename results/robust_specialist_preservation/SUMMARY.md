# Stage 2B: Robust specialist preservation under a fixed protection budget

Generated: 2026-08-15T04:49:20.426304+00:00

The method protects domain-specialized expert capacity directly and
maximizes the specialist coverage of the worst-protected domain. It does
not predict per-expert quantization delta NLL and uses no AOD, GQS, APD,
reconstruction-error, or fitted surrogate objective. The Stage-2A
SURROGATE_NO_GO result remains frozen.

Storage numbers are exact projected format bytes and effective
bits/weight for QDQ-simulated formats. No runtime speedup, latency,
GPU-memory, or energy claim is made.

## Calibration

- Calibration seed: 20260815
- Multi-domain budget: 25 examples/domain, 100 total
- Single-domain budget: 100 examples
- Calibration fingerprint: `7fc28a40a24f9c7684d9544f1d1524fd9329b6abfa717ce70248cf440c8632d4`

## Frozen allocations

- Registry entries: 108
- Registry SHA-256: `b0221262f0e51700cc16fa5e6a681f63ab6507a9d768714f853f3dfc3f87aa34`
- Protection fractions: [0.05, 0.1, 0.2, 0.3]
- Random seeds: [1001, 1002, 1003, 1004, 1005]
- 4to8: base 4-bit, protected 8-bit, total increment 3,221,225,472 bytes
- 3to8: base 3-bit, protected 8-bit, total increment 4,026,531,840 bytes

## Held-out splits

- Development: seed 43, 50 examples/domain
- Final: seed 44, 100 examples/domain
- Measured positions/example: 64
- general: prior pool fully excluded = True; disjointness verified = True
- math: prior pool fully excluded = False; disjointness verified = True
- coding: prior pool fully excluded = False; disjointness verified = True
- reasoning: prior pool fully excluded = False; disjointness verified = True

## Independent audit

- Passed: True
- Checks: 1676 passed, 0 failed

