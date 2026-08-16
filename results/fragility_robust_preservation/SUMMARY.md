# Stage 2C: Fragility-weighted robust specialist preservation

Generated: 2026-08-16T04:14:38.886130+00:00

Stage 2C balances predicted residual domain vulnerability
(ResidualRisk_d = q_norm[d] * (1 - Coverage_d)) instead of raw
specialist coverage. Fragility comes only from domain-level uniform
base-precision calibration NLL; no expert-level delta-NLL surrogate is
used, and the frozen Stage 2A SURROGATE_NO_GO and Stage 2B
ROBUST_PRESERVATION_NO_GO decisions are preserved unchanged.

Storage numbers are exact projected format bytes for QDQ-simulated
formats; no runtime speedup, latency, or measured-memory claim is made.

## Calibration fragility

- 4to8 (base 4-bit), regime valid: True
  - general: BF16 NLL 2.919788, base NLL 2.963314, relative fragility +0.014907, normalized 0.7957
  - math: BF16 NLL 2.274910, base NLL 2.305951, relative fragility +0.013645, normalized 0.7283
  - coding: BF16 NLL 1.834358, base NLL 1.900811, relative fragility +0.036227, normalized 1.9337
  - reasoning: BF16 NLL 1.910118, base NLL 1.929523, relative fragility +0.010159, normalized 0.5423
- 3to8 (base 3-bit), regime valid: True
  - general: BF16 NLL 2.919788, base NLL 3.204725, relative fragility +0.097588, normalized 1.3165
  - math: BF16 NLL 2.274910, base NLL 2.431876, relative fragility +0.068999, normalized 0.9308
  - coding: BF16 NLL 1.834358, base NLL 1.941175, relative fragility +0.058232, normalized 0.7856
  - reasoning: BF16 NLL 1.910118, base NLL 2.047042, relative fragility +0.071684, normalized 0.9671

- Fragility SHA-256: `82db7846ebfdedd62c6d2403a3af2869de507449dc7c42e63790e697291bb624`

## Frozen allocations

- New Fragility-Robust allocations: 8
- Reused frozen Stage 2B comparators: 108
- Registry SHA-256: `b1b6b9a68c0840e60b1d080678ed7a8fb7f56a0595100a76223c4f3860b52caf`
- Stage 2B registry SHA-256: `b0221262f0e51700cc16fa5e6a681f63ab6507a9d768714f853f3dfc3f87aa34`

## Seed-45 development split

- Seed: 45, 50 examples/domain, 64 measured positions/example
- general: prior pool fully excluded = True; disjointness verified = True
- math: prior pool fully excluded = False; disjointness verified = True
- coding: prior pool fully excluded = False; disjointness verified = True
- reasoning: prior pool fully excluded = False; disjointness verified = True

## Development results (seed 45, 20% budget)

### 4to8, 20% protection budget

| Method | General | Math | Coding | Reasoning | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|
| Average-Specialization | +0.002951 | +0.010261 | +0.031682 | +0.000840 | +0.011433 | +0.031682 |
| Coding-Only | +0.008512 | +0.006511 | +0.007570 | +0.007882 | +0.007619 | +0.008512 |
| Fragility-Robust | +0.002909 | +0.007772 | +0.025616 | +0.000588 | +0.009221 | +0.025616 |
| General-Only | +0.004537 | +0.005466 | +0.024148 | +0.006014 | +0.010041 | +0.024148 |
| Global-Importance | +0.006105 | +0.008122 | +0.012976 | +0.002371 | +0.007394 | +0.012976 |
| Math-Only | +0.008661 | +0.007315 | +0.023910 | +0.004551 | +0.011110 | +0.023910 |
| Reasoning-Only | +0.006584 | +0.009464 | +0.023444 | +0.003771 | +0.010816 | +0.023444 |
| Robust-Functional | +0.002540 | +0.009196 | +0.033205 | +0.001057 | +0.011500 | +0.033205 |
| Robust-Routing | +0.008446 | +0.016333 | +0.032752 | +0.008229 | +0.016440 | +0.032752 |
| Random (seed 1001) | +0.010079 | +0.014053 | +0.017145 | +0.012596 | +0.013468 | +0.017145 |
| Random (seed 1002) | +0.009298 | +0.015564 | +0.035226 | +0.009745 | +0.017458 | +0.035226 |
| Random (seed 1003) | +0.007324 | +0.014782 | +0.031262 | +0.011207 | +0.016144 | +0.031262 |
| Random (seed 1004) | +0.008403 | +0.012097 | +0.037524 | +0.012670 | +0.017674 | +0.037524 |
| Random (seed 1005) | +0.004026 | +0.007248 | +0.029213 | +0.004526 | +0.011253 | +0.029213 |

### 3to8, 20% protection budget

| Method | General | Math | Coding | Reasoning | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|
| Average-Specialization | +0.054626 | +0.043016 | +0.029847 | +0.039313 | +0.041701 | +0.054626 |
| Coding-Only | +0.057522 | +0.042257 | +0.023281 | +0.036448 | +0.039877 | +0.057522 |
| Fragility-Robust | +0.052080 | +0.048390 | +0.031713 | +0.043764 | +0.043987 | +0.052080 |
| General-Only | +0.041932 | +0.038412 | +0.070645 | +0.042501 | +0.048372 | +0.070645 |
| Global-Importance | +0.054973 | +0.036076 | +0.037885 | +0.028921 | +0.039464 | +0.054973 |
| Math-Only | +0.057564 | +0.032078 | +0.061639 | +0.035838 | +0.046780 | +0.061639 |
| Reasoning-Only | +0.057432 | +0.036660 | +0.059646 | +0.022285 | +0.044006 | +0.059646 |
| Robust-Functional | +0.050717 | +0.045563 | +0.026763 | +0.042048 | +0.041273 | +0.050717 |
| Robust-Routing | +0.061539 | +0.057313 | +0.039520 | +0.053210 | +0.052895 | +0.061539 |
| Random (seed 1001) | +0.070041 | +0.057517 | +0.058937 | +0.061879 | +0.062093 | +0.070041 |
| Random (seed 1002) | +0.068502 | +0.052620 | +0.073698 | +0.052343 | +0.061791 | +0.073698 |
| Random (seed 1003) | +0.068369 | +0.057836 | +0.064454 | +0.051920 | +0.060645 | +0.068369 |
| Random (seed 1004) | +0.064807 | +0.059675 | +0.056422 | +0.060549 | +0.060363 | +0.064807 |
| Random (seed 1005) | +0.064243 | +0.046195 | +0.041227 | +0.053512 | +0.051294 | +0.064243 |

**Stage 2C decision: FRAGILITY_ROBUST_NO_GO**

- 4to8: gate_a=PASS, gate_b=PASS, gate_c=FAIL, gate_d=PASS, gate_e=FAIL
- 3to8: gate_a=FAIL, gate_b=PASS, gate_c=PASS, gate_d=PASS, gate_e=FAIL

## Independent audit

- Passed: True
- Checks: 750 passed, 0 failed

