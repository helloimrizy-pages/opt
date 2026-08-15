# Stage 2B: Robust specialist preservation under a fixed protection budget

Generated: 2026-08-15T05:33:31.181129+00:00

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

## Development results (20% budget)

### 4to8, 20% protection budget

| Method | Protected experts | General | Math | Coding | Reasoning | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust-Functional | 204 | +0.010636 | +0.007839 | +0.031427 | -0.002668 | +0.011808 | +0.031427 |
| Robust-Routing | 204 | +0.014966 | +0.014574 | +0.029866 | +0.004471 | +0.015969 | +0.029866 |
| Average-Specialization | 204 | +0.010016 | +0.009303 | +0.028731 | -0.002481 | +0.011392 | +0.028731 |
| Global-Importance | 204 | +0.011853 | +0.010210 | +0.010774 | +0.000370 | +0.008302 | +0.011853 |
| General-Only | 204 | +0.009258 | +0.007747 | +0.019125 | +0.002142 | +0.009568 | +0.019125 |
| Math-Only | 204 | +0.010772 | +0.009630 | +0.021546 | +0.001099 | +0.010762 | +0.021546 |
| Coding-Only | 204 | +0.013010 | +0.008339 | +0.005998 | +0.006567 | +0.008479 | +0.013010 |
| Reasoning-Only | 204 | +0.012028 | +0.010566 | +0.020019 | -0.000535 | +0.010520 | +0.020019 |
| Random (seed 1001) | 204 | +0.015459 | +0.014508 | +0.014308 | +0.009251 | +0.013382 | +0.015459 |
| Random (seed 1002) | 204 | +0.015465 | +0.015520 | +0.030031 | +0.006375 | +0.016848 | +0.030031 |
| Random (seed 1003) | 204 | +0.016138 | +0.014444 | +0.026544 | +0.006035 | +0.015790 | +0.026544 |
| Random (seed 1004) | 204 | +0.013697 | +0.012828 | +0.032258 | +0.006144 | +0.016232 | +0.032258 |
| Random (seed 1005) | 204 | +0.009279 | +0.008070 | +0.027148 | -0.000649 | +0.010962 | +0.027148 |

### 3to8, 20% protection budget

| Method | Protected experts | General | Math | Coding | Reasoning | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust-Functional | 204 | +0.053559 | +0.042182 | +0.022821 | +0.047640 | +0.041551 | +0.053559 |
| Robust-Routing | 204 | +0.058779 | +0.050730 | +0.037237 | +0.055815 | +0.050640 | +0.058779 |
| Average-Specialization | 204 | +0.050999 | +0.038686 | +0.026749 | +0.043942 | +0.040094 | +0.050999 |
| Global-Importance | 204 | +0.056027 | +0.027183 | +0.029705 | +0.025567 | +0.034621 | +0.056027 |
| General-Only | 204 | +0.037899 | +0.040782 | +0.058810 | +0.032988 | +0.042620 | +0.058810 |
| Math-Only | 204 | +0.052637 | +0.023734 | +0.056605 | +0.030862 | +0.040959 | +0.056605 |
| Coding-Only | 204 | +0.059977 | +0.030752 | +0.018316 | +0.036846 | +0.036473 | +0.059977 |
| Reasoning-Only | 204 | +0.056596 | +0.027697 | +0.060391 | +0.018777 | +0.040865 | +0.060391 |
| Random (seed 1001) | 204 | +0.065336 | +0.052371 | +0.056873 | +0.063202 | +0.059446 | +0.065336 |
| Random (seed 1002) | 204 | +0.061137 | +0.049062 | +0.067192 | +0.052788 | +0.057545 | +0.067192 |
| Random (seed 1003) | 204 | +0.065328 | +0.049971 | +0.058410 | +0.054418 | +0.057032 | +0.065328 |
| Random (seed 1004) | 204 | +0.062781 | +0.056853 | +0.049763 | +0.061776 | +0.057793 | +0.062781 |
| Random (seed 1005) | 204 | +0.063478 | +0.043387 | +0.038883 | +0.049829 | +0.048894 | +0.063478 |

**Development decision: ROBUST_PRESERVATION_NO_GO**

- 4to8: gate_a=FAIL, gate_b=FAIL, gate_c=PASS, gate_d=PASS
- 3to8: gate_a=PASS, gate_b=FAIL, gate_c=PASS, gate_d=PASS

## Independent audit

- Passed: True
- Checks: 2340 passed, 0 failed
