# OLMoE Stage 2A Quantization-Cost Surrogate

## Scope

This stage validates fixed activation-aware replay scores against all 64 frozen Stage-1 expert-domain 4-bit ΔNLL observations. No regressor, coefficient fit, post-hoc formula search, mixed-precision allocation, or model update is used.

## Primary formula

`AOD(l,e,d,b) = Σ_t ||g(l,t,e) · (f_q(h)-f(h))||² / (Σ_t ||y_moe(l,t)||² + 1e-30)`

## AOD validation

- Overall Spearman: +0.122482 [-0.216702, +0.433970]
- Overall Kendall tau: +0.084325 [-0.168066, +0.303823]
- Specificity Spearman: +0.405882 [-0.200000, +0.816831]
- Specificity Kendall tau: +0.316667
- Specificity sign agreement: 0.6250
- Top-domain accuracy: 0.5000 [0.2500, 0.7500]
- Mean/median within-expert domain Spearman: +0.012500 / +0.200000
- Positive per-domain correlations: 3/4
- Improvement over WeightRiskFunctional: +0.030174 [-0.008350, +0.070272]

| Domain | AOD Spearman | 95% grouped-bootstrap CI |
|---|---:|---:|
| General | +0.302941 | [-0.348309, +0.768464] |
| Math | -0.579412 | [-0.856088, -0.105496] |
| Coding | +0.350000 | [-0.231364, +0.806320] |
| Reasoning | +0.667647 | [+0.124915, +0.973019] |

| AOD gate | Outcome |
|---|---:|
| Gate A | FAIL |
| Gate B | FAIL |
| Gate C | PASS |
| Gate D | PASS |
| Gate E | PASS |

## Fixed surrogate comparison

| Surrogate | Overall Spearman | Specificity Spearman | Top-domain accuracy |
|---|---:|---:|---:|
| WeightRiskFunctional | +0.092308 | +0.414706 | 0.5000 |
| WeightRiskRouting | +0.041896 | +0.341176 | 0.5000 |
| Functional specialization alone | +0.094322 | +0.414706 | 0.5000 |
| Routing specialization alone | +0.045033 | +0.341176 | 0.5000 |
| UOD | +0.015293 | +0.261765 | 0.3125 |
| REOD | -0.211081 | -0.305882 | 0.1250 |
| APD | +0.132784 | +0.544118 | 0.5000 |
| AOD | +0.122482 | +0.405882 | 0.5000 |
| GQS | +0.097527 | +0.467647 | 0.5625 |

## Pre-registered gradient fallback

AOD failed at least one gate, so GQS was activated exactly as pre-registered. GQS is primary; GQS2 remains diagnostic only.

- Overall GQS Spearman: +0.097527 [-0.223317, +0.400222]
- GQS specificity Spearman: +0.467647
- GQS top-domain accuracy: 0.5625

## Decision

**SURROGATE_NO_GO**

Neither AOD nor the predefined GQS fallback passed every gate.

The distributionally robust mixed-precision optimizer is not implemented by this stage.
