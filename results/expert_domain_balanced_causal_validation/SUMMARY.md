# Balanced Causal Validation of Domain-Specialized OLMoE Experts

## Experimental Setup

- Model: `allenai/OLMoE-1B-7B-0924`
- Revision: `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- Hardware and arithmetic: NVIDIA A40, CUDA, BF16, batch size 1
- Controlled corpus: 100 examples/domain, 64 measured positions/example, 6,400 measured positions/domain, shared `Input:\n` prefix
- Intervention: zero one selected expert gate coefficient at measured source positions without rerouting or weight changes
- Primary contrast: target-domain delta NLL minus the mean delta NLL of all three non-target domains
- Uncertainty: 1,000 fixed-seed bootstrap replicates from saved per-example losses

## Pre-Registered Expert Selection

The panel was frozen before masking with fingerprint `50a9eeb1f053385abe67cc94b2c4cc62570caf6d24f562cb8bcf05f5808cf714`. Selection read baseline routing, gate-mass, and functional-contribution artifacts only; masked outcomes were excluded.

| Target | Specialist | Target rank | Margin | Target route coverage | Control | Control margin | Control route coverage |
|---|---|---:|---:|---:|---|---:|---:|
| General | L13/E52 | 1 | 0.074097 | 0.3656 | L13/E36 | 0.001631 | 0.2306 |
| General | L12/E40 | 1 | 0.072506 | 0.4273 | L12/E2 | -0.070439 | 0.2219 |
| General | L3/E24 | 1 | 0.051658 | 0.2602 | L3/E30 | 0.001371 | 0.1938 |
| Math | L8/E11 | 1 | 0.086291 | 0.6609 | L8/E54 | -0.120670 | 0.5059 |
| Math | L12/E63 | 2 | 0.061547 | 0.4806 | L12/E43 | -0.092989 | 0.5128 |
| Math | L2/E4 | 1 | 0.053722 | 0.5658 | L2/E34 | 0.005539 | 0.3064 |
| Coding | L13/E2 | 1 | 0.196400 | 0.8872 | L13/E61 | 0.034029 | 0.4341 |
| Coding | L11/E27 | 1 | 0.181314 | 0.9633 | L11/E7 | 0.037155 | 0.5722 |
| Coding | L10/E56 | 1 | 0.159128 | 0.9537 | L10/E62 | 0.037391 | 0.4645 |
| Reasoning | L13/E20 | 1 | 0.097238 | 0.7255 | L13/E63 | -0.003051 | 0.2770 |
| Reasoning | L11/E48 | 1 | 0.095266 | 0.7053 | L11/E47 | -0.137040 | 0.3708 |
| Reasoning | L14/E33 | 1 | 0.094512 | 0.6887 | L14/E8 | -0.010784 | 0.4561 |

All four domains satisfied the strict pre-registered eligibility tier; no specialist threshold was relaxed. All controls are unique and from the same layer, and all satisfied the primary 25%-of-specialist margin cap. Routing matches are imperfect for several extremely high-coverage specialists and are reported rather than hidden.

## Integrity Validation

Integrity status: **PASSED**.

- Controlled collection fingerprint: `052956d26bac03d54e637c5812b84ae8a37fd7b5f9b51f8558082af2e38cb362`
- Selection-input fingerprint: `6555c24be1d20799239f5e38209083989790706aca5a96716da29ec1e5bcefbd`
- Runtime revision exact: true
- Dataset revisions and input hashes exact: true
- Fresh baseline reproduced all four source baselines: true
- Selected-route counts matched stored routing tensors: true
- Smoke and intervention hook-leak checks passed: true

## General-Specialized Experts

| Specialist | Target delta NLL | Target-minus-mean-other | 95% CI | Specialist-minus-control | 95% CI |
|---|---:|---:|---:|---:|---:|
| L13/E52 | +0.023808 | +0.023669 | [+0.014219, +0.032834] | +0.023510 | [+0.014364, +0.033041] |
| L12/E40 | +0.025583 | +0.025548 | [+0.015766, +0.036389] | +0.024737 | [+0.013438, +0.037379] |
| L3/E24 | +0.017214 | +0.017074 | [+0.009543, +0.025757] | +0.018761 | [+0.011135, +0.027714] |

## Math-Specialized Experts

| Specialist | Target delta NLL | Target-minus-mean-other | 95% CI | Specialist-minus-control | 95% CI |
|---|---:|---:|---:|---:|---:|
| L8/E11 | +0.009694 | +0.009934 | [+0.006073, +0.014179] | +0.010391 | [+0.005817, +0.015093] |
| L12/E63 | +0.010012 | +0.009599 | [+0.004641, +0.015276] | +0.017767 | [+0.012265, +0.023508] |
| L2/E4 | -0.004302 | -0.005188 | [-0.008501, -0.002065] | +0.003979 | [-0.005980, +0.014000] |

## Coding-Specialized Experts

| Specialist | Target delta NLL | Target-minus-mean-other | 95% CI | Specialist-minus-control | 95% CI |
|---|---:|---:|---:|---:|---:|
| L13/E2 | +0.033806 | +0.033304 | [+0.026959, +0.039108] | +0.025168 | [+0.017937, +0.031395] |
| L11/E27 | +0.034239 | +0.035272 | [+0.029272, +0.040984] | +0.032363 | [+0.026175, +0.038602] |
| L10/E56 | +0.028811 | +0.030623 | [+0.024840, +0.035744] | +0.029092 | [+0.023044, +0.034857] |

## Reasoning-Specialized Experts

| Specialist | Target delta NLL | Target-minus-mean-other | 95% CI | Specialist-minus-control | 95% CI |
|---|---:|---:|---:|---:|---:|
| L13/E20 | +0.002778 | +0.002288 | [-0.000717, +0.005230] | -0.001200 | [-0.006551, +0.004338] |
| L11/E48 | +0.008152 | +0.007232 | [+0.003409, +0.010618] | +0.013182 | [+0.009090, +0.017424] |
| L14/E33 | +0.010538 | +0.009208 | [+0.004206, +0.014528] | +0.006336 | [-0.000067, +0.012374] |

## Matched-Routing Controls

| Target | Specialist | Control | Specialist contrast | Control contrast | Difference | 95% CI | Route-frequency gap |
|---|---|---|---:|---:|---:|---:|---:|
| General | L13/E52 | L13/E36 | +0.023669 | +0.000160 | +0.023510 | [+0.014364, +0.033041] | 0.016875 |
| General | L12/E40 | L12/E2 | +0.025548 | +0.000811 | +0.024737 | [+0.013438, +0.037379] | 0.025684 |
| General | L3/E24 | L3/E30 | +0.017074 | -0.001687 | +0.018761 | [+0.011135, +0.027714] | 0.008301 |
| Math | L8/E11 | L8/E54 | +0.009934 | -0.000458 | +0.010391 | [+0.005817, +0.015093] | 0.019375 |
| Math | L12/E63 | L12/E43 | +0.009599 | -0.008168 | +0.017767 | [+0.012265, +0.023508] | 0.004023 |
| Math | L2/E4 | L2/E34 | -0.005188 | -0.009167 | +0.003979 | [-0.005980, +0.014000] | 0.032422 |
| Coding | L13/E2 | L13/E61 | +0.033304 | +0.008136 | +0.025168 | [+0.017937, +0.031395] | 0.056641 |
| Coding | L11/E27 | L11/E7 | +0.035272 | +0.002909 | +0.032363 | [+0.026175, +0.038602] | 0.048887 |
| Coding | L10/E56 | L10/E62 | +0.030623 | +0.001531 | +0.029092 | [+0.023044, +0.034857] | 0.061152 |
| Reasoning | L13/E20 | L13/E63 | +0.002288 | +0.003489 | -0.001200 | [-0.006551, +0.004338] | 0.056055 |
| Reasoning | L11/E48 | L11/E47 | +0.007232 | -0.005950 | +0.013182 | [+0.009090, +0.017424] | 0.041816 |
| Reasoning | L14/E33 | L14/E8 | +0.009208 | +0.002872 | +0.006336 | [-0.000067, +0.012374] | 0.029082 |

## Aggregate Causal Effects

| Scope | Mean specialist contrast | 95% CI | Mean control contrast | 95% CI | Mean specialist-control | 95% CI | Positive specialists |
|---|---:|---:|---:|---:|---:|---:|---:|
| General | +0.022097 | [+0.015570, +0.029472] | -0.000239 | [-0.002121, +0.001678] | +0.022336 | [+0.015488, +0.029732] | 100.0% |
| Math | +0.004781 | [+0.002261, +0.007299] | -0.005931 | [-0.009741, -0.002555] | +0.010712 | [+0.006498, +0.015448] | 66.7% |
| Coding | +0.033066 | [+0.029103, +0.037092] | +0.004192 | [+0.002695, +0.005637] | +0.028874 | [+0.024813, +0.033054] | 100.0% |
| Reasoning | +0.006243 | [+0.003793, +0.008815] | +0.000137 | [-0.001840, +0.002236] | +0.006106 | [+0.002624, +0.009624] | 100.0% |
| All domains | +0.016547 | [+0.014339, +0.018615] | -0.000460 | [-0.001543, +0.000620] | +0.017007 | [+0.014618, +0.019411] | 91.7% |

Aggregate bootstrap intervals condition on the 12 fixed pre-registered specialists and resample examples within domain; they do not estimate checkpoint, dataset-choice, or expert-selection uncertainty.

## Specialization Score vs Causal Sensitivity

| Baseline predictor | Spearman | 95% bootstrap CI |
|---|---:|---:|
| Functional Specialization Margin | +0.753 | [+0.666, +0.804] |
| Routing Specialization Margin | +0.711 | [+0.621, +0.760] |
| Target Routing Frequency | +0.417 | [+0.312, +0.477] |

## Failures / Counterexamples

- Math L2/E4: target masking did not increase NLL; another-domain mean effect was at least as large; specialist-minus-control difference CI includes zero.
- Reasoning L13/E20: target-domain delta NLL CI includes zero; primary contrast CI includes zero; matched control contrast was at least as large.
- Reasoning L14/E33: specialist-minus-control difference CI includes zero.

## Scientific Interpretation

The balanced panel asks whether baseline functional specialization predicts domain-conditioned causal reliance beyond routing frequency alone. The answer under the frozen decision rule is **STRONG GO**: 4 of 4 domains have both a positive mean specialist contrast and a positive mean specialist-minus-control difference; the overall specialist-minus-control effect is +0.017007 nats/token.

This conclusion applies to the fixed 100-example, 64-position controlled subsets of one OLMoE checkpoint. Selected-route masking is a local causal sensitivity test, not expert deletion and not a simulation of quantization.

## Decision

### STRONG GO

The label follows the decision rule frozen in `selected_experts_preregistered.json`; failed experts were not replaced.

## Recommendation

The evidence justifies proceeding to a separately designed, reversible distributionally robust mixed-precision quantization experiment. No quantization, pruning, fine-tuning, or weight modification was performed here.
