# OLMoE Stage-1 Quantization Sensitivity Pilot

## Experimental Setup

- Model: `allenai/OLMoE-1B-7B-0924`
- Revision: `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- Runtime: NVIDIA A40, bfloat16
- Frozen inputs: 100 examples/domain, 64 measured positions/example, 6,400 positions/domain
- Intervention: symmetric group-wise expert-only weight QDQ, one expert at a time, group size 128 along input features
- Memory values are projected packed expert storage, not measured runtime memory.
- Uncertainty: 1,000 deterministic bootstrap replicates from per-example losses.

## Frozen Pilot Panel

The panel was selected only from the frozen functional-specialization statistics and controls, with fingerprint `404927664048259fb623a7b3181e811c8f18c68d5e32825b943b056257220af7`.

| Target | Specialist | Frozen functional margin | Rank | Matched control |
|---|---|---:|---:|---|
| General | L13/E52 | 0.074097 | 1 | L13/E36 |
| General | L12/E40 | 0.072506 | 2 | L12/E2 |
| Math | L8/E11 | 0.086291 | 1 | L8/E54 |
| Math | L12/E63 | 0.061547 | 2 | L12/E43 |
| Coding | L13/E2 | 0.196400 | 1 | L13/E61 |
| Coding | L11/E27 | 0.181314 | 2 | L11/E7 |
| Reasoning | L13/E20 | 0.097238 | 1 | L13/E63 |
| Reasoning | L11/E48 | 0.095266 | 2 | L11/E47 |

## Quantization Results

### 4-bit

| Scope | Specialist contrast | 95% CI | Control contrast | Specialist-control | 95% CI |
|---|---:|---:|---:|---:|---:|
| General | +0.00060343 | [-0.00005380, +0.00121732] | -0.00007331 | +0.00067673 | [-0.00006141, +0.00144478] |
| Math | -0.00038530 | [-0.00100304, +0.00020214] | -0.00040960 | +0.00002430 | [-0.00067532, +0.00066798] |
| Coding | +0.00140514 | [+0.00057416, +0.00237507] | +0.00077341 | +0.00063173 | [-0.00020135, +0.00154685] |
| Reasoning | +0.00039157 | [-0.00020519, +0.00098627] | +0.00037625 | +0.00001532 | [-0.00064563, +0.00067173] |
| All domains | +0.00050371 | [+0.00020826, +0.00080889] | +0.00016669 | +0.00033702 | [-0.00004978, +0.00072856] |

Gate results: A=PASS, B=PASS, C=PASS, D=PASS.

Per-pair target effects and specificity contrasts:

| Target | Specialist | Target ΔNLL (95% CI) | Specialist contrast (95% CI) | Control | Target ΔNLL (95% CI) | Control contrast (95% CI) | Specialist-control (95% CI) |
|---|---|---:|---:|---|---:|---:|---:|
| General | L13/E52 | +0.00104046 [+0.00021207, +0.00187619] | +0.00106171 [+0.00020875, +0.00188284] | L13/E36 | -0.00011879 [-0.00069802, +0.00042105] | -0.00021304 [-0.00080452, +0.00037971] | +0.00127475 [+0.00031233, +0.00227697] |
| General | L12/E40 | +0.00025697 [-0.00046688, +0.00097280] | +0.00014514 [-0.00061455, +0.00087062] | L12/E2 | -0.00013633 [-0.00081036, +0.00047236] | +0.00006643 [-0.00069143, +0.00075248] | +0.00007871 [-0.00080790, +0.00095519] |
| Math | L8/E11 | -0.00055330 [-0.00147987, +0.00039154] | -0.00050827 [-0.00152203, +0.00049050] | L8/E54 | -0.00029477 [-0.00094945, +0.00038896] | -0.00086302 [-0.00169846, +0.00004785] | +0.00035474 [-0.00085055, +0.00141549] |
| Math | L12/E63 | -0.00027519 [-0.00095196, +0.00038851] | -0.00026233 [-0.00094165, +0.00042154] | L12/E43 | -0.00040885 [-0.00101674, +0.00017584] | +0.00004382 [-0.00074229, +0.00076819] | -0.00030615 [-0.00125681, +0.00065633] |
| Coding | L13/E2 | +0.00029935 [-0.00094367, +0.00199418] | +0.00064854 [-0.00062220, +0.00228074] | L13/E61 | +0.00061056 [-0.00016198, +0.00128184] | +0.00062778 [-0.00012913, +0.00136487] | +0.00002076 [-0.00135324, +0.00192247] |
| Coding | L11/E27 | +0.00193343 [+0.00072546, +0.00305617] | +0.00216174 [+0.00099103, +0.00332721] | L11/E7 | +0.00068172 [-0.00009432, +0.00149071] | +0.00091904 [+0.00007676, +0.00178467] | +0.00124270 [-0.00034949, +0.00264823] |
| Reasoning | L13/E20 | +0.00044015 [-0.00027480, +0.00120904] | +0.00041111 [-0.00034395, +0.00120398] | L13/E63 | +0.00032858 [-0.00022071, +0.00089898] | +0.00031263 [-0.00029015, +0.00095737] | +0.00009848 [-0.00085815, +0.00106121] |
| Reasoning | L11/E48 | +0.00040290 [-0.00040688, +0.00108760] | +0.00037202 [-0.00044584, +0.00112966] | L11/E47 | +0.00023743 [-0.00035318, +0.00089313] | +0.00043986 [-0.00033554, +0.00122743] | -0.00006784 [-0.00101535, +0.00084387] |

## Correlations

| Bits | Predictor | Outcome | Spearman (95% CI) | Kendall tau (95% CI) | Sign agreement (95% CI) |
|---:|---|---|---:|---:|---:|
| 4 | masking_causal_contrast | quantization_causal_contrast | +0.4235 [+0.0499, +0.6176] | +0.3167 [+0.0167, +0.4833] | +0.6875 [+0.5000, +0.8125] |
| 4 | functional_specialization | quantization_causal_contrast | +0.4324 [+0.0412, +0.6088] | — | — |
| 4 | routing_specialization | quantization_causal_contrast | +0.3382 [-0.0353, +0.5560] | — | — |
| 4 | target_routing_frequency | quantization_causal_contrast | +0.2706 [-0.1353, +0.5206] | — | — |
| 4 | risk_functional | domain_level_quantization_delta_nll | +0.0923 [-0.0923, +0.2508] | — | — |
| 4 | risk_routing | domain_level_quantization_delta_nll | +0.0419 [-0.1346, +0.2176] | — | — |

## Retained Failures and Counterexamples

- 6/8 specialist contrasts were positive; 2/8 had 95% intervals strictly above zero.
- 6/8 specialist-control differences were positive; 1/8 had 95% intervals strictly above zero.
- Non-positive domain aggregate(s): Math -0.00038530.
- Non-positive specialist contrast(s): L8/E11 (Math) -0.00050827, L12/E63 (Math) -0.00026233.
- Controls more target-specific than their specialist: math_L12_E63 -0.00030615, reasoning_L11_E48 -0.00006784.
- The overall specialist-control point difference is positive, but its 95% interval includes zero. Gate C preregistered positivity of the point estimate; interval exclusion was preferred, not required.
- The 3-bit fallback was not triggered because 4-bit passed the measurability Gate D.

## Stage-1 Decision

### GO

4-bit satisfied all four preregistered Stage-1 gates; Stage 2 remains a separate future task.

This file does not authorize or implement Stage 2 or a mixed-precision optimizer.

## Limitations

- Fake quantization measures numerical effects after QDQ but does not provide a low-bit kernel or measured runtime-memory savings.
- Results condition on one checkpoint, the frozen controlled corpora, and 16 selected interventions.
- Bootstrap intervals capture example sampling only; they do not cover checkpoint, dataset, prompt, or expert-selection uncertainty.
- The pilot reuses the causal-validation inputs to test mechanism consistency, not held-out generalization.

## Independent Raw-Artifact Audit

- Result: **PASS**; decision independently recomputed as `GO`.
- Checks performed: 6,223.
- Maximum absolute published-versus-recomputed numeric difference: 1.39e-17.
- The audit rebuilt the final [bit, intervention, domain, example] array from all raw checkpoints, reconstructed masking contrasts from the balanced raw NPZ, and verified CSV/JSON and artifact hashes.
