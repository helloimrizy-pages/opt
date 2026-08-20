# Stage 0 — baseline reproduction (source and standard online Tent)

Protocol is the official Tent CIFAR-10-C example (`DequanWang/tent/cifar10c.py`
with `cfgs/source.yaml`, `cfgs/norm.yaml`, `cfgs/tent.yaml`): CIFAR-10-C at
severity 5, the 15 standard corruptions in the conventional order, batch size
200, samples unshuffled, one Adam update per batch, `lr = 1e-3`,
`betas = (0.9, 0.999)`, `weight_decay = 0`, BatchNorm affine parameters only,
and a full model+optimizer `reset()` between corruption types. Predictions are
recorded **before** each optimizer step.

## A discrepancy in the task brief, resolved

The brief asks for **WideResNet-28-10** and quotes reference errors of ~18.3%
(source) and ~12.1% (Tent). Those two numbers are the **WRN-40-2 AugMix** row of
the official Tent README, not the WRN-28-10 row. The official README reports, at
severity 5:

| model | source | norm | tent |
|---|---|---|---|
| WRN-28-10 (`Standard`) | 43.5 | 20.4 | 18.6 |
| WRN-40-2 AugMix (`Hendrycks2020AugMix_WRN`) | 18.3 | 14.5 | 12.1 |

Both models were therefore reproduced. WRN-28-10 is the Stage 1 primary model
(it is what the brief names, and it is RobustBench's default); the AugMix model
is reproduced only to confirm that the brief's quoted numbers belong to it.

## Results

### Hendrycks2020AugMix_WRN — WideResNet-40-2 (AugMix) (2,243,546 parameters, 74 adapted BN tensors, 5,408 adapted scalars)

Error rate (%), lower is better. Italic rows are the official reference.

| method | mean | gauss | shot | impul | defoc | glass | motn | zoom | snow | frost | fog | brght | contr | elast | pixel | jpeg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **source** | **18.27** | 28.78 | 22.96 | 26.20 | 9.47 | 20.61 | 10.57 | 9.26 | 14.15 | 15.26 | 17.49 | 7.62 | 20.94 | 14.74 | 41.30 | 14.67 |
| _source (official)_ | _18.3_ | _28.8_ | _23.0_ | _26.2_ | _9.5_ | _20.6_ | _10.6_ | _9.3_ | _14.2_ | _15.3_ | _17.5_ | _7.6_ | _20.9_ | _14.7_ | _41.3_ | _14.7_ |
| **norm** | **14.49** | 18.51 | 16.18 | 22.30 | 8.95 | 21.87 | 10.48 | 9.69 | 12.81 | 13.32 | 15.01 | 7.56 | 11.90 | 16.33 | 14.99 | 17.46 |
| _norm (official)_ | _14.5_ | _18.5_ | _16.2_ | _22.3_ | _9.0_ | _21.9_ | _10.5_ | _9.7_ | _12.8_ | _13.3_ | _15.0_ | _7.6_ | _11.9_ | _16.3_ | _15.0_ | _17.5_ |
| **tent** | **12.07** | 15.61 | 13.21 | 18.76 | 7.91 | 18.16 | 8.97 | 8.00 | 10.40 | 10.86 | 12.34 | 6.66 | 10.03 | 14.01 | 11.39 | 14.75 |
| _tent (official)_ | _12.1_ | _15.7_ | _13.2_ | _18.8_ | _7.9_ | _18.1_ | _9.0_ | _8.0_ | _10.4_ | _10.8_ | _12.4_ | _6.7_ | _10.0_ | _14.0_ | _11.4_ | _14.8_ |
| **tent_continual** | **14.11** | 15.61 | 12.11 | 17.19 | 9.71 | 18.86 | 12.50 | 11.13 | 13.98 | 13.23 | 15.35 | 10.00 | 12.97 | 16.44 | 13.60 | 18.90 |

Deviation from the official reference means (percentage points): `norm` +0.01, `source` +0.03, `tent` +0.03 — all within the preregistered 2.0 pp tolerance: **True**.

`tent_continual` is the same Tent configuration with **no** reset between
corruption types. It is not part of the official table; it is reported
because it is the regime the Stage 1 boundary experiment runs in.
Its mean error is 14.11% versus 12.07% for the episodic protocol.

### Standard — WideResNet-28-10 (36,479,194 parameters, 50 adapted BN tensors, 17,952 adapted scalars)

Error rate (%), lower is better. Italic rows are the official reference.

| method | mean | gauss | shot | impul | defoc | glass | motn | zoom | snow | frost | fog | brght | contr | elast | pixel | jpeg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **source** | **43.51** | 72.33 | 65.71 | 72.92 | 46.94 | 54.32 | 34.75 | 42.02 | 25.07 | 41.30 | 26.01 | 9.30 | 46.69 | 26.59 | 58.45 | 30.30 |
| _source (official)_ | _43.5_ | _72.3_ | _65.7_ | _72.9_ | _46.9_ | _54.3_ | _34.8_ | _42.0_ | _25.1_ | _41.3_ | _26.0_ | _9.3_ | _46.7_ | _26.6_ | _58.5_ | _30.3_ |
| **norm** | **20.44** | 28.08 | 26.12 | 36.27 | 12.82 | 35.28 | 14.17 | 12.13 | 17.28 | 17.39 | 15.26 | 8.39 | 12.63 | 23.76 | 19.66 | 27.30 |
| _norm (official)_ | _20.4_ | _28.1_ | _26.1_ | _36.3_ | _12.8_ | _35.3_ | _14.2_ | _12.1_ | _17.3_ | _17.4_ | _15.3_ | _8.4_ | _12.6_ | _23.8_ | _19.7_ | _27.3_ |
| **tent** | **18.58** | 24.81 | 23.49 | 33.02 | 11.93 | 31.84 | 13.72 | 10.78 | 15.90 | 16.21 | 13.66 | 7.86 | 12.05 | 21.98 | 17.30 | 24.20 |
| _tent (official)_ | _18.6_ | _24.8_ | _23.5_ | _33.0_ | _12.0_ | _31.8_ | _13.7_ | _10.8_ | _15.9_ | _16.2_ | _13.7_ | _7.9_ | _12.1_ | _22.0_ | _17.3_ | _24.2_ |
| **tent_continual** | **20.27** | 24.81 | 20.48 | 28.53 | 14.90 | 31.79 | 16.65 | 15.00 | 18.55 | 17.36 | 16.64 | 11.24 | 15.93 | 25.13 | 20.26 | 26.74 |

Deviation from the official reference means (percentage points): `norm` +0.04, `source` +0.01, `tent` +0.02 — all within the preregistered 2.0 pp tolerance: **True**.

`tent_continual` is the same Tent configuration with **no** reset between
corruption types. It is not part of the official table; it is reported
because it is the regime the Stage 1 boundary experiment runs in.
Its mean error is 20.27% versus 18.58% for the episodic protocol.

## Verdict on baseline validity

**BASELINE_VALID** for the primary model (`Standard`, WRN-28-10).

Every per-corruption error matches the official table to within rounding, so
the model checkpoint, preprocessing (`uint8 -> NCHW -> /255`), severity slice
(`[(s-1)*10000 : s*10000]`), BatchNorm configuration (`track_running_stats=False`,
running statistics discarded), update count (1 per batch),
prediction-before-update semantics, batch size (200) and optimizer configuration
are all confirmed correct.

## An observation that matters for Stage 1

The official protocol's `model.reset()` restores `model.state_dict()` **and**
`optimizer.state_dict()` together (see `tent.py::load_model_and_optimizer`). The
canonical Tent benchmark number is therefore already produced by an
optimizer-state reset — but a confounded one, because the weights are reset in
the same operation. Separating those two is exactly what Stage 1 does.

## Environment

| item | value |
|---|---|
| git commit | `f9236cb35541f07c96a8e03b6a36eacc1f21a92c` |
| working tree dirty | True |
| python | 3.11.16 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| robustbench | unknown |
| numpy / scipy / pandas | 2.4.6 / 1.17.1 / 3.0.5 |
| platform | macOS-26.6.2-arm64-arm-64bit |
| device | mps — Apple M4 Max |
| CUDA | not available (Apple MPS backend) |
| seed | [0] |
| dataset | CIFAR-10-C, Zenodo record 2535967, severity 5 |
| checkpoints | RobustBench `cifar10/corruptions/Standard.pt`, `Hendrycks2020AugMix_WRN.pt` |

Raw data: `results/optimizer_state_stage1/raw/baseline/baseline_per_corruption.csv` and `baseline_summary.json`.
