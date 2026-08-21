# Optimizer-State Memory under Continual Test-Time Adaptation — Stage 1

**Scope.** This is a *phenomenon-validation* study, not an algorithm. It asks
whether Adam's first moment, second moment and step counter, carried across a
distribution shift during continual test-time adaptation, causally change early
adaptation behaviour. It deliberately does **not** design, implement, tune or
evaluate a new optimizer, does not implement gradient-similarity moment decay,
and does not implement adaptive resets. The only outputs are measurements and a
preregistered go / no-go verdict.

This sub-project is isolated from the OLMoE expert-importance work and from
`stage3_residency/`; it shares no code or results with them and reorganises
nothing outside `optimizer_state_tta/`.

## Verdict namespace

The task brief specifies `RACE_STAGE1_*`. `RACE Stage 1` is already taken in this
repository by `stage3_residency/stage1_prediction` (routing-prediction headroom),
so the verdict is issued as **`OSM_STAGE1_*`** (Optimizer-State Memory) and
always printed together with its `RACE_STAGE1_*` equivalent, so either name
resolves to the same single verdict.

## Design in one paragraph

One continual Tent chain per (corruption order, seed) runs from the RobustBench
source checkpoint through the 15 CIFAR-10-C corruptions at severity 5, 50 batches
of 200 each, one Adam update per batch, updating BatchNorm affine parameters
only, with the official prediction-before-update semantics. At every domain
boundary the exact model checkpoint and the exact Adam snapshot are cloned into
six state interventions — `CARRY_ALL`, `RESET_M_KEEP_V_STEP`,
`RESET_V_KEEP_M_STEP`, `RESET_MV_KEEP_STEP`, `RESET_STEP_ONLY`, `FRESH_ADAM` —
so **every branch begins the target domain with bitwise-identical model weights**
and consumes the identical batches in the identical order. Only optimizer state
differs. A matched pseudo-boundary control branches from one mid-domain
checkpoint into a stationary continuation and a shifted continuation, so the two
conditions differ *only* in the incoming distribution.

Corruption identity and boundary location are oracle information used for
analysis; every such experiment is tagged `ORACLE_BOUNDARY_DIAGNOSTIC`. Target
labels are used for scoring only and never enter adaptation, diagnostics or any
hyper-parameter choice — enforced by a unit test that shuffles the labels and
asserts the adapted weights are bit-identical.

## Preregistration

`configs/optimizer_state_stage1/prereg_stage1.json` fixes the hypotheses, the
primary metric (`early10_accuracy`), the secondary windows, the recovery-time
rule, the statistical plan and the go / no-go thresholds. Its SHA-256 sidecar is
the seal. Nothing outcome-sensitive was chosen after seeing results, and no
transition was dropped for any reason.

## Layout

```
configs/optimizer_state_stage1/   preregistration + sha256 seal
src/optstate/                     env, data, model, tent_core, adam_state,
                                  diagnostics, metrics, stats
scripts/                          runners, analysis, figures, verdict, driver
tests/                            optimizer-state and Tent-semantics unit tests
results/optimizer_state_stage1/   raw/ (JSONL), summary.csv, aggregate.json
figures/optimizer_state_stage1/   publication figures
reports/                          related work, baseline, Stage 1 report
data/                             CIFAR-10-C + checkpoints (git-ignored)
```

## Reproduce

```bash
bash optimizer_state_tta/scripts/run_optimizer_state_stage1.sh
```

The driver creates its own virtual environment, downloads CIFAR-10-C and the
RobustBench checkpoints, runs the unit tests, the toy mechanism check, the
baseline reproduction, the primary experiment, the stationary control, the
`beta1` and learning-rate sweeps, the gradual-shift control, the analysis, the
figures and the verdict. Completed runs are skipped on re-invocation, so it is
resumable. Stages can be skipped with `SKIP_PRIMARY=1`, `SKIP_BETA1=1`, etc.

### Running the grid on GPUs

The grid is 51 independent runs totalling **119,175** Tent optimizer steps
(WRN-28-10, batch 200, forward + backward + Adam step). They share no state, so
they parallelise perfectly:

```bash
# on the machine that already has the data, bundle it for transfer
bash optimizer_state_tta/scripts/stage_assets.sh

# on the GPU host, after copying and extracting the bundle
bash optimizer_state_tta/scripts/setup_gpu_host.sh
N_GPUS=2 WORKERS_PER_GPU=2 bash optimizer_state_tta/scripts/run_optimizer_state_stage1.sh
```

`setup_gpu_host.sh` provisions the environment, stages or downloads the assets,
runs the tests and the determinism check, and finishes with a throughput probe
that prints ms/step and the implied wall-clock for the full grid, so
`WORKERS_PER_GPU` can be tuned from measurement rather than guesswork. This
workload under-occupies a datacentre GPU — the model is 146 MB and the batch is
200 images at 32x32 — so 2-3 workers per device is usually worth roughly another
factor of two.

`scripts/run_parallel.py` is the worker pool underneath; `--dry-run` prints the
plan and the step budget without running anything. It only changes scheduling,
never results.

Two host-specific notes:

- **Staging beats downloading.** The RobustBench checkpoints come from Google
  Drive through `gdown`, which is regularly rate-limited from datacentre IPs.
  `stage_assets.sh` bundles only what this study reads — the 15 standard
  corruptions, `labels.npy` and the checkpoints — and writes a SHA-256 sidecar.
- **`CUBLAS_WORKSPACE_CONFIG`** is set to `:4096:8` before any CUDA context is
  created (`optstate.env.prepare_cuda_determinism`), because cuBLAS only reads
  it when the handle is first created and `use_deterministic_algorithms(True)`
  raises without it.

### Mixing backends

Each run is self-contained, and the matched-branch comparison is internally
valid on whatever device produced it, so a grid assembled from more than one
backend is not wrong — but it is an avoidable confound. Every row in
`summary.csv` carries a `run_device` column and `aggregate.json` reports
`run_devices`, so the composition is always visible. The recommended practice is
to run one complete grid per backend and treat a second backend as an
independent replication rather than merging the two.

## Reports

- `reports/optimizer_state_related_work.md` — novelty / overlap audit
- `reports/optimizer_state_stage0_baseline.md` — baseline reproduction
- `reports/optimizer_state_stage1.md` — full Stage 1 report and verdict
