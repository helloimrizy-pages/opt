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

## Reports

- `reports/optimizer_state_related_work.md` — novelty / overlap audit
- `reports/optimizer_state_stage0_baseline.md` — baseline reproduction
- `reports/optimizer_state_stage1.md` — full Stage 1 report and verdict
