# Repository guidance

Before changing this experiment or interpreting its results, read:

1. `README.md` for the implemented workflow and metric definitions.
2. `EXPERIMENT_STATUS.md` for the latest validated runs, findings, caveats, and
   next experimental gate.

The current project is a diagnostic study of domain-conditioned expert importance
in OLMoE. Do not implement quantization, mixed-precision allocation, compression,
fine-tuning, or weight modification unless the user explicitly changes the project
stage. Stages 1 through 3D have each been authorized in turn; `EXPERIMENT_STATUS.md`
is the authority on which stage is current and what it may do. The prompt-only, controlled causal, and balanced causal runs are complete
and audited. The balanced panel met its pre-registered STRONG GO rule, so a
separately designed distributionally robust mixed-precision experiment is the next
scientifically justified stage once the user explicitly authorizes it.

The separately authorized RACE residency branch has completed Stage 0
(`RACE_STAGE0_STRONG_GO`), Stage 1 (`RACE_STAGE1_STRONG_GO`) and Stage 2
(`RACE_STAGE2_NO_GO`). `stage3_residency/`, `stage3_residency/stage1_prediction/`
and `stage3_residency/stage2_race/` are immutable audited archives; preserve their
frozen preregistrations, configurations, results, reports and archive hashes.
Stage 2's negative result does not reduce the Stage 0 oracle headroom and must not be
answered by adding a neural predictor, prefetching, or any capability the RACE stages
exclude.

The two audited 100-example snapshots are committed under
`results/expert_domain_importance_prompts_only/` and
`results/expert_domain_importance_with_answers/`. Other generated runs remain
ignored except the frozen balanced preregistration files. Validate raw files when
making numerical claims; do not treat the handoff's rounded tables as a substitute
for `results.json`, CSV, or NPZ data. Never alter the frozen balanced expert panel
in response to masking outcomes.

When a new run is completed and audited, update `EXPERIMENT_STATUS.md` with its
configuration, exact or reconstructed command, artifact location, validation
outcome, main comparisons, limitations, and revised go/no-go decision.
