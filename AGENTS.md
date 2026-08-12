# Repository guidance

Before changing this experiment or interpreting its results, read:

1. `README.md` for the implemented workflow and metric definitions.
2. `EXPERIMENT_STATUS.md` for the latest validated runs, findings, caveats, and
   next experimental gate.

The current project is a diagnostic study of domain-conditioned expert importance
in OLMoE. Do not implement quantization, mixed-precision allocation, compression,
fine-tuning, or weight modification unless the user explicitly changes the project
stage. The present next step is controlled validation and expert masking/ablation.

The two audited 100-example snapshots are committed under
`results/expert_domain_importance_prompts_only/` and
`results/expert_domain_importance_with_answers/`. Other generated runs remain
ignored. Validate the committed raw files when making numerical claims; do not
treat the handoff's rounded tables as a substitute for `results.json`, CSV, or NPZ
data.

When a new run is completed and audited, update `EXPERIMENT_STATUS.md` with its
configuration, exact or reconstructed command, artifact location, validation
outcome, main comparisons, limitations, and revised go/no-go decision.
