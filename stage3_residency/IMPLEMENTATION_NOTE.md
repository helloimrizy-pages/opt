# RACE Stage 0 implementation note

## Repository infrastructure reused

This study is isolated from the pre-existing Stage 3 measured-damage and Stage
3D selection-headroom work.  It reuses only read-only, mechanism-level utilities:

- `expert_analysis.modeling.load_model_and_tokenizer`, `discover_moe_layers`, and
  `architecture_metadata` for the pinned OLMoE checkpoint and structural MoE
  discovery;
- `expert_analysis.hooks.extract_routing` for the already validated interpretation
  of router logits, selected expert IDs, selected gate weights, top-k
  normalization, and current Transformers OLMoE outputs;
- `expert_analysis.datasets.load_domain_examples` and the existing four domain
  formatters/revision pins;
- `expert_analysis.hardware` for device selection and deterministic seeding;
- `expert_analysis.io_utils` conventions for atomic JSON/NPZ writes and package
  version capture; and
- the repository's fixed-seed, paired-resampling convention.  Stage 0 resamples
  whole decode sequences, not routing events.

The source snapshot is commit
`48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`.

## Why existing traces cannot be reused

The audited Stage 1/2 routing arrays contain `routing_counts`, `gate_sums`, and
`contribution_sums` with shape `[example, 16, 64]`.  They aggregate teacher-forced
prompt positions.  They do not contain generated-token indices or the atomic
top-8 set chosen by a layer at a particular decode step.  `EXPERIMENT_STATUS.md`
also records `Generation: None; teacher-forced/prompt forward inference only`.
Reconstructing event order from those aggregates would fabricate temporal
locality.  A new decode-only routing trace is therefore necessary.

## Runtime organization and cache identity

The validated checkpoint has 16 MoE layers, 64 experts per layer, and top-8
routing.  Expert 7 in layer 3 and expert 7 in layer 12 are different parameter
tensors, so the simulator uses `(layer_id, expert_id)` as the global identity.
Residency is modeled as one independent cache of capacity `C` **per MoE layer**.
Thus a capacity of 16 means 16 of that layer's 64 experts can remain resident; it
does not mean 16 experts shared globally across all 16 layers.

## Atomic event and transition semantics

For one generated token and one layer, the request is the complete top-k set
`R_t`.  The simulator performs exactly one transition:

1. Observe the pre-event resident set `S_t`.
2. Compute all hits and misses against `S_t`; the cache cannot change while the
   top-k request is being checked.
3. Transfer/admit every missing requested expert.  All experts in `R_t` are
   required to remain in the post-event set.
4. If capacity is exceeded, the policy chooses which experts in `S_t - R_t` to
   retain.  No policy may prefetch an expert outside `S_t union R_t`.
5. Emit `S_{t+1}`, then advance to the next layer event.

Capacities in the scientific grid are all at least top-k.  Capacity zero is
implemented only as a streaming limiting check (every requested expert misses and
nothing persists).  A positive capacity smaller than the atomic request is
invalid.  With the mandatory-admission semantics, the persistent admissions at an
event equal its misses.  Consequently the normalized
`miss + lambda * admission` sensitivity has the same policy ordering for every
lambda when expert costs are uniform.  Stage 0 still reports the preregistered
lambda grid, but the primary decision is unit miss cost at `lambda=0`.

This choice prevents an artificial intra-top-k order from creating hits and gives
the scalable oracle an exact generalized farthest-in-future solution.  The oracle
keeps the complete current request plus the cached, non-requested experts with the
nearest future request.  An exchange argument gives optimality for equal-size,
equal-cost experts; the implementation additionally compares it with an exact
cache-state dynamic program on exhaustive and random tiny set-valued traces.

## Leakage and adaptation controls

Each domain's deterministic trace is split by sequence ID into a calibration
prefix and a disjoint evaluation remainder.  Static Hotset scores are per-layer
frequencies from calibration events only.  LFU-decay's single primary alpha is
selected globally from `{0.90, 0.95, 0.99}` by aggregate calibration miss count
across all five capacities; all three values remain reported as diagnostics.
Evaluation requests are never used for either choice.  LRU, LFU, LFU-decay, and
Static Hotset all start the evaluation stream empty, so no policy receives free
initial transfers.

## Compute finding

The current host is Apple ARM with neither CUDA nor MPS available in the active
PyTorch build. The pinned checkpoint loaded successfully for a real CPU trace smoke
using one prompt/domain and two decode tokens/prompt. It produced 128 complete
atomic layer events and passed trace, event-accounting, oracle-dominance,
cache-monotonicity, and unlimited-cache checks. This validates the capture path but
is far too small for inference. The preregistered 10-prompt/domain pilot and
100-prompt/domain full decode runs still require the documented CUDA command. No
prompt aggregate is substituted for a real decode trace and no scientific result
is invented.
