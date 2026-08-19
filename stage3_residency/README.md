# RACE Stage 0: expert-residency oracle headroom

This isolated stage asks one question:

> Given real OLMoE decode routing traces and constrained per-layer expert
> residency, how much lower is offline-optimal expert-transfer cost than the
> strongest simple implementable residency policy?

It does **not** implement RACE, prediction, online optimization, regret
minimization, mirror descent, learning, RL, quantization, compression,
fine-tuning, or weight modification.  The repository's earlier Stage 2A
`SURROGATE_NO_GO`, Stage 2B `ROBUST_PRESERVATION_NO_GO`, Stage 2C
`FRAGILITY_ROBUST_NO_GO`, and the separate Stage 3/3D artifacts are read-only.

Current execution status: infrastructure and tests are complete.  The current
host exposes neither CUDA nor MPS, so the real 10-prompt/domain pilot and the
100-prompt/domain full run are pending a CUDA machine.  No aggregate prompt trace
or synthetic result is substituted for decode data.  See
[`reports/stage3_residency_headroom_report.md`](reports/stage3_residency_headroom_report.md).
The sealed local validation and config/source hashes are in
[`reports/implementation_manifest.json`](reports/implementation_manifest.json).

## Reused repository components

[`IMPLEMENTATION_NOTE.md`](IMPLEMENTATION_NOTE.md) records the inspection findings.
The implementation reuses the pinned model loader, structural MoE discovery,
validated router-output parser, domain loaders, seed/device helpers, atomic I/O,
and sequence-level bootstrap conventions.  Existing Stage 1/2 NPZs are
`[example, layer, expert]` teacher-forced aggregates and cannot preserve atomic
decode order.

## Layout

```text
stage3_residency/
  IMPLEMENTATION_NOTE.md
  README.md
  configs/
    smoke.json
    pilot.json
    full_preregistered.json
    *.sha256
  src/residency_headroom/
    trace.py                 # versioned compact NPZ schema
    trace_generation.py      # resumable real greedy-decode capture
    workloads.py             # calibration split and four regimes
    policies.py              # Random/LRU/LFU/LFU-decay/Static Hotset
    simulator.py             # atomic per-layer residency simulator
    exact_solver.py          # tiny exact cache-state dynamic program
    oracle.py                # validated generalized farthest-future oracle
    diagnostics.py           # concentration/locality/shift metrics
    statistics.py            # paired sequence bootstrap and decision
    reporting.py             # required tables, plots, report, audit
    transfer_calibration.py  # optional isolated CUDA copy timing
  tests/
  scripts/
  traces/                    # generated and gitignored
  results/                   # generated and gitignored
  reports/                   # durable status/final report location
```

## Atomic event semantics

An event is one generated token at one MoE layer.  Its request is the complete
top-k set `R_t`; OLMoE's validated top-k is eight.  Hits and misses are computed
against one immutable pre-event state.  All missing experts are transferred and
admitted together, and all members of `R_t` remain in the post-event state.  A
policy may retain old cached experts up to capacity but may not prefetch an expert
outside `S_t union R_t`.  There is no arbitrary serialization of the top-8 set.

Expert IDs are layer-local.  A global identity is `(layer_id, expert_id)`, and the
simulator maintains one cache of capacity `C` for each of the 16 MoE layers.  The
grid `C = {8, 12, 16, 24, 32}` therefore corresponds to 12.5%, 18.75%, 25%, 37.5%,
and 50% of each layer's 64 experts.

Capacity zero exists only as a streaming sanity limit.  A positive capacity below
top-k is invalid.  All scientific runs start with empty caches, so compulsory
loads are included.

Under mandatory admission, `admissions == misses`; with uniform expert costs,
`miss + lambda * admission` is `(1 + lambda) * misses`.  The full lambda grid is
reported to make this explicit, but it cannot change policy rankings and the
primary decision is unit miss cost at `lambda=0`.

## Decode trace schema

`routing_trace.npz` uses schema `olmoe_decode_atomic_routing_v1` and contains only
non-object NumPy arrays:

```text
event_index
sequence_id
domain_id
prompt_index
generated_token_index
layer_index
requested_expert_ids[event, top_k]
router_weights[event, top_k]
token_id
prompt_length
generation_length
```

The adjacent `routing_trace.metadata.json` records prompt/dataset IDs and hashes,
domain mapping, model/checkpoint/precision/tokenizer, generation settings, seed,
architecture, actual per-expert parameter bytes, environment, per-layer
utilization, generated token IDs, schema, and logical content hash.

A generated token is selected greedily from the preceding logits and then fed
through the model with its KV cache; the routing of that token's hidden state is
captured once at every MoE layer.  Prompt-prefill hooks are inactive.  The last
selected token is explicitly forwarded as well, so every recorded generated token
has all 16 routing events.

Generation is checkpointed atomically per prompt.  Resume validates the config,
prompt ID/hash, model revision, and every chunk hash before reuse.

## Calibration, workloads, and simple baselines

The first 20% of each domain's deterministic sequence list is calibration-only;
the remainder is evaluation-only.  Static Hotset uses per-layer frequencies from
that calibration panel.  LFU-decay selects one global alpha from
`{0.90, 0.95, 0.99}` by the sum of calibration misses over all five capacities,
with numeric alpha as a fixed tie break.  All three alphas are still emitted as
diagnostics, but only the globally selected alpha may enter “best simple.”

Evaluation regimes are frozen from source sequence IDs:

- stationary: General, Coding, Math, and Reasoning separately;
- abrupt: General→Coding, Coding→General, General→Math, Math→Reasoning;
- repeated: General→Coding→Math→Reasoning→General, using fixed segment lengths;
- mixed: all evaluation prompts randomly interleaved with seed `20260819`.

Random eviction uses five fixed seeds and is reported as mean ± sample standard
deviation.  It is excluded from strongest-simple selection.  LRU, cumulative LFU,
the selected LFU-decay, and Static Hotset are eligible.  Every policy respects the
same atomic transition and deterministic tie breaking.

## Offline oracle and exact validation

For equal-size/equal-cost experts, the generalized farthest-future policy retains
the full current request plus cached non-requested experts in nearest-next-use
order.  This is the exact offline optimum for the documented transition system.

The independent tiny solver enumerates feasible post-event cache sets and
minimizes both miss-only and `miss + lambda * movement` objectives.  Before any
evaluation is frozen, the scalable oracle must match exact DP on all enumerated
instances with up to four experts (within the validation bounds) and 500 random
instances with up to eight experts, capacity up to four, and up to 15 atomic
events.  Any mismatch aborts the run.  Heterogeneous-cost scalable optimality is
not claimed; current OLMoE expert parameter tensors are expected to be identical
and this is checked from the loaded model.

## Costs and metrics

The primary cost is the number of missing expert transfers.  Secondary fields are
hits, miss rate, admissions, evictions, expert transfers, actual parameter bytes
transferred, admission bytes, and cache churn.  Results are emitted in JSONL with
trace/config hashes and in CSV tables.  Byte cost is explicitly labeled
proportional when all expert sizes are equal.

The optional CUDA benchmark times preallocated expert-sized tensor copies with
warmup, synchronization, repeated samples, pinned/pageable host memory, both
directions, medians, quantiles, and effective bandwidth.  Its label is “measured
host-device expert transfer cost.”  It is not end-to-end inference latency and is
not double-counted with a transfer already represented by a miss.

## Statistical unit and decision

The paired bootstrap resamples complete prompt/decode sequences, never individual
routing events. Within a workload it is stratified by segment and domain. For a
regime summary, one source prompt receives the same bootstrap multiplicity everywhere
it reappears across workload components (notably the abrupt-shift pairs), with
sampling stratified by domain. This prevents one decode trace from being counted as
independent evidence multiple times. In every replicate it reselects the cheapest
eligible simple policy, then compares its summed cost with the aligned oracle cost.
Reports include absolute gap, relative headroom, 95% percentile CI, mean/median
sequence gap, and a paired standardized effect.

The frozen full decision uses unit miss cost and `lambda=0`:

- `RACE_STAGE0_STRONG_GO`: at least one workload regime has point headroom ≥15%
  and CI lower bound ≥5% at at least 3/5 capacities.  The report states whether
  this replicates across regimes.
- `RACE_STAGE0_WEAK_GO`: no STRONG GO, but a regime has point headroom ≥5% with CI
  lower bound >0 at at least 3/5 capacities.
- `RACE_STAGE0_NO_GO`: neither rule is met.

Pilot and smoke configs have `decision_enabled=false`; they can only emit
`PILOT_ONLY_NO_STAGE0_DECISION`.

## Commands

Run all dependency-light Stage 0 tests:

```bash
stage3_residency/scripts/run_tests.sh
```

Reproduce the completed real-checkpoint CPU smoke audit without loading the model:

```bash
PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/audit_smoke.py \
  --trace stage3_residency/traces/smoke/routing_trace.npz \
  --output stage3_residency/reports/real_decode_smoke_audit.json
```

Run the combined prior + Stage 0 suite (244 tests in the validated local
environment):

```bash
PYTHONPATH=stage3_residency/src:src python -m pytest -q tests stage3_residency/tests
```

The test suite covers atomic multi-expert requests, LRU, LFU, LFU-decay, Static
Hotset, random determinism, tie breaking, capacity invariants, exact DP, scalable
oracle agreement, workload construction, trace serialization, generation-hook
lifecycle, metrics, byte accounting, bootstrap determinism, all sanity gates,
figures, and reports.

On the pinned CUDA/BF16 machine, run the mechanics pilot (10 prompts/domain, up to
32 new tokens/prompt):

```bash
stage3_residency/scripts/run_pilot.sh
```

The wrapper expands to these independently resumable commands:

```bash
PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/generate_traces.py \
  --config stage3_residency/configs/pilot.json \
  --output-dir stage3_residency/traces/pilot \
  --cache-dir .hf_cache

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/freeze_evaluation.py \
  --config stage3_residency/configs/pilot.json \
  --trace stage3_residency/traces/pilot/routing_trace.npz \
  --output-dir stage3_residency/results/pilot/frozen

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/run_evaluation.py \
  --trace stage3_residency/traces/pilot/routing_trace.npz \
  --frozen-dir stage3_residency/results/pilot/frozen \
  --output-dir stage3_residency/results/pilot/evaluation

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/analyze.py \
  --trace stage3_residency/traces/pilot/routing_trace.npz \
  --frozen-dir stage3_residency/results/pilot/frozen \
  --evaluation-dir stage3_residency/results/pilot/evaluation \
  --output-dir stage3_residency/results/pilot/report
```

Only after `sanity_checks.json` and `pilot_audit_report.md` pass, generate the full
trace (100 prompts/domain, up to 128 new tokens/prompt):

```bash
stage3_residency/scripts/generate_traces.sh
```

Both full-run wrappers invoke `verify_pilot_gate.py` first and abort unless the
pilot trace is real, its hash matches the frozen pilot, all simulator sanity checks
pass, the analysis audit passes, and the pilot emitted no scientific decision.

Then freeze exact prompt IDs, trace hash, calibration/evaluation IDs, workload
orders, global decay alpha, Static Hotset score hash, and oracle validation **before
evaluation**:

```bash
stage3_residency/scripts/run_full.sh
```

`freeze_evaluation.py` refuses to overwrite a different frozen configuration.
Any post-result change must use a separately named exploratory directory.

Optional transfer calibration on an available RTX 4090/CUDA host:

```bash
PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/measure_transfer.py \
  --trace stage3_residency/traces/full/routing_trace.npz \
  --output stage3_residency/results/full/measured_host_device_transfer.json \
  --warmup 20 \
  --repeats 100
```

## Required generated outputs

The evaluation writes `results.jsonl`, `per_sequence_results.jsonl`, condition
checkpoints, exact hashes/manifests, and `sanity_checks.json`.  Analysis writes:

- Table A (stationary), Table B (abrupt), and Table C (repeated/mixed);
- paired headroom CIs and random mean/standard deviation;
- routing concentration, reuse/locality, autocorrelation, Jaccard, and JS-shift
  diagnostics;
- all three required figures as PNG and PDF; and
- `stage3_residency_headroom_report.md` plus an audit report.

Only a completed, sanity-passing, audited real full run may update
`EXPERIMENT_STATUS.md` with a Stage 0 GO/NO-GO result.
