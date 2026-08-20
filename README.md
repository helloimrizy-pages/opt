# OLMoE expert importance across domains

This repository implements a diagnostic experiment for the question:

> Do per-layer expert-importance rankings in OLMoE change across general text,
> mathematics, coding, and reasoning inputs?

The latest validated runs, cross-run comparison, current go/no-go decision, and
next experimental gate are recorded in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).
Future Codex sessions are directed there by the repository-level `AGENTS.md`.

It collects routing utilization, selected gate mass, and a functional-contribution
proxy. The completed diagnostic stages also apply reversible selected-route masks.
The Stage-1 pilot code can temporarily replace one expert's FFN weights with
quantize-dequantize values, but it never saves modified model weights, fine-tunes,
generates from, or permanently compresses the model.

The default checkpoint is the official base model
[allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924).
At runtime the collector discovers MoE blocks, routers, top-k behavior, expert count,
and expert storage layout instead of relying on a fixed module path.

## Metrics

All statistics are computed independently for each MoE layer and domain.
Padding tokens are excluded.

- Routing frequency is the number of valid-token top-k assignments to an expert
  divided by all valid-token top-k assignments in that layer.
- Gate mass is the sum of the selected routing coefficients actually applied to an
  expert, divided by valid token count. OLMoE-1B-7B-0924 uses top-8 routing and, in
  the current Hugging Face implementation, norm_topk_prob=false. Therefore the
  selected weights generally sum to less than one; the collector preserves those
  exact coefficients. A separate normalized expert vector is used for comparisons.
- Functional contribution is the sum of
  L2(gate weight times individual expert output), divided by valid token count.
  OLMoE's tensorized expert output is reconstructed exactly from the same expert
  parameters and routed hidden states inside a removable hook. This adds roughly
  one extra expert-FFN evaluation but stores no hidden states.
- Optional gradient attribution is
  abs(gate weight times d(next-token CE sum)/d(gate weight)). It is enabled with
  --compute-gradient-attribution. Parameters are frozen for gradient allocation,
  while detached input embeddings seed autograd. This mode is much more expensive
  because the backbone activation graph must be retained.

The functional statistic is an activation-magnitude proxy. It is not causal expert
importance.

## Domain data

Reference answers are included by default so the measured coding and math inputs
contain code and worked mathematical text, not only natural-language task
descriptions. Use --prompts-only for a prompt-only sensitivity run.

| Domain | Primary dataset | Split | Automatic fallback |
|---|---|---|---|
| General | Salesforce/wikitext, wikitext-103-raw-v1 | test | EleutherAI/lambada_openai |
| Math | openai/gsm8k, main | test | EleutherAI/hendrycks_math, algebra |
| Coding | google-research-datasets/mbpp, full | test | openai/openai_humaneval |
| Reasoning | allenai/ai2_arc, ARC-Challenge | test | allenai/openbookqa |

Every substitution, dataset fingerprint, available Hub revision, selected example
ID, example count, and token count is written to the domain metadata. HumanEval has
fewer than 500 examples; if that fallback is used, all available examples are used
and the smaller count is recorded.

## Installation

Python 3.10 or newer is required.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

The non-quantized checkpoint has about 6.9B total parameters. Budget approximately
14 GB for BF16 weights, plus runtime memory and Hugging Face cache space. CUDA is
preferred, followed by MPS and CPU. No multi-GPU setup is required.

## Reproducible commands

First validate model loading, all routing invariants, expert-output dimensions,
expert IDs, contribution reconstruction, and hook cleanup on four fixed examples:

    python scripts/collect_expert_importance.py \
      --model allenai/OLMoE-1B-7B-0924 \
      --smoke-only \
      --output-dir results/expert_domain_importance

Run the 100-example/domain quick experiment:

    python scripts/collect_expert_importance.py \
      --model allenai/OLMoE-1B-7B-0924 \
      --domains general math coding reasoning \
      --quick \
      --max-length 512 \
      --seed 42 \
      --output-dir results/expert_domain_importance

Analyze and plot:

    python scripts/analyze_expert_importance.py \
      --input-dir results/expert_domain_importance \
      --bootstrap-replicates 100

    python scripts/plot_expert_importance.py \
      --input-dir results/expert_domain_importance

For the intended 500-example/domain run, omit --quick or pass --num-examples 500:

    python scripts/collect_expert_importance.py \
      --model allenai/OLMoE-1B-7B-0924 \
      --domains general math coding reasoning \
      --num-examples 500 \
      --max-length 512 \
      --batch-size 1 \
      --seed 42 \
      --output-dir results/expert_domain_importance

Completed domains are resumed by default. A domain is written atomically only after
all its examples finish. Configuration fingerprints prevent accidental reuse under
a different model, revision, dtype, seed, sequence length, or sample definition.
Use --no-resume to deliberately rerun completed domains and --overwrite only when
reusing an output directory for a different configuration.

Dataset revisions can be pinned independently:

    --dataset-revision general=REVISION \
    --dataset-revision math=REVISION

## Controlled causal validation

The next project gate is implemented as a single command. It uses a separately
tokenized, identical `Input:\n` prefix for every domain, removes reference answers
and domain-name wrappers, and retains exactly 64 measured content positions plus
one look-ahead next-token label per example. At 100 examples, every domain therefore
has the same 6,400-token measurement and loss budget. The shared prefix and
look-ahead token condition the model but are excluded from expert statistics. This
is a length-conditioned sample: the report records how many candidates in each
dataset were long enough to enter the controlled corpus.

Run this on the A40 RunPod instance:

    python scripts/run_causal_validation.py \
      --model allenai/OLMoE-1B-7B-0924 \
      --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
      --domains general math coding reasoning \
      --num-examples 100 \
      --tokens-per-example 64 \
      --candidate-pool-size 1000 \
      --max-length 512 \
      --batch-size 1 \
      --seed 42 \
      --device cuda \
      --dtype bfloat16 \
      --no-allow-dataset-substitution \
      --dataset-revision general=b08601e04326c79dfdd32d625aee71d232d685c3 \
      --dataset-revision math=740312add88f781978c0658806c59bc2815b9866 \
      --dataset-revision coding=4bb6404fdc6cacfda99d4ac4205087b89d32030c \
      --dataset-revision reasoning=210d026faf9955653af8916fad021475a3f00453 \
      --mask-expert 11:27:coding:reasoning \
      --mask-expert 10:56:coding:general \
      --mask-expert 1:25:coding:general \
      --bootstrap-replicates 100 \
      --split-half-replicates 100 \
      --mask-bootstrap-replicates 1000 \
      --output-dir results/expert_domain_causal_validation

This one invocation validates instrumentation and masking, constructs and verifies
the controlled corpus, collects all three activation metrics, computes the original
cross-domain analysis, estimates repeated same-domain split-half reliability, runs
the three pre-registered expert/domain contrasts, bootstraps loss effects, writes
the summary, and creates all figures. Completed collection domains and individual
loss passes resume automatically.

The 100-example run is the controlled quick experiment and remains preliminary.
Its generated report states whether the evidence supports expanding causal
validation, stopping, or considering only a limited reversible pilot; it does not
authorize quantization automatically.

The intervention zeros the selected expert's gate coefficient only at the measured
source-token positions. It does not reroute the token, change model parameters, or
claim to simulate quantization. The evaluated next-token loss positions are exactly
aligned with the positions used for routing and contribution collection.

Additional causal-validation outputs include:

- `controlled_corpus.json` and `controlled_inputs/DOMAIN.npz`
- `same_domain_split_half.csv`
- `expert_masking_loss.csv`
- `expert_masking_domain_contrasts.csv`
- `masking_results.json` and resumable per-example arrays under `masking/`
- split-half, masking-loss, and proxy-versus-loss figures in PNG and PDF

This output directory remains ignored until the completed run is copied back,
audited, and deliberately promoted as a validated snapshot.

## Balanced causal validation

After the controlled importance tensors and the first three Coding interventions
have been audited, freeze the balanced panel in a separate command. This command
reads only `collection_config.json`, architecture/corpus metadata, controlled
inputs, and baseline `domains/*.npz` tensors. It deliberately does not read any
masked-loss artifact:

    PYTHONPATH=src python scripts/preregister_balanced_causal_panel.py \
      --source-dir results/expert_domain_causal_validation \
      --output-dir results/expert_domain_balanced_causal_validation

The preregistration uses normalized functional contribution and the margin
`I(target) - max(I(non-target))`. The strict specialist tier requires target rank
at most 7 of 64, at least one non-target rank below the top half, positive margin,
and at least 1% target routing coverage. It chooses three layer-diverse experts per
domain. Controls are assigned one-to-one within the same layer by deterministic
minimum-cost matching, primarily on target routing coverage, subject to a target
specialization margin no larger than 25% of the paired specialist's margin. The
complete ranking and exact hashes are frozen in
`selected_experts_preregistered.json` before masking.

The controlled source run remains intentionally ignored by Git. Before using a
fresh RunPod checkout, copy the complete local directory
`results/expert_domain_causal_validation/` into the same path on the pod. Do not
regenerate or substitute it: the runner validates its frozen hashes before loading
the model.

Run the frozen panel on the NVIDIA A40 with:

    PYTHONPATH=src python scripts/run_balanced_causal_validation.py \
      --source-dir results/expert_domain_causal_validation \
      --output-dir results/expert_domain_balanced_causal_validation \
      --model allenai/OLMoE-1B-7B-0924 \
      --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
      --device cuda \
      --dtype bfloat16 \
      --batch-size 1 \
      --seed 42 \
      --bootstrap-replicates 1000 \
      --cache-dir .hf_cache

After the model cache has been populated, `--local-files-only` may be added for a
fully offline resume.

Before panel masking, the runner revalidates every source fingerprint, requires an
A40/CUDA/BF16 runtime, verifies the model revision and architecture, runs the mask
smoke test, reproduces every baseline loss, and freshly reproduces every baseline
routing tensor. It aborts on any mismatch. Every
expert/domain pass is saved immediately under `masking/`; rerunning validates and
skips completed checkpoints. Previously validated Coding interventions are reused
only when the package environment and freshly reproduced baselines are bitwise
identical; otherwise they are rerun automatically.

The primary causal statistic is target delta NLL minus the arithmetic mean of the
three non-target delta NLLs. The analysis uses 1,000 fixed-seed bootstrap replicates
from saved per-example losses and also reports all named pairwise contrasts,
specialist-minus-control paired differences, domain and all-panel aggregates, and
functional/routing predictors of causal specificity. No inference is performed by
the bootstrap analysis.

## Stage-1 quantization sensitivity pilot

The mandatory pilot between causal masking and any mixed-precision allocator is
implemented in `scripts/run_quantization_pilot.py`. It selects the two specialists
with the largest frozen functional-specialization margins in each domain, retains
their already preregistered same-layer routing controls, and freezes the resulting
8-pair/16-intervention panel before reading masking or quantization outcomes.

The intervention applies deterministic symmetric group-wise weight-only fake
quantization to one expert at a time. OLMoE stores `gate_up_proj` and `down_proj` as
three-dimensional `[expert, output, input]` parameters; the runner indexes the
expert axis structurally and groups each two-dimensional expert slice along its
final input-feature dimension. Scales are rounded to FP16 for the simulated format;
maximum, round/clamp, and dequantization calculations use FP32 around those stored
scales. Router, attention, embeddings,
normalization, language-model head, and every unrelated expert remain unchanged.
Each context fingerprints all experts in the affected layer, verifies isolation,
then restores the selected expert bit-for-bit.

Run the dependency-light local smoke path without materializing the full model. If
the pinned checkpoint is present in `.hf_cache`, this also audits its safetensors
headers and applies reversible QDQ to two loaded real expert slices:

    PYTHONPATH=src python scripts/run_quantization_pilot.py --smoke-only

Copy the complete frozen controlled and balanced result directories to the A40;
the three tracked panel-selection files alone are not enough because the runner
revalidates controlled inputs, BF16 baselines, and raw per-example masking changes.
Then run:

    PYTHONPATH=src python scripts/run_quantization_pilot.py \
      --source-dir results/expert_domain_causal_validation \
      --balanced-results-dir results/expert_domain_balanced_causal_validation \
      --output-dir results/expert_quantization_pilot \
      --model allenai/OLMoE-1B-7B-0924 \
      --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
      --device cuda \
      --dtype bfloat16 \
      --group-size 128 \
      --primary-bits 4 \
      --fallback-bits 3 \
      --bootstrap-replicates 1000 \
      --seed 42 \
      --batch-size 1 \
      --cache-dir .hf_cache \
      --resume

The production path requires CUDA/BF16 and repeats a real-checkpoint expert
isolation/restoration smoke test before inference. It evaluates 4-bit first. The
3-bit fallback runs automatically only if the 4-bit result meets the preregistered
too-small-to-measure condition; a clearly negative or otherwise unattractive
4-bit result does not trigger fallback. Every expert/domain/bit-width loss pass is
atomically checkpointed and validated on resume.

The pilot writes the frozen panel, result/pairwise/control/masking-comparison and
distortion CSVs, per-example NPZ arrays, exact projected expert-storage accounting,
five figures in PNG and PDF, `results.json`, `stage1_decision.json`, and `SUMMARY.md`
under `results/expert_quantization_pilot/`. Projected bytes include packed weight
bits and FP16 scales; they are not measured runtime-memory savings. This runner does
not implement Stage 2 or a mixed-precision optimizer.

## Stage 2A activation-aware quantization-cost surrogate

Stage 2A is implemented as a separate validation gate before any mixed-precision
allocation. It asks whether a fixed, cheap replay score predicts the 64 real 4-bit
expert/domain DeltaNLL observations already measured in Stage 1. It does not train a
regressor, tune a coefficient, search formulas, modify a checkpoint permanently, or
implement an allocator.

The primary score is gated output perturbation:

```text
AOD(l,e,d,b) = sum_t ||g(l,t,e) * (f_e(h; W_hat_b) - f_e(h; W))||_2^2
               / (sum_t ||y_moe(l,t)||_2^2 + 1e-30)
```

Only measured tokens actually routed to the expert enter the numerator. A clean
baseline forward captures exact BF16 expert inputs, selected expert IDs, actual gate
coefficients, per-example MoE-output energy, token/example alignment, and small
random replay-validation samples. It stores no attention state or unrelated full
transformer activation. Offline structural reconstruction must match both an
isolated contribution produced by the model's actual expert-container forward path
and the untouched full-forward summed MoE output before Stage 2A can continue.

The same pass also produces fixed UOD, REOD, and APD diagnostics. All pilot QDQ
weights must reproduce the Stage-1 original and quantized expert fingerprints. The
validation uses all 16 experts and all four domains, with 1,000 expert-grouped
bootstrap replicates. If AOD misses any preregistered gate, and only then, the runner
captures one mean-NLL MoE-output gradient per example and evaluates the predefined
GQS fallback. GQS is primary in that branch; GQS2 is diagnostic only.

On a local machine, validate all frozen artifacts and the planned configuration
without loading the 7B checkpoint:

```bash
PYTHONPATH=src python scripts/validate_quantization_cost_surrogate.py \
  --source-dir results/expert_domain_causal_validation \
  --stage1-dir results/expert_quantization_pilot \
  --output-dir results/quantization_cost_surrogate \
  --preflight-only
```

Run the real pilot-surrogate validation on the NVIDIA A40:

```bash
PYTHONPATH=src python scripts/validate_quantization_cost_surrogate.py \
  --source-dir results/expert_domain_causal_validation \
  --stage1-dir results/expert_quantization_pilot \
  --output-dir results/quantization_cost_surrogate \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 1 \
  --group-size 128 \
  --primary-bits 4 \
  --bootstrap-replicates 1000 \
  --seed 42 \
  --replay-chunk-size 512 \
  --cache-dir .hf_cache \
  --resume
```

The command automatically runs the standalone auditor. Its only final decisions are
`AOD_GO`, `SURROGATE_GO_GRADIENT`, and `SURROGATE_NO_GO`. An audit failure blocks GO.
If and only if the decision file contains an independently audited GO, build the
full `[16, 64, 4, 4]` cost matrix for 3-, 4-, 8-, and reference 16-bit weights:

```bash
PYTHONPATH=src python scripts/build_quantization_cost_matrix.py \
  --surrogate-dir results/quantization_cost_surrogate \
  --stage1-dir results/expert_quantization_pilot \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 1 \
  --group-size 128 \
  --bit-widths 3 4 8 16 \
  --replay-chunk-size 512 \
  --seed 42 \
  --cache-dir .hf_cache \
  --resume
```

The full-matrix runner checkpoints by domain, layer, and precision; records zero-route
experts as `cost=0, unobserved=true`; preserves raw, non-renormalized costs; writes
exact payload/FP16-scale storage accounting; checks the 16-bit zero reference; reports
rather than repairs monotonicity exceptions; and reproduces the 16 pilot experts from
the final 4-bit slice. It invokes the independent audit again. Captured activations,
gradients, and resumable chunks are generated artifacts and must not be committed.

The standalone audit can also be repeated explicitly:

```bash
PYTHONPATH=src python scripts/audit_quantization_cost_surrogate.py \
  --stage1-dir results/expert_quantization_pilot \
  --surrogate-dir results/quantization_cost_surrogate \
  --output results/quantization_cost_surrogate/independent_audit.json
```

None of these commands implements or tunes the future distributionally robust
mixed-precision optimizer.

## RACE Stage 0 expert-residency oracle headroom

The separately authorized residency study is isolated under
`stage3_residency/`. It does not modify or reinterpret the existing Stage 3
measured-damage or Stage 3D selection-headroom work, and it does not implement
RACE. Its question is whether an exact future-aware expert-residency oracle has
substantial transfer-cost headroom over Random, LRU, LFU, globally selected
LFU-decay, and calibration-only Static Hotset on real generated-token routing.

The simulator uses one independent cache per MoE layer and one atomic top-k set
request per generated token/layer. Existing result NPZs are teacher-forced
`[example, layer, expert]` aggregates and are not reused as temporal traces. The
new collector performs deterministic greedy decode and records the full top-k set,
gate weights, generated-token index, token ID, layer, prompt ID, domain, and exact
trace/config hashes.

Implementation, frozen pilot/full source configs, event semantics, tests, exact
commands, and the audited result are documented in
[`stage3_residency/README.md`](stage3_residency/README.md). The completed A40/BF16
full run contains 100 prompts/domain, 51,112 generated tokens, and 817,792 atomic
layer events. Its trace hash is
`ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`, and its
frozen evaluation hash is
`7ce228983b6547d61341757234e77ca7f59a4d0ba53b1e04b64243e9b2ea0971`.

The audited decision is **`RACE_STAGE0_STRONG_GO`**. The oracle beat the strongest
eligible simple policy by 17.56%--18.24% at capacity 12 and 45.48%--46.73% at
capacity 32, with STRONG-GO support at capacities 12/16/24/32 in all four workload
regimes. Capacity 8 has zero headroom because the mandatory atomic request itself
contains eight experts. The source/base commit recorded by the preregistration is
`48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`; the actual Stage 0 runtime commit is
`0f70c61131b877dd9c297663886d563d9e27f55b`.

The raw pilot trace was not retained in the repository archive, although its
frozen/evaluation/audit artifacts remain and the validated full raw trace is
complete. This is an archival limitation, not a failure of the validated full
run. Bootstrap intervals are conditional on the frozen workload ordering and
reweight per-sequence contributions; stateful cache trajectories are not
regenerated under reordered bootstrap workloads. Results concern simulated
expert residency/miss counts; no end-to-end latency improvement is claimed.

## RACE Stage 1 simple-prediction headroom

The follow-on prediction study is isolated under
`stage3_residency/stage1_prediction/` and reuses the exact frozen Stage 0 trace,
workload paths, cache semantics, baseline costs, and offline oracle. It implements
only Persistence, LastGate, Gate-EWMA, calibration-fitted first-order Markov,
direct Markov-H, and a simple Markov-plus-popularity hybrid behind one common
prediction-score/LRU/expert-ID retention rule. It contains no RACE optimizer and
does not prefetch.

The completed result is **`RACE_STAGE1_STRONG_GO`**. Calibration selected
`markov_plus_ewma_h2_beta0.5_alpha0.95` globally. Across all ten frozen workload
paths, this method closed 9.04%, 10.69%, 11.05%, and 9.26% of the Stage 0 oracle
gap at capacities 12, 16, 24, and 32, respectively. Residual oracle headroom was
16.17%, 23.82%, 33.68%, and 41.63% of Stage 0 baseline cost. Both preregistered
STRONG-GO conditions therefore pass at 4/4 non-degenerate capacities.

The exact report, raw JSONL, 95% paired conditional-bootstrap intervals,
lookahead diagnostics, prediction-quality metrics, required tables/figures, and
hash manifest are documented in
[`stage3_residency/stage1_prediction/README.md`](stage3_residency/stage1_prediction/README.md).
Run or verify the complete workflow with:

```bash
RACE_STAGE1_WORKERS=4 stage3_residency/stage1_prediction/scripts/run_full.sh
```

Finite-lookahead validation matched exact DP on 37,052 enumerated and 300 random
tiny traces. Perfect next-use scoring with the same simple eviction mechanism
matched the full Stage 0 oracle, while exact H=4 recovered 97.34% of oracle
advantage at capacity 12 but only 32.22% at capacity 32. This supports a
longer-horizon prediction target at larger spare budgets; it is not evidence that
RACE already works.

Bootstrap intervals reweight saved per-sequence contributions conditional on the
frozen workload ordering; stateful cache trajectories are not regenerated under
reordered bootstrap workloads. Results concern simulated expert residency/miss
counts; no end-to-end latency improvement or hardware speedup is claimed.

## RACE Stage 2 adaptive multi-horizon future-reuse ranking

Stage 2 is isolated under `stage3_residency/stage2_race/` and implements the first
actual RACE algorithm. It reuses the exact frozen Stage 0 trace, workload paths,
cache semantics, baseline costs and offline oracle, and it leaves the Stage 1
eviction rule completely unchanged. Only the retention score changes: nine causal
advisers (`MARKOV_H{1,2,4,8,16,32}`, `GATE_EWMA`, `LFU_DECAY`, `PERSISTENCE`) are
percentile-normalized over the current eviction-candidate set and combined as
`S_e = sum_j w_j z_{j,e}`, with the adviser weights adapted online by
multiplicative weights under a mandatory 32-event delayed-feedback protocol.

The completed result is **`RACE_STAGE2_NO_GO`**. Calibration selected learning rate
0.1, uniform initialization and the unweighted pairwise rank loss, making
`RACE_ONLINE` the frozen primary variant. Across the ten frozen workload paths it
cost 1.06%--1.98% *more* than the Stage 1 winner at capacities 12--32 and closed
only 3.17%--6.38% of the original Stage 0 oracle gap, against 9.04%--11.05% for
Stage 1.

The ablation chain explains the outcome: uniform rank aggregation is 4.23% worse
than Stage 1 on average, calibration-learned static weights recover 1.80%, online
adaptation recovers 0.82%, and a labeled ablation adding the frozen Stage 1 winner
itself as a tenth adviser recovers 1.77% more, reaching only 0.30% better than
Stage 1. The measured cause is representational: the Stage 1 winner is a raw-scale
blend of two pool members, and per-adviser percentile normalization discards the
magnitude that blend uses. The online learner itself is near-optimal for its own
objective, with empirical per-example adviser regret between -0.00004 and +0.00049
over 8,995,644 delayed updates.

Two structural equivalences are enforced by tests and by the pilot audit and prove
that the eviction mechanism was not changed: a Stage 2 variant with all weight on
one adviser reproduces the corresponding Stage 1 single-predictor cost exactly, and
the same mechanism driven by exact next-use scores reproduces the Stage 0 oracle
exactly.

Full details, all required tables and figures, the causality audit, the diagnostic
analysis and the archive hashes are in
[`stage3_residency/stage2_race/README.md`](stage3_residency/stage2_race/README.md)
and `stage3_residency/stage2_race/reports/race_stage2_report.md`. A separate theory
note proves a delayed-Hedge regret bound for the exact implemented update and states
explicitly that no regret theorem is claimed for the combined ranking loss or for
transfer cost. Run or verify the complete workflow with:

```bash
RACE_STAGE2_WORKERS=12 stage3_residency/stage2_race/scripts/run_full.sh
```

Stage 2 remains simulation-only. It makes no end-to-end latency or hardware-speedup
claim, and its negative result does not reduce the Stage 0 oracle headroom, which is
unchanged.

## RACE Stage 3 learned causal future-reuse ranking

Stage 3 is isolated under `stage3_residency/stage3_ranking/` and acts on Stage 2's own
diagnostic. It keeps the Stage 0 cache semantics and the Stage 1 eviction rule
completely unchanged and replaces only the retention score with one calibration-fitted
linear ranking function per cache capacity over 45 raw-scale causal features, fitted by
a convex weighted pairwise logistic ranking loss on within-candidate-set groups from the
frozen 80-sequence calibration path. It trains no neural network, uses no reinforcement
learning, and does not adapt online at evaluation time.

The completed result is **`RACE_STAGE3_PARTIAL_SUCCESS`**. Across the ten frozen
workload paths it closed 24.83%, 25.16%, 23.41% and 21.25% of the Stage 0 oracle gap at
capacities 12, 16, 24 and 32 — against 9.04%, 10.69%, 11.05% and 9.26% for the Stage 1
winner, so more than double — while improving on Stage 1 cost by 2.85% to 5.75% with
every paired 95% interval excluding zero and zero regressions in fifty
workload/capacity cells.

It does not reach the Stage 2 STRONG threshold, and the same study measured why.
Held-out pairwise ranking accuracy saturates near 69%: scaling training data 13-fold and
model capacity 8-fold moves it by under one point, so the ceiling is the information
carried by causal routing history rather than the estimator, and a neural network meets
the same wall. Restricted to exactly the information Stage 1 had, Stage 3 still improves
cost by 2.57% to 3.84%; decode-request-boundary awareness, which a serving stack always
knows but no earlier RACE stage used, adds the remainder.

Two mechanism proofs are enforced by tests and by the pilot audit: a Stage 3 replay
driven by the frozen Stage 1 winner score reproduces the frozen Stage 1 cost exactly,
and the same mechanism driven by exact next-use scores reproduces the Stage 0 oracle
exactly.

Full details are in
[`stage3_residency/stage3_ranking/README.md`](stage3_residency/stage3_ranking/README.md)
and `stage3_residency/stage3_ranking/reports/race_stage3_report.md`. Run or verify with:

```bash
RACE_STAGE3_WORKERS=12 stage3_residency/stage3_ranking/scripts/run_full.sh
```

Stage 3 remains simulation-only and makes no end-to-end latency or hardware-speedup
claim.

## Outputs

The analysis creates:

- expert_importance_by_domain.csv
- cross_domain_correlations.csv
- topk_overlap.csv
- routing_vs_functional_correlation.csv
- domain_specialized_experts.csv
- same_domain_split_half.csv
- results.json
- SUMMARY.md, including the go/no-go assessment
- PNG and PDF figures under figures/

Raw per-example arrays are stored as domains/DOMAIN.npz. Their shape is
[example, MoE layer, expert], with separate arrays for routing counts, gate sums,
functional-contribution sums, and optional gradient sums. Full hidden states and
token-level activations are never persisted.

The two audited 100-example snapshots are versioned at
`results/expert_domain_importance_with_answers/` and
`results/expert_domain_importance_prompts_only/`. Other generated result
directories remain ignored until explicitly promoted as validated artifacts.

Bootstrap intervals independently resample the already-collected examples within
each domain, aggregate expert vectors, recompute layer-wise Spearman correlations,
and then average each replicate across MoE layers. Top-k comparisons report both
intersection divided by k and Jaccard similarity at 10%, 25%, and 50%.

## Validation and failure policy

The collector refuses to continue when it observes an unsupported expert layout,
invalid expert IDs, non-finite or negative routing weights, a mismatch between
selected weights and router softmax behavior, incorrect assignment counts, missing
router/contribution hooks, non-positive layer totals, or leaked hooks.

Run the dependency-light local tests with:

    PYTHONPATH=src python -m unittest discover -s tests -v

The tests cover the current Hugging Face OLMoE router and tensorized experts, older
ModuleList-style experts, the production checkpoint tokenizer API, exact controlled
token geometry, padding exclusion, optional gradient attribution, expert masking,
hook cleanup, unchanged parameters, bootstrapping, split-half reliability, ranking,
top-k overlap, causal report generation, deterministic group-wise QDQ, tensorized
expert isolation and exact restoration, projected storage accounting, frozen pilot
selection, checkpoint resume, Stage-1 fallback decisions, and PNG/PDF figure
creation.

## Scientific limitations

- Weighted output magnitude can disagree with loss sensitivity and cannot establish
  causal importance.
- Domain corpora differ in sequence length and surface format. The primary run
  normalizes by valid tokens, but prompt-only and length-matched sensitivity runs
  remain useful.
- Dynamic padding is excluded from the statistics, although padded rows may still
  be computed internally by the model.
- The Transformers expert forward is evaluated a second time to recover individual
  contributions because its normal output is already summed across experts.
- Results from one OLMoE checkpoint should be replicated and followed by targeted
  expert masking/ablation before they guide mixed-precision allocation.
