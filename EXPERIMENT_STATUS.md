# Expert-domain importance experiment: persistent handoff

Last updated: 2026-08-12

This file is the durable cross-session research handoff. It records the current
experimental state and the conclusions supported by the committed, validated run
artifacts. The per-run `collection_config.json`, domain metadata, NPZ arrays, CSV
files, `results.json`, and `SUMMARY.md` remain the authoritative source for exact
values.

## Current research question and scope

The diagnostic question is:

> Do per-layer expert-importance rankings in OLMoE change substantially across
> general text, mathematics, coding, and reasoning inputs?

This stage measures routing utilization, selected gate mass, and a functional
contribution proxy. It does not establish causal importance. No model weights have
been modified, fine-tuned, compressed, or quantized.

Current decision: **GO for the frozen balanced causal-validation run, not yet for
quantization or compression**. Prompt-only validation and the controlled causal
run are complete and independently auditable. The balanced 12-specialist plus
12-control panel is frozen before masking; its A40 execution is pending.

The controlled run supplies strong domain-dependent functional evidence and three
successful Coding interventions. Because those interventions are all
Coding-specialized, the final causal gate is the frozen balanced panel across
General, Math, Coding, and Reasoning with same-layer routing controls.

## Code baseline

The implemented and committed workflow consists of:

- `a3df0a8` — Set up the expert analysis project
- `42f777c` — Implement adaptive OLMoE metric collection
- `6cdb41c` — Add cross-domain analysis and reporting
- `aa53fdc` — Add expert analysis validation tests
- `116d4c1` — Document the expert-domain experiment workflow
- `2703de2` — Add controlled fixed-token domain inputs
- `eb5f8c0` — Add split-half expert ranking reliability
- `38d46bc` — Support current OLMoE tokenizer API
- `1f04739` — Add reversible expert masking analysis
- `fccf288` — Add controlled causal validation runner
- `40c83e4` — Report controlled corpus eligibility
- `4065720` — Document controlled causal validation workflow

The balanced-panel implementation adds deterministic baseline-
only preregistration, strict integrity gates, intervention-level resume support,
paired control/aggregate bootstraps, figures, and reporting. No quantization or
weight modification has been implemented.

The code is split across collection, analysis, plotting, model discovery,
instrumentation, datasets, statistics, reporting, and tests. See `README.md` for
commands and metric definitions.

## Validated model and instrumentation setup

| Item | Validated value |
|---|---|
| Checkpoint | `allenai/OLMoE-1B-7B-0924` |
| Resolved model revision | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| Model class | `OlmoeForCausalLM` |
| MoE layers | 16, corresponding to model layers 0–15 |
| Experts per layer | 64 |
| Experts selected per token | 8 |
| Selected-weight behavior | Full-softmax top-k; `norm_topk_prob=false` |
| Contribution backend | Tensorized gate/up expert reconstruction |
| Contribution statistic | L2 norm of selected gate weight times expert output |
| Device | NVIDIA A40, CUDA |
| Weight dtype | BF16 |
| Seed | 42 |
| Maximum sequence length | 512 tokens |
| Batch size | 1 |
| Bootstrap replicates | 100 |
| Generation | None; teacher-forced/prompt forward inference only |

Package versions recorded by both runs:

- Python 3.12.3
- PyTorch 2.8.0+cu128
- Transformers 5.15.0
- Datasets 5.0.1
- huggingface-hub 1.27.0
- NumPy 2.1.2
- SciPy 1.18.0
- Matplotlib 3.11.1
- safetensors 0.8.0

The smoke test passed on four examples. Across all 16 layers, router probabilities
summed to approximately one, selected weights matched the model's full-softmax
top-k behavior within BF16 tolerance, contribution tensors had hidden dimension
2048, expert IDs were valid, routing/contribution counts were nonzero, and hook
counts were zero before and after collection.

## Datasets and paired samples

No dataset substitution occurred. Both runs used the same selected 100 examples in
each domain and the same resolved dataset revisions.

| Domain | Dataset and split | Resolved revision | Prompt-only tokens | With-answer tokens |
|---|---|---|---:|---:|
| General | WikiText-103 raw, test | `b08601e04326c79dfdd32d625aee71d232d685c3` | 9,653 | 9,653 |
| Math | GSM8K main, test | `740312add88f781978c0658806c59bc2815b9866` | 6,425 | 16,392 |
| Coding | MBPP full, test | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | 10,841 | 18,675 |
| Reasoning | ARC-Challenge, test | `210d026faf9955653af8916fad021475a3f00453` | 7,868 | 7,968 |

Prompt-only mean sequence lengths were 96.53, 64.25, 108.41, and 78.68 tokens for
General, Math, Coding, and Reasoning, respectively. No selected example reached the
512-token cap. General still contains short WikiText rows, including 32 of 100
examples below 16 tokens, and should be filtered or length-matched in the next
control.

## Completed runs and artifact locations

### Reference-answer run

Tracked artifact directory: `results/expert_domain_importance_with_answers/`

Exact collection command supplied for the RunPod run:

```bash
python scripts/collect_expert_importance.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --domains general math coding reasoning \
  --num-examples 100 \
  --max-length 512 \
  --batch-size 1 \
  --seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --output-dir results/expert_domain_importance
```

This run used the default `include_reference_answers=true` behavior. Its collection
fingerprint is
`64e951904e42a52caca82051cc6292e8854da1bfffaa413d99d725006bebd2be`.
The RunPod command originally wrote `results/expert_domain_importance`; the imported
snapshot was renamed to `expert_domain_importance_with_answers` to distinguish it
unambiguously from the prompt-only control.

### Prompt-only run

Tracked artifact directory: `results/expert_domain_importance_prompts_only/`

Reproducible equivalent command reconstructed from `collection_config.json`:

```bash
python scripts/collect_expert_importance.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --domains general math coding reasoning \
  --num-examples 100 \
  --max-length 512 \
  --batch-size 1 \
  --seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --prompts-only \
  --output-dir results/expert_domain_importance_prompts_only
```

Its collection fingerprint is
`087c60fee5950c81a34a276cbbce23e94643fb1cdc0190641acb331d16eea045`.

Analysis and plotting are reproducible with:

```bash
python scripts/analyze_expert_importance.py \
  --input-dir results/expert_domain_importance_prompts_only \
  --bootstrap-replicates 100

python scripts/plot_expert_importance.py \
  --input-dir results/expert_domain_importance_prompts_only
```

Both audited 100-example snapshots are versioned so their exact raw statistics,
reports, and figures remain available across Codex sessions and fresh clones. The
repository still ignores other ad hoc `results/` directories unless they are
explicitly promoted as validated snapshots in `.gitignore`.

## Independent artifact audit

The prompt-only result was independently checked after copying it into this
workspace:

- Each domain NPZ contains `routing_counts`, `gate_sums`, and
  `contribution_sums` with shape `[100, 16, 64]`, plus 100 token counts.
- Every valid token has exactly eight routing assignments in every layer.
- All stored routing, gate, and contribution values are finite and nonnegative.
- Layer-wise normalized expert vectors sum to one within floating-point tolerance.
- Every domain/layer has a complete 64-expert table; zero-use experts are retained
  with zero importance rather than omitted.
- Published Spearman correlations and top-25% Jaccards match independent
  recomputation from the NPZ arrays exactly.
- CSV row counts are complete: 4,096 main expert rows, 306 correlation rows, 918
  top-k rows, 68 routing-versus-functional rows, and 160 specialization rows.
- All nine PNG and nine PDF figures exist; all PNGs decode and representative
  figures were visually inspected.
- General text is unaffected by the answer flag, and every General NPZ array is
  bit-for-bit identical across the two runs. Dataset revisions and selected example
  IDs also match across all domains.

## Main cross-domain results

Values below are means over all 16 MoE layers and all six domain pairs.

| Metric | Prompt-only Spearman | With-answer Spearman | Prompt-only top-25% Jaccard | With-answer top-25% Jaccard |
|---|---:|---:|---:|---:|
| Routing frequency | 0.233 | 0.227 | 0.238 | 0.259 |
| Gate mass | 0.277 | 0.273 | 0.247 | 0.268 |
| Functional contribution | 0.323 | 0.315 | 0.277 | 0.299 |

The prompt-only functional result is only 0.009 higher in mean Spearman and 0.022
lower in mean top-25% Jaccard. Domain dependence therefore remains after reference
answers are removed.

Prompt-only functional-contribution comparisons:

| Domain pair | Spearman | 95% bootstrap CI | Top-25% Jaccard | With-answer Spearman |
|---|---:|---:|---:|---:|
| General–Coding | -0.046 | [-0.089, 0.010] | 0.134 | -0.079 |
| General–Reasoning | 0.263 | [0.186, 0.319] | 0.219 | 0.261 |
| Math–Coding | 0.294 | [0.263, 0.322] | 0.283 | 0.507 |
| Coding–Reasoning | 0.339 | [0.296, 0.378] | 0.316 | 0.314 |
| General–Math | 0.384 | [0.329, 0.438] | 0.284 | 0.261 |
| Math–Reasoning | 0.707 | [0.676, 0.729] | 0.428 | 0.623 |

In the prompt-only run, 75 of 96 functional layer/domain-pair correlations are
below 0.5; 89 are below 0.75. The five layers with the lowest mean functional
correlation are layers 5, 13, 12, 10, and 14. The same five layers were the most
domain-dependent in the with-answer run, although their order differed slightly.

## Reference-answer sensitivity

Same-domain functional-ranking stability between prompt-only and with-answer runs:

| Domain | Mean layer-wise Spearman | Mean top-25% Jaccard | Token reduction |
|---|---:|---:|---:|
| General | 1.000 | 1.000 | 0.0% |
| Math | 0.847 | 0.570 | 60.8% |
| Coding | 0.973 | 0.858 | 42.0% |
| Reasoning | 0.999 | 0.934 | 1.3% |

Math rankings are meaningfully sensitive to whether worked answers are included.
Most notably, Math–Coding functional correlation decreases from 0.507 with answers
to 0.294 with prompts only. Worked GSM8K solutions appear to make the Math
calibration distribution more coding-like. This is evidence that calibration
composition matters, but not evidence of causal expert specialization.

The two runs are not exact suffix-only controls for every domain. The GSM8K
formatter changes `Worked answer:` to `Answer:`, while MBPP changes `Reference
solution:` to `Solution:` and changes the trailing code context. ARC appends only
the answer key, and WikiText is unchanged. Future paired runs should preserve an
identical token prefix and append the reference answer only in the with-answer arm.

## Domain-specialized expert examples

The strongest examples should be chosen using absolute normalized range and rank
range, not only the max/min ratio, because ratios can explode when the minimum is
near zero.

Specialists that remain prominent across both runs include:

| Layer | Expert | Prompt-only maximum | Prompt-only minimum | Prompt-only ranks | Normalized range |
|---:|---:|---|---|---|---:|
| 11 | 27 | Coding | Reasoning | Coding 1, Reasoning 60 | 0.1761 |
| 10 | 56 | Coding | General | Coding 1, General 61 | 0.1623 |
| 1 | 25 | Coding | General | Coding 2, General 64 | 0.1418 |
| 5 | 42 | Coding | Reasoning | Coding 2, Reasoning 62 | 0.1394 |
| 13 | 25 | Coding | Math | Coding 2, Math 63 | 0.1121 |

These are candidates for controlled masking, not yet candidates for irreversible
precision reduction.

## Routing frequency versus functional contribution

Prompt-only mean within-domain Spearman correlations across layers are:

| Domain | Routing versus functional contribution |
|---|---:|
| General | 0.888 |
| Math | 0.897 |
| Coding | 0.943 |
| Reasoning | 0.911 |

Routing utilization is therefore a useful but imperfect proxy within a domain.
Across domains, functional rankings are more stable than routing rankings by about
0.090 mean Spearman (0.323 versus 0.233), but both show substantial distribution
dependence.

## Interpretation and limitations

The result meets the project's heuristic **strong support** criterion: many layers
have functional Spearman below 0.5 and mean top-25% Jaccard is only 0.277. The
answer-removal sensitivity run does not remove this signal, and the most affected
layers and several specialized experts persist.

Do not overstate this conclusion:

- Functional contribution is a weighted-output magnitude, not loss sensitivity or
  causal importance.
- Domain identity is entangled with dataset, vocabulary, prompt wrapper, task
  format, sequence length, and total token count.
- Prompt-only Math, Coding, and Reasoning explicitly announce their task type,
  whereas WikiText is unwrapped prose.
- MBPP prompts retain tests and an opening Python code fence; these are legitimate
  coding-domain tokens but also strong surface-format cues.
- Bootstrap intervals measure example-sampling uncertainty within these four
  corpora. They do not cover prompt-template, checkpoint, model, or dataset-choice
  uncertainty.
- This is one 100-example/domain run on one base checkpoint.

## Controlled causal validation: completed and audited

Artifact directory: `results/expert_domain_causal_validation/`

Reconstructed exact command:

```bash
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
```

The run used the fixed revision and all four pinned dataset revisions, without
substitution, on an NVIDIA A40 in BF16. Its collection fingerprint is
`052956d26bac03d54e637c5812b84ae8a37fd7b5f9b51f8558082af2e38cb362`.
Each domain contains 100 examples, exactly 64 measured source positions/example,
and exactly 6,400 measured positions. All controlled inputs use the same prefix IDs
`[8982, 27, 187]`, the same 68-token model sequence, and the same measurement-mask
hash.

Independent raw-artifact validation on 2026-08-12 confirmed:

- Every domain NPZ has `routing_counts`, `gate_sums`, and `contribution_sums` with
  shape `[100, 16, 64]` and exactly 64 token counts per example.
- Every example/layer has exactly `64 * 8 = 512` selected expert assignments.
- All statistics and loss arrays are finite and nonnegative.
- Requested and resolved model/dataset revisions and controlled-input hashes match.
- The masking smoke test changed loss and left zero hooks before and after.
- Every intervention's route-count vector exactly matches the independently stored
  routing tensor for that layer/expert/domain.

Main controlled functional results are mean cross-domain Spearman **0.317660**,
top-25% Jaccard **0.260797**, and **75/96** layer/domain-pair correlations below
0.5. General–Coding Spearman is **0.056321**, with 95% bootstrap interval
**[0.002165, 0.098871]**. Mean same-domain functional split-half correlations are
General **0.936777**, Math **0.978906**, Coding **0.981322**, and Reasoning
**0.977453**.

The three pre-registered Coding interventions all reproduced the expected causal
direction:

| Expert | Target delta NLL | Pre-registered contrast | Contrast | 95% CI |
|---|---:|---|---:|---:|
| L11/E27 | +0.034239 | Coding minus Reasoning | +0.035539 | [0.028813, 0.041402] |
| L10/E56 | +0.028811 | Coding minus General | +0.031183 | [0.025796, 0.036232] |
| L1/E25 | +0.022161 | Coding minus General | +0.030664 | [0.023681, 0.037605] |

These results establish causal loss sensitivity for the tested Coding specialists,
but not balanced generalization because all three targets are Coding.

## Balanced causal validation: preregistered, execution pending

Output directory: `results/expert_domain_balanced_causal_validation/`

The deterministic baseline-only selection was frozen on 2026-08-12 before any new
panel masking. Its selection-input fingerprint is
`6555c24be1d20799239f5e38209083989790706aca5a96716da29ec1e5bcefbd` and its
preregistration fingerprint is
`50a9eeb1f053385abe67cc94b2c4cc62570caf6d24f562cb8bcf05f5808cf714`.
No specialist threshold or control-matching tier required relaxation.

| Target | Specialists | Same-layer matched controls |
|---|---|---|
| General | L13/E52, L12/E40, L3/E24 | L13/E36, L12/E2, L3/E30 |
| Math | L8/E11, L12/E63, L2/E4 | L8/E54, L12/E43, L2/E34 |
| Coding | L13/E2, L11/E27, L10/E56 | L13/E61, L11/E7, L10/E62 |
| Reasoning | L13/E20, L11/E48, L14/E33 | L13/E63, L11/E47, L14/E8 |

All 12 specialists meet the strict target-rank, non-target-rank, positive-margin,
and routing-coverage criteria. All 12 controls are unique, same-layer matches with
target specialization no greater than 25% of their specialist's margin. Exact
candidate statistics, ranks, routing coverage, matching distances, hashes, and
rationale are in `selected_experts_preregistered.json`, `candidate_experts.csv`,
and `matched_controls.csv`.

Implementation validation completed locally with all **16 tests passing**, including
new deterministic-selection, unique-control, bootstrap, NPZ/CSV, and ten-figure
checks. This is not an experimental result. The actual A40 execution remains
pending. The ignored controlled source artifacts must be copied alongside a fresh
RunPod checkout before execution. Do not infer balanced causal findings or a
quantization decision until the run finishes and its raw artifacts are audited.

Exact frozen commands are documented in the `Balanced causal validation` section
of `README.md`. The required next action is to provision the A40 RunPod, copy the
controlled source artifacts, run the resumable command, audit all 24 interventions,
generate the full report, and then apply the pre-registered STRONG GO / GO WITH
QUALIFICATIONS / WEAK–NO GO rule.

## Fresh-session checklist

A new Codex session should:

1. Read `AGENTS.md`, `README.md`, and this file.
2. Run `git status --short` and preserve local result artifacts.
3. Inspect the relevant run's `collection_config.json`, metadata, `results.json`,
   and NPZ arrays before making new numerical claims.
4. Treat `results/expert_domain_causal_validation/` as the current validated
   controlled run; validate its raw NPZ/JSON/CSV artifacts for exact claims.
5. Treat `results/expert_domain_balanced_causal_validation/` as a frozen
   preregistration with execution pending until its masking checkpoints and final
   report are present and audited.
6. Keep all rankings layer-wise; expert IDs are not comparable across layers.
7. Keep “routing utilization,” “gate mass,” and “functional contribution proxy”
   terminology unless an intervention supports a stronger claim.
8. Describe selected-route masking as a causal loss-sensitivity intervention, not
   as expert deletion or a quantization simulation.
9. Do not begin quantization unless the balanced causal decision justifies it and
   the user explicitly advances the project stage.
10. Update this handoff after every new validated run.
