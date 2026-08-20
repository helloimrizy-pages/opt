# Expert-domain importance experiment: persistent handoff

Last updated: 2026-08-19

This file is the durable cross-session research handoff. It records the current
experimental state and the conclusions supported by the available validated run
artifacts. The per-run `collection_config.json`, domain metadata, NPZ arrays, CSV
files, `results.json`, and `SUMMARY.md` remain the authoritative source for exact
values.

## Current research question and scope

The diagnostic question is:

> Do per-layer expert-importance rankings in OLMoE change substantially across
> general text, mathematics, coding, and reasoning inputs?

The completed audited evidence measures routing utilization, selected gate mass, a
functional-contribution proxy, and selected-route masking loss sensitivity. No
modified checkpoint has been saved, fine-tuned, or compressed. The Stage-1,
Stage-2A, and Stage-2B interventions apply only temporary, exactly restored expert
QDQ.

Current decision: the mandatory Stage-1 reversible quantization-sensitivity pilot
is **GO** and complete after production execution and independent raw-artifact audit
on the NVIDIA A40. Four-bit QDQ passed all four preregistered gates; the 3-bit
fallback was not triggered. Stage 2A activation-aware surrogate validation is now
also complete and independently audited on the A40, with a final decision of
**SURROGATE_NO_GO**: both AOD and the preregistered GQS fallback failed Gates A and
B. The full cost matrix was therefore not authorized or generated, and any
optimizer that predicts per-expert quantization delta NLL remains blocked.
Prompt-only, controlled causal, balanced causal, Stage-1, and Stage-2A validation
are complete and independently audited.

Stage 2B (robust specialist preservation) development evaluation is also complete
and independently audited on the A40. It deliberately does not predict
quantization damage: it protects domain-specialized expert capacity directly under
a fixed incremental-memory budget. Neither the 4-to-8-bit nor 3-to-8-bit regime
passed every frozen development gate, so the final decision is
**ROBUST_PRESERVATION_NO_GO**. The seed-44 final split remains uninspected and must
not be evaluated under the Stage 2B preregistration. See the Stage 2B section below.

Stage 2C (fragility-weighted robust specialist preservation) development is now
also complete and independently audited on the A40. Neither regime passed all
five frozen seed-45 gates, so the final decision is **FRAGILITY_ROBUST_NO_GO**.
The seed-44 final split remains uninspected. See the Stage 2C section below.

Stage 3 (measured expert-damage preservation) is the next stage, explicitly
authorized by the user on 2026-08-16. Its status is **implemented and
preregistered in code / pending CUDA execution**. Instead of predicting
per-expert quantization damage (every predictive route failed in Stages 2A-2C),
it MEASURES the damage directly: one single-expert QDQ calibration evaluation
per layer/expert/bit-width, then a min-max allocation over the measured values,
with a preregistered additivity gate that must pass on calibration probes
before the new seed-46 development split may be evaluated. See the Stage 3
section below.

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
- `35849b7` — Add balanced causal validation panel
- `af4edb8` — Implement reversible OLMoE quantization sensitivity pilot
- `4b51cf7` — Complete and audit the Stage-1 quantization pilot
- `f04ea8d` — Add validated causal and quantization result artifacts
- `eb160bb` — Implement the Stage-2A quantization-cost surrogate
- `daeedb3` — Add the frozen Stage-2B specialist-preservation workflow

The balanced-panel implementation adds deterministic baseline-only
preregistration, strict integrity gates, intervention-level resume support, paired
control/aggregate bootstraps, figures, and reporting. The new Stage-1 pilot adds
temporary expert-only fake quantization/QDQ with exact restoration; it does not
save modified model weights or implement a mixed-precision allocator. The new
Stage-2A implementation adds exact routed-activation replay, fixed activation- and
gradient-aware cost formulas, expert-grouped validation, independent auditing, and
a conditional full cost-matrix builder. It still does not contain an allocator,
MILP, compression policy, fine-tuning, or packed inference kernel.

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

## Balanced causal validation: completed and independently audited

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

Exact command used:

```bash
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
```

The A40/BF16 run completed all 24 interventions across all four domains. Fresh
baselines reproduced the source arrays bitwise, routing tensors reproduced bitwise,
all 96 masked route-count and zeroed-gate-mass arrays exactly matched the stored
baseline tensors, masking changed loss, and all hook checks passed. The source
collection, selection-input, preregistration, and inference fingerprints are,
respectively, `052956d26bac03d54e637c5812b84ae8a37fd7b5f9b51f8558082af2e38cb362`,
`6555c24be1d20799239f5e38209083989790706aca5a96716da29ec1e5bcefbd`,
`50a9eeb1f053385abe67cc94b2c4cc62570caf6d24f562cb8bcf05f5808cf714`,
and `13f53ddfef0d2155626642ce68acdce3a2e4abd1a1d5a9d19fed52e4597c3bb3`.
The preregistration JSON and selection/control CSVs are byte-identical to commit
`35849b7`, establishing that the panel was frozen before the masking run.

Independent recomputation on 2026-08-13 reconstructed all domain means, 72 named
pairwise contrasts, 12 paired-control differences, five aggregate rows, and three
Spearman analyses from the 96 raw checkpoint NPZs. All point estimates and all
1,000-replicate fixed-seed confidence intervals matched the generated tables.

| Target domain | Mean specialist contrast | 95% CI | Mean control contrast | Specialist minus control | 95% CI |
|---|---:|---:|---:|---:|---:|
| General | +0.022097 | [0.015570, 0.029472] | -0.000239 | +0.022336 | [0.015488, 0.029732] |
| Math | +0.004781 | [0.002261, 0.007299] | -0.005931 | +0.010712 | [0.006498, 0.015448] |
| Coding | +0.033066 | [0.029103, 0.037092] | +0.004192 | +0.028874 | [0.024813, 0.033054] |
| Reasoning | +0.006243 | [0.003793, 0.008815] | +0.000137 | +0.006106 | [0.002624, 0.009624] |
| All domains | +0.016547 | [0.014339, 0.018615] | -0.000460 | +0.017007 | [0.014618, 0.019411] |

Eleven of 12 specialist primary contrasts were positive, ten had 95% intervals
strictly above zero, and 11 of 12 specialist-minus-control point differences were
positive. Functional specialization was more predictive of causal specificity
than routing frequency alone: Spearman **0.753** [0.666, 0.804] versus **0.417**
[0.312, 0.477]. Routing specialization Spearman was **0.711** [0.621, 0.760].

Failures were retained. Math L2/E4 had target delta NLL **-0.004302** and primary
contrast **-0.005188** [-0.008501, -0.002065]; its paired difference was positive
but inconclusive. Reasoning L13/E20 had an inconclusive primary contrast and its
control was nominally more target-specific. Reasoning L14/E33 had a positive
primary contrast but an inconclusive paired-control difference.

The pre-registered decision rule yields **STRONG GO**: all four domain aggregates
have positive specialist contrasts and positive specialist-minus-control effects,
the overall paired effect is positive, and 91.7% of paired point differences are
positive. The evidence supports designing the next distributionally robust
mixed-precision quantization experiment, while retaining the one-checkpoint,
fixed-corpus, selected-route-masking limitations. No quantization was performed.

## Stage-1 quantization sensitivity pilot: complete and independently audited

Final status: **GO at 4-bit**.

The pilot was implemented on 2026-08-13 as the mandatory mechanism-consistency gate
between balanced causal masking and any future mixed-precision allocator. It does
not implement robust allocation, optimization, Stage 2, pruning, fine-tuning, or a
low-bit runtime kernel. Production inference and the independent raw-artifact audit
completed on 2026-08-14 on the required NVIDIA A40.

The panel was frozen at
`results/expert_quantization_pilot/pilot_panel_preregistered.json` before any
quantization result was inspected. Its fingerprint is
`404927664048259fb623a7b3181e811c8f18c68d5e32825b943b056257220af7`.
Selection uses only the functional-specialization margin already frozen by the
balanced causal preregistration; masking effects and quantization behavior are
explicitly excluded.

| Target | Selected specialists, strongest first | Existing matched controls |
|---|---|---|
| General | L13/E52, L12/E40 | L13/E36, L12/E2 |
| Math | L8/E11, L12/E63 | L8/E54, L12/E43 |
| Coding | L13/E2, L11/E27 | L13/E61, L11/E7 |
| Reasoning | L13/E20, L11/E48 | L13/E63, L11/E47 |

The implementation applies deterministic symmetric group-wise weight-only QDQ to
one expert at a time. The primary precision is 4-bit, the preregistered fallback is
3-bit, and the group size is 128 along the input-feature dimension. OLMoE's fused
`gate_up_proj` and `down_proj` parameters are indexed structurally on their expert
axis. Quantization arithmetic is performed in FP32 around FP16-stored scales, and
dequantized tensors are restored to the model dtype. Each intervention fingerprints
the selected expert and every unrelated expert in the layer, verifies isolation,
evaluates all four frozen controlled domains, restores the original slices exactly,
and checks for hook leakage.

Projected expert storage accounts for packed quantized weight bits, the exact
number of groups, and one FP16 scale per group. BF16 uses 16 bits/weight without
scale overhead. These are projected format bytes and effective bits/weight, not
measured runtime-memory savings.

The runner checkpoints each expert/domain/bit-width pass independently, validates
the run, input, original-weight, and quantized-weight fingerprints on resume, and
recomputes only incomplete checkpoint pairs. It reproduces the four BF16 baselines,
audits the frozen raw per-example masking array, repeats a real-checkpoint expert
isolation/restoration smoke test, and then runs 4-bit. The 3-bit fallback is launched
only when the 4-bit result satisfies the preregistered too-small-to-measure rule; a
clearly negative result does not trigger fallback.

The analysis saves per-example losses and reports target and non-target delta NLL,
all target-minus-domain contrasts, normalized relative delta NLL, routing coverage,
weight distortion, specialist-control differences, masking-versus-quantization
Spearman/Kendall/sign agreement, frozen importance correlations, fixed functional
and routing risk-proxy correlations, 1,000-replicate confidence intervals, and all
four Stage-1 gates. The final decision file can contain `GO` or `NO_GO`; the internal
`PENDING_FALLBACK` state is never emitted as the final production decision.

Exact production command used:

```bash
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
```

The controlled directory was present with all frozen input hashes intact. The
balanced directory initially contained only its three tracked selection files, so
the exact documented balanced A40 command was rerun against the frozen panel to
restore its missing BF16 baselines and raw per-example masking package. The restored
run reproduced every recorded aggregate and correlation exactly, passed all
baseline/routing/hook gates, and did not reselect or replace any expert.

Completed final outputs are
`quantization_pilot_results.csv`, `quantization_pilot_pairwise.csv`,
`specialist_vs_control.csv`, `quantization_vs_masking.csv`,
`quantization_distortion.csv`, `per_example_quantization_losses.npz`,
`results.json`, `stage1_decision.json`, `SUMMARY.md`, and five PNG/PDF figure pairs,
plus resumable checkpoints under `quantization/`.

Artifact directory: `results/expert_quantization_pilot/`.

The production inference fingerprint is
`082aad66cabb047abd889beba99a2ab6201b3f98e26cd4254883bee25a217884`.
The controlled-input fingerprint remained
`6555c24be1d20799239f5e38209083989790706aca5a96716da29ec1e5bcefbd`;
the balanced raw masking NPZ hash is
`4d9a3103cadc14eb03891dee50d666eeb172ddcfdf55616fbe024d9158593786`;
and the final per-example quantization NPZ hash is
`d0288414932272b5c496b106abf01a72b6ab28bf95493070a45ce2c146e18754`.

All smoke and production integrity gates passed:

- the complete suite passed **30/30 tests** before production and again after the
  reporting-only audit update;
- the pinned checkpoint resolved to
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, with 16 MoE layers, 64
  experts/layer, top-8 routing, expert axis 0, and BF16 expert matrices;
- real-checkpoint L13/E52 QDQ changed only the selected expert, produced finite
  loss and relative weight distortion **0.013894860693**, leaked no hooks, and
  restored all fingerprints exactly;
- every intervention verified all 63 unrelated experts in its layer unchanged,
  restored the selected expert exactly, and wrote all four domain checkpoints;
- all four fresh BF16 baselines were bitwise equal to the restored balanced
  baselines, so observed baseline reproduction noise was exactly **0.0 NLL**;
- the completed array has shape **[1, 16, 4, 100]**, all values are finite, every
  example has exactly 64 evaluated positions, and no 3-bit directory exists.

PyTorch emitted its CUDA warning that `CUBLAS_WORKSPACE_CONFIG` was not set while
deterministic algorithms were requested. This was retained as a caveat, not hidden:
the frozen source/balanced baselines and routing tensors nevertheless reproduced
bitwise, and the raw checkpoint/resume audit found no inconsistency.

Primary 4-bit aggregate results:

| Target domain | Mean specialist contrast | 95% CI | Mean control contrast | Specialist minus control | 95% CI |
|---|---:|---:|---:|---:|---:|
| General | +0.000603428 | [-0.000053796, 0.001217323] | -0.000073306 | +0.000676733 | [-0.000061414, 0.001444780] |
| Math | -0.000385300 | [-0.001003041, 0.000202139] | -0.000409598 | +0.000024298 | [-0.000675318, 0.000667981] |
| Coding | +0.001405140 | [0.000574155, 0.002375074] | +0.000773413 | +0.000631727 | [-0.000201355, 0.001546852] |
| Reasoning | +0.000391567 | [-0.000205189, 0.000986266] | +0.000376246 | +0.000015322 | [-0.000645631, 0.000671734] |
| All domains | +0.000503709 | [0.000208258, 0.000808887] | +0.000166689 | +0.000337020 | [-0.000049781, 0.000728557] |

Mechanism-consistency and predictor results:

| Comparison | Spearman | 95% CI | Kendall tau | 95% CI | Sign agreement |
|---|---:|---:|---:|---:|---:|
| Masking vs quantization specificity | 0.423529 | [0.049926, 0.617647] | 0.316667 | [0.016667, 0.483333] | 0.6875 [0.5000, 0.8125] |
| Functional specialization vs quantization specificity | 0.432353 | [0.041176, 0.608824] | — | — | — |
| Routing specialization vs quantization specificity | 0.338235 | [-0.035294, 0.556029] | — | — | — |
| Target routing frequency vs quantization specificity | 0.270588 | [-0.135294, 0.520588] | — | — | — |
| Fixed functional risk vs domain-level delta NLL | 0.092308 | [-0.092310, 0.250781] | — | — | — |
| Fixed routing risk vs domain-level delta NLL | 0.041896 | [-0.134641, 0.217633] | — | — | — |

Failures and limitations were retained. Six of eight specialist contrasts were
positive, but only two had 95% intervals above zero. Both Math specialists were
negative: L8/E11 contrast **-0.000508272** and L12/E63
**-0.000262328**. Six of eight specialist-control differences were positive, but
only General L13/E52 excluded zero. Math L12/E63 and Reasoning L11/E48 were less
target-specific than their controls. Gate C passed its preregistered positive point
criterion, but the overall paired interval includes zero. The fixed risk proxies
were weak and their intervals include zero.

Independent audit command:

```bash
python scripts/audit_quantization_pilot.py \
  --source-dir results/expert_domain_causal_validation \
  --balanced-results-dir results/expert_domain_balanced_causal_validation \
  --pilot-results-dir results/expert_quantization_pilot \
  --output results/expert_quantization_pilot/independent_audit.json
```

The standalone auditor imported none of the production analysis functions. It
reconstructed all 16 expert contrasts, eight paired differences, five aggregates,
all 1,000-replicate intervals, masking comparisons, and risk correlations from raw
checkpoints. It also verified all CSV/JSON values, NPZ dimensions, finite values,
input/panel/model/config hashes, restoration metadata, and artifact-manifest
hashes. All **6,223** checks passed; the largest numeric discrepancy was
**1.39e-17**, and the independent decision was `GO`.

The preregistered gates were A=PASS, B=PASS (General, Coding, and Reasoning
positive), C=PASS, and D=PASS (median absolute domain-level delta NLL
**0.000215903** versus threshold **1e-7**). Because 4-bit passed Gate D, the
preregistered 3-bit fallback was not triggered. Stage 1 is therefore **complete /
GO**. A separately designed distributionally robust mixed-precision experiment is
scientifically justified but remains **pending**; Stage 2 was not implemented.

## Stage 2A quantization-cost surrogate: complete and independently audited

Final status: **SURROGATE_NO_GO**. The full cost matrix was not authorized or
generated, and robust mixed-precision optimization remains blocked and
unimplemented.

Artifact directory: `results/quantization_cost_surrogate/`.

The Stage-2A implementation was added on 2026-08-14 to test a cost function before
allowing it to drive a future robust optimizer. The primary fixed score is

```text
AOD(l,e,d,b) = sum_t ||g(l,t,e) * (f_e(h; W_hat_b) - f_e(h; W))||_2^2
               / (sum_t ||y_moe(l,t)||_2^2 + 1e-30).
```

The numerator contains only frozen measured tokens actually routed to the expert
and therefore incorporates route frequency, actual selected gate strength, domain,
real expert inputs, and the bit-specific Stage-1 QDQ perturbation directly. UOD,
REOD, and APD are fixed secondary diagnostics. The failed Stage-1 functional/routing
weight-risk formulas are preserved without redefinition as negative controls.

Implementation files:

- `src/expert_analysis/expert_replay.py` — exact native-BF16 routed-activation
  capture, selected IDs/gates, LayerEnergy, example/token alignment, resumable
  domain/layer artifacts, and sampled actual-container-versus-structural replay
  validation against the untouched full-forward MoE output;
- `src/expert_analysis/activation_quantization_cost.py` — AOD, REOD, APD, UOD,
  selected-route filtering, and exact Stage-1 pilot QDQ fingerprint checks;
- `src/expert_analysis/gradient_quantization_cost.py` — the pre-registered GQS
  fallback with one frozen-model backward per example and GQS2 diagnostic;
- `src/expert_analysis/surrogate_validation.py` — all-64-observation loading,
  1,000-replicate expert-grouped bootstrap, specificity/domain/ranking analyses,
  strict gates, tables, reports, and figures;
- `src/expert_analysis/cost_matrix.py` — conditional `[16,64,4,4]` matrix,
  domain/layer/precision checkpoints, route coverage, exact projected storage,
  16-bit zero reference, monotonicity diagnostics, and pilot-slice reproduction;
- `scripts/validate_quantization_cost_surrogate.py`,
  `scripts/build_quantization_cost_matrix.py`, and the standalone
  `scripts/audit_quantization_cost_surrogate.py`;
- four new synthetic test modules plus current-Transformers OLMoE replay coverage.

Before implementation, the frozen Stage-1 decision was revalidated as `GO`; all 69
Stage-1 NPZ files were inspected for shape, dtype, finiteness, and token geometry;
all 16 expert metadata files still reported exact restoration; and the authoritative
per-example loss NPZ reproduced its recorded SHA-256
`d0288414932272b5c496b106abf01a72b6ab28bf95493070a45ce2c146e18754`.
The four controlled inputs remain `[100,68]` with exactly 6,400 measured positions
per domain and match their frozen hashes. The dependency-light Stage-2A preflight
passed and selected replay-validation layers 6, 7, 9, and 11 deterministically from
seed 42. Its capture fingerprint is
`672316b5b7ea8be1d7bf328aca2c9bd7f367238eccdbde549d791fa10c9f8fe2`.

The implementation was first validated locally on Apple Silicon/MPS with PyTorch
2.12.1; the full repository suite passed **57/57 tests**, and a synthetic end-to-end
package passed 1,060 standalone audit checks. Those checks validated code paths
only. The real frozen 64-observation validation subsequently ran on the NVIDIA A40
with CUDA BF16, batch size 1, 4-bit Stage-1 QDQ, group size 128, seed 42, and 1,000
expert-grouped bootstrap replicates.

Exact A40 validation command used:

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

All integrity gates passed. The capture fingerprint remained
`672316b5b7ea8be1d7bf328aca2c9bd7f367238eccdbde549d791fa10c9f8fe2`
and the run fingerprint was
`8c17ff9db9a4f4fa36e3c343a3b9f78356e7a86deb095b949894873ebb895758`.
Replay validation passed 48 samples across layers 6, 7, 9, and 11, covering 42
experts; maximum isolated-contribution and aggregate-MoE-output absolute errors
were **0.000244140625** and **0.001788616180**, respectively. All 16 pilot experts
exactly reproduced their Stage-1 original and quantized fingerprints, and every
intervention retained exact restoration and unrelated-expert isolation metadata.

Primary AOD results:

| Metric | Result | Preregistered requirement | Outcome |
|---|---:|---:|---:|
| Overall Spearman | +0.122482 [-0.216702, +0.433970] | > +0.25 and CI low > 0 | FAIL |
| Improvement over WeightRiskFunctional | +0.030174 [-0.008350, +0.070272] | >= +0.15 | FAIL |
| Specificity Spearman | +0.405882 [-0.200000, +0.816831] | > +0.30 | PASS |
| Top-domain accuracy | 0.5000 [0.2500, 0.7500] | > 0.40 | PASS |
| Positive domain correlations | 3/4 | >= 3/4 | PASS |

AOD therefore failed Gates A and B, activating the GQS fallback exactly as
preregistered. GQS also failed Gates A and B: overall Spearman was **+0.097527**
[-0.223317, +0.400222], improvement over WeightRiskFunctional was only
**+0.005220** [-0.069827, +0.099279], specificity Spearman was **+0.467647**, and
top-domain accuracy was **0.5625**. Both methods were negatively correlated with
actual 4-bit delta NLL in Math (AOD **-0.579412**, GQS **-0.620588**), a retained
domain-level counterexample.

The standalone auditor independently recomputed the final
`SURROGATE_NO_GO` decision from raw saved values. All **1,528** checks passed with
no errors and maximum published-versus-recomputed numeric difference **0.0**.
Because neither fixed primary surrogate passed every gate, the preregistered
stopping rule blocks the full `[16,64,4,4]` cost matrix and any robust
mixed-precision optimizer. The result remains limited to one model revision, 16
pilot experts, four reused controlled domains, and 100 examples/domain; grouped
bootstrap intervals do not cover checkpoint, prompt, or dataset-choice uncertainty.

## Stage 2B robust specialist preservation: development complete and audited / NO-GO

Artifact directory: `results/robust_specialist_preservation/`.

Final status: **ROBUST_PRESERVATION_NO_GO**. No regime passed every frozen
development gate, so the final seed-44 split was not evaluated and no final-stage
claim is authorized.

Stage 2B tests one fixed idea: when exact per-expert quantization damage is not
predictably estimable (the frozen Stage-2A result), protect domain-specialized
expert capacity directly and maximize the specialist coverage of the
worst-protected domain. The optimization objective uses only the frozen
domain-conditioned functional-contribution arrays; no AOD, GQS, APD,
reconstruction error, fitted surrogate, or any delta-NLL prediction enters any
objective, and the Stage-2A `SURROGATE_NO_GO` decision is preserved unchanged.

Implementation (2026-08-15, local Apple-Silicon machine, no model inference):

- Calibration: 25 examples/domain (seed `20260815`) drawn deterministically from
  the frozen controlled 100/domain arrays; single-domain baselines use their
  full 100 examples so every method receives exactly 100 calibration examples.
  Calibration fingerprint
  `7fc28a40a24f9c7684d9544f1d1524fd9329b6abfa717ce70248cf440c8632d4`.
- Scores: layer-normalized, layer-equalized functional importance `F[l,e,d]`;
  specialization margin `S_raw = F[l,e,d] - max_{d'!=d} F[l,e,d']`, positive
  part normalized per domain; an identical routing-based score for the
  preregistered Robust-Routing ablation.
- Memory accounting reuses the exact Stage-1 `projected_expert_storage`:
  per expert `M(3)=2,457,600`, `M(4)=3,244,032`, `M(8)=6,389,760` bytes
  (packed payload plus FP16 group scales, group size 128). Regimes `4to8` and
  `3to8` with 8-bit protection; budgets 5/10/20/30% of the exact total
  increment; every method at a regime/budget shares the identical byte budget.
- Allocations: `scipy.optimize.milp` (HiGHS, SciPy 1.18.0, `mip_rel_gap=0`)
  solved Robust-Functional (max-min coverage), Robust-Routing,
  Average-Specialization, Global-Importance, and four single-domain baselines;
  five deterministic score-independent random allocations (seeds 1001-1005);
  uniform BF16/8/4/3-bit reference records. 108 allocation files frozen under
  `allocations/` with per-file SHA-256 and registry hash
  `b0221262f0e51700cc16fa5e6a681f63ab6507a9d768714f853f3dfc3f87aa34`.
- Score-space sanity (20% budget, 4to8): Robust-Functional equalizes coverage
  (min 0.5826); Average-Specialization mean 0.6013 but min 0.4316;
  Global-Importance min 0.0957 (General); single-domain methods cover only
  their own domain; randoms sit near 0.15-0.27. No allocation exceeds the
  max-min optimum, confirming MILP optimality.
- Held-out splits: development seed 43 (50/domain) and final seed 44
  (100/domain), identical 68-token neutral-prefix geometry with 64 measured
  positions. The seed-42 prior selection was reconstructed and verified against
  the frozen input hashes bit for bit before exclusion. General excluded the
  entire previously inspected candidate pool; Math/Coding/Reasoning datasets
  are too small for full-pool exclusion, so they exclude the 100 previously
  evaluated examples (limitation recorded in `splits/split_manifest.json`).
  All development/final/prior overlaps verified empty at content-token level.
- Pre-inference independent audit:
  `scripts/audit_specialist_preservation.py` (imports no production analysis
  code) passed 1,676/1,676 calibration, allocation, MILP, and split checks with
  maximum numeric difference 0.0. Full test suite: 118/118 passing.

Preregistered evaluation rule (frozen before any held-out NLL):

- Development: 20% budget only, both regimes, all methods plus randoms, on the
  seed-43 split; gates A (beats random mean), B (beats Global-Importance and
  Average-Specialization on worst-domain relative delta NLL), C (mean
  degradation within 10% of Average-Specialization, absolute floor 1e-4), and
  D (positive recovery vs the all-base model in >= 3 of 4 domains). At least
  one passing regime writes `FULL_EVALUATION_GO`; otherwise
  `ROBUST_PRESERVATION_NO_GO` stops the experiment with the negative result
  preserved and the final split uninspected.
- Final (only if GO): all budgets/regimes/methods from the frozen registry,
  1,000-replicate paired bootstrap, single-domain transfer matrix, and the
  frozen STRONG SUCCESS / SUCCESS WITH QUALIFICATIONS / NEGATIVE RESULT rule.
- The 3-bit regime is newly preregistered here as an intentionally aggressive
  compression setting; the untriggered Stage-1 3-bit fallback (a
  mechanism-gate rule) does not prohibit it.
- Strict CUDA determinism is required: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  before launch, TF32 disabled, eager attention, deterministic algorithms, and
  a mandatory bitwise repeated-BF16-baseline gate that stops the run on any
  mismatch.

Exact A40 development command used:

```bash
PYTHONPATH=src python scripts/run_stage2b_specialist_preservation.py --stage development --device cuda --cache-dir .hf_cache
```

The run used the pinned OLMoE revision on NVIDIA A40, CUDA BF16, batch size 1,
seed 42, strict deterministic algorithms, eager attention,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and TF32 disabled. The run fingerprint is
`dd954304806fd7f5eb7ebbd124b11f432cf85a1bbd283cd9e17262436e82456d`.
It evaluated the frozen 20% budget allocations on the seed-43 development split
with 50 examples/domain and 64 measured positions/example.

Development gate outcomes:

| Regime | Robust worst-domain relative delta NLL | Random mean | Global-Importance | Average-Specialization | Gate A | Gate B | Gate C | Gate D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4to8 | +0.031427 | +0.026288 | +0.011853 | +0.028731 | FAIL | FAIL | PASS | PASS |
| 3to8 | +0.053559 | +0.064823 | +0.056027 | +0.050999 | PASS | FAIL | PASS | PASS |

At 4to8, Robust-Functional was worse than the random mean and both non-robust
preservation baselines on worst-domain relative delta NLL. At 3to8 it beat the
random mean and Global-Importance, but not Average-Specialization, so Gate B still
failed. Both regimes stayed within the frozen mean-degradation tolerance and
recovered positively from the all-base model in all four domains. The
Robust-Routing ablation was retained as an ablation and was not allowed to replace
the preregistered primary method.

The final independent audit recomputed the development tables, bootstrap results,
all four gates, and the stopping decision without importing production analysis
functions. It passed **2,340/2,340** checks with maximum numeric difference **0.0**;
development results are audited and final results are explicitly absent. The
negative result therefore stops Stage 2B as preregistered. The final seed-44 split
must remain unevaluated, and the allocation objective, budgets, gates, and failed
interventions must remain unchanged. QDQ simulation permits no runtime speedup,
latency, or measured-memory claims; only exact projected storage is reported.

## Stage 2C fragility-weighted robust specialist preservation: development complete and audited / NO-GO

Artifact directory: `results/fragility_robust_preservation/`.

Final status: **FRAGILITY_ROBUST_NO_GO**. The A40 development run on the new
seed-45 split completed on 2026-08-16 and was independently audited (1,497/1,497
checks, zero numeric discrepancy). Neither precision regime passed all five
frozen gates, so the seed-44 final split was not evaluated and remains
untouched. Nothing in this stage modified the frozen Stage 2A or Stage 2B
negative results.

Calibration fragility (frozen before any allocation was solved) showed that
domain vulnerability is regime-dependent: at 4-bit, Coding is by far the most
fragile domain (normalized 1.93 vs 0.54-0.80 elsewhere), while at 3-bit General
is the most fragile (1.32) and Coding the least (0.79).

Development gate outcomes at the 20% budget (worst-domain relative delta NLL):

| Regime | Fragility-Robust | Robust-Functional | Random mean | Global-Importance | Average-Specialization | Gates |
|---|---:|---:|---:|---:|---:|---|
| 4to8 | +0.025616 | +0.033205 | +0.030074 | +0.012976 | +0.031682 | A PASS, B PASS, C FAIL, D PASS, E FAIL |
| 3to8 | +0.052080 | +0.050717 | +0.068232 | +0.054973 | +0.054626 | A FAIL, B PASS, C PASS, D PASS, E FAIL |

Fragility-Robust fixed the Stage 2B failure at 4to8 (gate A) but lost badly to
Global-Importance and Coding-Only there; Coding-Only reached worst-domain
+0.008512 with the same budget, showing large headroom that no coverage-based
objective found. At 3to8 it beat both simple baselines but lost gate A to
Robust-Functional by 2.7% and gate E by 1.5 points. The mechanism reading is
that specialist coverage does not track realized quantization loss, echoing the
Stage 2A result one abstraction level higher.

Stage 2C tests one fixed correction to Stage 2B: equalizing specialist coverage
is suboptimal when domains differ in baseline quantization vulnerability, so the
optimizer should balance predicted residual domain vulnerability instead. The
PRIMARY method (Fragility-Robust) minimizes the largest
`ResidualRisk_d = q_norm[d] * (1 - Coverage_d(x))` across the four domains under
the exact Stage 2B incremental-memory budgets, where `q_norm` is
calibration-measured domain fragility:
`q_raw[d,b] = (NLL_base_cal[d,b] - NLL_BF16_cal[d]) / NLL_BF16_cal[d]`, clipped
at zero (never absolute value) and normalized to mean one across domains. A
regime whose four clipped fragilities are all zero is invalid and is not
evaluated. The method uses no expert-level delta-NLL, no AOD/GQS/APD, no fitted
coefficients, no tuned exponents or lambdas, and no Stage 2B development
outcomes. If Stage 2C fails its gates, the negative result is preserved and
optimization development on this branch stops.

Temporal separation (preregistered): Stage 2C was conceived after the Stage 2B
development results were observed, so the seed-43 split is contaminated and is
never used. A completely new seed-45 development split (50 examples/domain, 64
measured positions, identical 68-token neutral-prefix geometry, pinned dataset
revisions) was built and frozen locally on 2026-08-15. Exclusions per domain:
the reconstructed seed-42 prior usage (bit-for-bit verified; entire inspected
candidate pool for General, the 100 previously evaluated examples for
Math/Coding/Reasoning, limitation recorded), all 50 seed-43 texts, and all 100
seed-44 texts. All overlap checks are empty at the content-token level. Split
input hashes: general `45b5d6e4f139dac0`..., math `c564af3b87fcf034`...,
coding `81eef2be91d5aa9f`..., reasoning `4425a57aec5614b2`... (full values in
`splits/split_manifest.json`). The untouched Stage 2B seed-44 final split is
reused for final confirmation only after a GO; its hashes were re-verified
without any model evaluation.

Implementation (2026-08-15, local Apple-Silicon machine, no model inference):

- `src/expert_analysis/stage2c_preflight.py` — verifies STRONG GO / GO /
  SURROGATE_NO_GO / ROBUST_PRESERVATION_NO_GO with the frozen Stage 2B gate
  values and registry hash, and verifies seed-44 isolation by hash only.
- `src/expert_analysis/fragility.py` — the fixed fragility formula, clipping,
  mean-one normalization, frozen-record integrity hashing, hash-verified reuse
  of the frozen Stage 2B specialization artifacts, and 25/domain calibration
  subset slicing verified against the frozen Stage 2B row hashes.
- `src/expert_analysis/fragility_optimization.py` — the Fragility-Robust MILP
  (`scipy.optimize.milp`/HiGHS, minimize z with
  `q_norm[d]*(1-Coverage_d(x)) <= z` and the exact byte budget), allocation
  records, a registry that freezes all eight regime/budget allocations before
  any development NLL and reuses every Stage 2B comparator by hash, plus an
  optimality sanity check against all reused comparators.
- `src/expert_analysis/fragility_evaluation.py` — seed-45 split construction
  and loading, preregistration freezing/immutability checks, run fingerprints,
  and phase record selection. Model evaluation reuses the audited Stage 2B
  mixed-precision manager, bitwise restoration, and loss checkpoints unchanged.
- `src/expert_analysis/fragility_statistics.py` — the five preregistered
  development gates (A: beats frozen Robust-Functional; B: beats the random
  mean; C: beats both Global-Importance and Average-Specialization; D: positive
  recovery in >= 3 of 4 domains; E: mean degradation at most 10% relatively
  worse than the better of Global-Importance/Average-Specialization with an
  epsilon only for denominator safety), the FINAL_CONFIRMATION_GO /
  FRAGILITY_ROBUST_NO_GO rule, the frozen final requirements 1-5 with the
  qualified-success rule, and the descriptive protection-shift and
  fragility-transfer analyses.
- `src/expert_analysis/fragility_reporting.py` — analysis driver, tables,
  decision files, pre-evaluation diagnostics (optimization-space only), the six
  preregistered figures, and summaries.
- Scripts: `build_stage2c_development_split.py` (done locally),
  `build_calibration_fragility.py`, `solve_fragility_robust_allocations.py`,
  `run_fragility_robust_development.py`, `run_fragility_robust_final.py`, and
  the standalone `audit_fragility_robust_preservation.py` (imports no
  production analysis code).
- Tests: 56 new tests; the full suite passes **174/174** locally.
- Independent audit of the current local state: **61/61** checks passed with
  maximum numeric difference 0.0 (frozen prior state, reused score hashes,
  seed-45 split disjointness, seed-44 isolation). Fragility, allocations, and
  development results are correctly reported as not yet present.

The preregistered order was followed exactly on the A40:
`build_calibration_fragility.py`, `solve_fragility_robust_allocations.py`, the
pre-evaluation audit, `run_fragility_robust_development.py`, and the
development audit. On FRAGILITY_ROBUST_NO_GO the run stopped as preregistered:
the negative result is preserved, no alternative fragility weighting may be
searched, and seed 44 stays untouched under this preregistration. The complete
tables are in `results/fragility_robust_preservation/SUMMARY.md` and
`development_seed45/development_results.json`.

## Stage 3 measured expert-damage preservation: on hold pending Stage 3D Sweep A

Artifact directory: `results/measured_damage_preservation/` (created by the
runs below).

Current status: **implemented, locally validated, and on hold**. No damage
matrix, allocation, probe, or held-out NLL has been produced yet. Nothing in
this stage modifies the frozen Stage 2A, 2B, or 2C negative results.

Two things changed on 2026-08-16, after this stage was implemented and before
any of it ran. Both are recorded in `prereg/stage3d.md`.

**It is gated on Stage 3D Sweep A.** This stage assumes per-expert damage is
heterogeneous enough to be worth measuring 1024 times. Stage 3D Sweep A tests
that assumption directly, so it is a gate on this stage rather than a stage
beside it. The go condition is Sweep A returning `HEADROOM` under the Step 5
rule in `prereg/stage3d.md`. If Sweep A returns `FLAT`, the negative result is
the paper and this stage does not run. This stage is **not** superseded; that
word would imply the answer is already known.

**Its step 1 cannot succeed as written.** Running
`scripts/build_stage3_development_split.py` aborts: the coding domain cannot
supply 50 disjoint seed-46 examples. `google-research-datasets/mbpp` full test
has 500 rows, 332 of them long enough for the 68-token geometry, and 300 are
already spent across the frozen controlled set and seeds 43, 44 and 45, leaving
32. Free eligible examples per domain are general 1301, math 104, coding 32,
reasoning 198, so a balanced fresh split now caps at 32 per domain. If this
stage is ever authorized to run, it needs a smaller development split, a larger
coding corpus, or a different evaluation set; seed 46 itself remains unused and
available, because Stage 3D builds no data split.

Authorization and scope: the user explicitly authorized this stage on
2026-08-16. Stages 2A-2C established that per-expert quantization damage
cannot be usefully PREDICTED from local weight distortion, activations,
gradients, specialist coverage, or fragility-weighted coverage. Stage 3
therefore stops predicting and MEASURES the damage directly:

- `m[l,e,d,b] = NLL_cal(only expert (l,e) at b-bit QDQ) - NLL_cal(BF16)` for
  all 16 layers x 64 experts x 4 domains x bit widths {3, 4, 8}, on the same
  frozen 25-example/domain Stage 2B calibration subset, with the audited
  Stage-1 QDQ and bitwise repeated-evaluation reproduction.
- The frozen Stage 2A SURROGATE_NO_GO decision blocks predictive surrogates
  only; a measured ground-truth delta NLL is the quantity those surrogates
  tried and failed to predict, and Stage 1 already measured it for 16 experts.
  The Stage 2C rule against searching alternative fragility weightings is
  respected: no score-based weighting is used at all.
- The optimizer minimizes the largest additively predicted domain delta NLL,
  `PredictedDelta_d(x) = sum_{l,e} m[l,e,d,bits(l,e)]`, under the exact frozen
  Stage 2B byte budgets (`scipy.optimize.milp`/HiGHS, no clipping, no
  weighting, no fitted term). Comparators are every frozen Stage 2B method
  plus the frozen Stage 2C Fragility-Robust allocation, reused by hash.
- Whether measured damage composes additively is itself a preregistered gate:
  all thirty frozen 20%-budget allocations are evaluated on calibration data
  as probes, and a regime is authorized for development evaluation only if the
  additive model ranks them with Spearman >= 0.8 per domain and >= 0.8 on the
  worst-domain delta. If both regimes fail, the stage decision is
  MEASURED_DAMAGE_NO_GO and neither seed 46 nor seed 44 is ever evaluated —
  and the non-additivity finding itself explains the Stage 2A-2C failures.
- Development uses a completely new seed-46 split (50/domain, identical
  68-token geometry), disjoint from the seed-42 prior usage, seed 43, seed 44,
  and seed 45. Seeds 43 and 45 are contaminated for Stage 3 and are never
  used. The five development gates mirror Stage 2C (gate A additionally
  requires beating BOTH Robust-Functional and Fragility-Robust); final
  confirmation on seed 44 requires FINAL_CONFIRMATION_GO plus a passing
  independent audit, exactly as before.

Implementation (2026-08-16, local machine, no model inference):

- `src/expert_analysis/stage3_preflight.py` — verifies the full frozen chain
  (STRONG GO / GO / SURROGATE_NO_GO / ROBUST_PRESERVATION_NO_GO /
  FRAGILITY_ROBUST_NO_GO) with the recorded Stage 2C gate values, registry,
  preregistration, and audit hashes, and verifies seed-44 isolation across all
  three stages by hash only.
- `src/expert_analysis/measured_damage.py` — damage definition, per
  (bit-width, layer) chunk checkpoints with hash verification, damage-matrix
  assembly and SHA-256 freezing, the additive prediction, and the two
  preregistered additivity gates.
- `src/expert_analysis/measured_damage_optimization.py` — the min-max MILP
  over measured damages, allocation records, a registry that freezes all eight
  regime/budget allocations before any probe or development NLL and reuses
  every Stage 2B/2C comparator by hash, plus an optimality sanity check
  against all reused comparators.
- `src/expert_analysis/measured_damage_evaluation.py` — seed-46 split
  construction/loading, preregistration freezing and immutability checks, run
  fingerprints, and phase record selection.
- `src/expert_analysis/measured_damage_statistics.py` — the five development
  gates, the FINAL_CONFIRMATION_GO / MEASURED_DAMAGE_NO_GO rule, the final
  requirements 1-5 with the qualified-success rule, and the descriptive
  predicted-versus-realized transfer check.
- `src/expert_analysis/measured_damage_reporting.py` — additivity and phase
  analysis drivers, tables, decision files, figures, and summaries.
- Scripts: `build_stage3_development_split.py`,
  `run_expert_damage_profiling.py`, `solve_measured_damage_allocations.py`,
  `check_damage_additivity.py`, `run_measured_damage_development.py`,
  `run_measured_damage_final.py`, and the standalone
  `audit_measured_damage_preservation.py` (imports no production analysis
  code).
- Tests: 27 new tests; the full suite passes **201/201** locally.
- End-to-end dry validation: a synthetic damage matrix was pushed through the
  real solver, registry freeze, preregistration, additivity analysis,
  development analysis, decisions, figures, and the standalone auditor against
  the real frozen Stage 2B/2C registries; the auditor passed 1,292/1,292
  checks with maximum numeric difference 4.4e-16. No synthetic artifact was
  written into `results/`.

Pending CUDA work, in the preregistered order (split → damage matrix → freeze
allocations + preregistration → audit → additivity gate → audit → seed-46
development → GO/NO-GO → seed-44 only if GO). All commands run from the
repository root on the pinned A40/BF16 environment with batch size 1 and
seed 42:

1. `PYTHONPATH=src python scripts/build_stage3_development_split.py
   --cache-dir .hf_cache` — data only; builds and freezes the disjoint seed-46
   split (minutes; needs dataset downloads, no GPU).
2. `PYTHONPATH=src python scripts/run_expert_damage_profiling.py --device cuda
   --cache-dir .hf_cache` — the measurement run: BF16 + uniform 8/4/3
   references, then 3,072 single-expert states, every state evaluated twice
   with a bitwise reproduction requirement, checkpointed per
   (bit-width, layer) chunk and safely resumable. Verifies the BF16 and
   uniform-4/3 values against the frozen Stage 2C record, then freezes
   `damage/damage_matrix.json`. Roughly 6-12 hours on one A40.
3. `PYTHONPATH=src python scripts/solve_measured_damage_allocations.py` —
   solves and freezes all eight Measured-Damage-Robust allocations, the
   Stage 3 registry with reused Stage 2B/2C comparator hashes, the
   preregistration file plus `stage3_preregistration_sha256.txt`, and the
   allocation summary (seconds, no GPU).
4. `python scripts/audit_measured_damage_preservation.py` — pre-evaluation
   audit gate.
5. `PYTHONPATH=src python scripts/check_damage_additivity.py --device cuda
   --cache-dir .hf_cache` — evaluates the 30 frozen 20%-budget probes on the
   calibration subsets (about 1-2 hours) and applies the additivity gates. If
   no regime passes, it writes MEASURED_DAMAGE_NO_GO and the stage stops with
   seed 46 unevaluated.
6. `python scripts/audit_measured_damage_preservation.py` — additivity audit.
7. `PYTHONPATH=src python scripts/run_measured_damage_development.py --device
   cuda --cache-dir .hf_cache` — seed-46, 20% budget, authorized regime(s),
   all frozen comparators (about 2-4 hours); writes `stage3_decision.json`.
8. `python scripts/audit_measured_damage_preservation.py` — development audit.
9. Only on FINAL_CONFIRMATION_GO: `PYTHONPATH=src python
   scripts/run_measured_damage_final.py --device cuda --cache-dir .hf_cache`
   for the authorized regime(s) at all four budgets on seed 44, followed by
   the final audit. On MEASURED_DAMAGE_NO_GO: stop, preserve the negative
   result, and leave seed 44 untouched permanently under this
   preregistration.

## Stage 3D selection-headroom diagnostics: preregistered / pending CUDA execution

Artifact directory: `results/stage3d_diagnostics/`. Preregistration:
`prereg/stage3d.md`, frozen and committed before the first sweep.

Current status: **preregistered, evaluation set and all 53 configurations
frozen, code written and locally validated, pending CUDA execution**. No loss
has been measured yet.

Authorization and scope: the user authorized this stage on 2026-08-16 as
diagnosis only. Stages 2A, 2B and 2C each predicted which experts to keep at
8 bits under a 20% budget and none beat baseline. Two explanations remain: the
predictors were bad and sensitivity is heterogeneous, or sensitivity is
near-uniform so the objective is flat and no selection rule can win. Stage 3D
separates them. It implements no allocator, runs no 1024-expert sweep, adds no
domain, and changes no budget.

- **Sweep A, 36 runs.** Twenty random protection sets (seeds 46-65) at the 20%
  Stage 2C budget in the 4to8 regime, the same ten of those sets in 3to8, plus
  most-routed, least-routed and no-protection in each regime. Both regimes
  protect exactly 204 of 1024 experts. The random sets are selected once and
  reused across regimes so the arms are paired; the build verifies set equality
  by SHA-256. The 4to8 arm carries the decision; the 3to8 arm can escalate a
  flat primary outcome to inconclusive but can never on its own authorize the
  1024-expert sweep.
- **Sweep B, 16 runs.** Each layer's 64 experts at 4 bits, everything else
  BF16.
- **Sweep C, 1 run.** Every expert at 4 bits with the 16 `mlp.gate.weight`
  router tensors also at 4 bits. Step 0 confirmed the pipeline never quantized
  routers, so per the specification only the quantized-router state is new and
  it is labelled a diagnostic on baseline strength. Its BF16-router comparison
  point is Sweep A's `a_4to8_no_protection` run, the identical expert
  assignment, verified by bit-matrix hash rather than re-evaluated.

Evaluation set: the frozen seed-43 and seed-45 development splits concatenated,
100 examples per domain, 400 examples, 25,600 measured tokens. A fresh split is
not buildable (see the Stage 3 section above). The two source splits are
verified mutually disjoint and free of seed-44 rows. **The seed-44 final
reserve and split seed 46 are both untouched by this stage.** Seeds 43 and 45
were observed by Stages 2B and 2C, which matters for selection and tuning;
Stage 3D does neither, and its expert rankings come from Stage 2B calibration
routing counts measured on the disjoint frozen controlled set.

Both worst-domain definitions are recorded in every table and every JSONL
record: the maximum relative increase over BF16, which is the Stage 1 through
2C definition, and the maximum raw per-domain loss. **Step 5 is applied to the
relative definition only**, fixed in the preregistration before any result
existed.

Implementation (2026-08-16, local machine, no model inference):

- `src/expert_analysis/stage3d_diagnostics.py` — evaluation-set pooling and
  disjointness proof, the frozen memory matrix reload, calibration routing
  counts anchored to a hashed Stage 2B artifact, all three sweeps' protection
  sets, `ReversibleRouterQuantization`, router memory accounting, both
  worst-domain metrics, JSONL run records, and the Step 5 decision functions
  with the thresholds as module constants.
- Scripts: `build_stage3d_evaluation_set.py` (no model),
  `run_stage3d_harness.py` (the four Step 2 checks),
  `run_stage3d_sweeps.py --sweep a|b|c`, and `report_stage3d.py` (no model).
- Tests: 25 new tests; the full suite passes **226/226** locally.
- End-to-end dry validation: synthetic run records were pushed through the real
  reporter, which produced every table, CSV, `stage3d_decision.json`, and
  `SUMMARY.md`, with each Step 5 branch exercised by a unit test. No synthetic
  artifact was written into `results/`.

Frozen locally already, committed: the evaluation set manifest and all 53
configurations, registry SHA-256
`74d36c2d6d9694611155310a5d6a167703ad2934a34aa20e8c44429f7db74a37`.

Pending CUDA work, in order, from the repository root on the pinned A40/BF16
environment with batch size 1 and seed 42:

1. `PYTHONPATH=src python scripts/run_stage3d_harness.py --device cuda
   --cache-dir .hf_cache` — the four correctness checks. Check 4 reproduces two
   Stage 1 single-expert measurements at exact tolerance and stops the run on
   any disagreement, reporting its size.
2. `PYTHONPATH=src python scripts/run_stage3d_sweeps.py --sweep a --device cuda
   --cache-dir .hf_cache`, then
   `python scripts/report_stage3d.py --sweeps a`. Sweep A is reported before
   Sweep B starts.
3. `--sweep b`, then `--sweep c`, then
   `python scripts/report_stage3d.py`.

Records append and fsync one at a time and per-domain losses checkpoint, so a
crash loses at most the run in flight and a resumed run recomputes nothing it
already finished.

## RACE Stage 0 residency oracle-headroom: completed / audited STRONG GO

Final decision: **`RACE_STAGE0_STRONG_GO`**.

Artifact/code area: `stage3_residency/`. Durable final report:
`stage3_residency/reports/stage3_residency_headroom_report.md`. Detailed report:
`stage3_residency/results/full/report/stage3_residency_headroom_report.md`. Frozen
file-level archive manifest:
`stage3_residency/reports/final_archive_manifest.json`.

This is a separately authorized residency experiment, not a continuation or
reinterpretation of the existing Stage 3 measured-damage experiment or Stage 3D
selection-headroom diagnostics. Those frozen stages and all Stage 2A/2B/2C
decisions remain unchanged. Stage 0 implemented no RACE optimizer, learning,
regret method, quantization, precision allocation, compression, fine-tuning, or
weight change.

Scientific question: whether an offline future-aware expert-residency optimum has
enough transfer-cost headroom over strong simple policies on real OLMoE decode
routing to justify later RACE design. The primary comparison is the best eligible
simple policy among LRU, LFU, globally calibration-selected LFU-decay, and
calibration-only Static Hotset versus the offline oracle. Random uses five fixed
seeds and is descriptive, never eligible as best simple.

### Frozen provenance

The two commit identifiers have different meanings:

- source/base commit recorded by the preregistration:
  `48fe6e2dd9b42af8b7d30cff536a06cd49181eb9`;
- actual Stage 0 runtime commit recorded by the trace:
  `0f70c61131b877dd9c297663886d563d9e27f55b`.

Additional frozen identifiers:

- Stage 0 source bundle:
  `a88d593a1ab686138b5c62e1be23b7d08200169c5f62826af272c11a5eb287d4`;
- preregistered full configuration:
  `17017dd4c3019e1ea625d21a7102afefdaa2c03129381f3f403c0184cc6576fc`;
- full logical trace:
  `ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea`;
- frozen evaluation configuration:
  `7ce228983b6547d61341757234e77ca7f59a4d0ba53b1e04b64243e9b2ea0971`;
- Static Hotset scores:
  `f26bef3216f1ed5c5ae6124d993a9d4441443aba6e7842f106e4aacb1eb634e6`.

### Full execution and reconstructed command

The real full trace was generated on an NVIDIA A40 using bfloat16, Python 3.12.3,
PyTorch 2.8.0+cu128, greedy decode seed 42, up to 128 new tokens, and the pinned
`allenai/OLMoE-1B-7B-0924` revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`. Dataset revisions and exact selected
IDs are recorded in `stage3_residency/traces/full/routing_trace.metadata.json`.
The trace completed at 2026-08-19T22:41:22Z; evaluation completed at
2026-08-20T01:09:31Z.

The repository-recorded wrapper sequence reconstructing the run is:

```bash
stage3_residency/scripts/run_pilot.sh
stage3_residency/scripts/generate_traces.sh
stage3_residency/scripts/run_full.sh
```

The full trace contains 400 prompts (100/domain), 51,112 generated tokens,
817,792 atomic token/layer events, and 6,542,336 requested experts. The full
evaluation contains 600 policy conditions, 4,800 result rows, and 82,800
per-sequence rows. The first 20 prompts/domain are calibration-only; the remaining
80/domain are evaluation-only. LFU-decay `alpha=0.95` was selected once from the
preregistered three-value calibration grid.

### Validation outcome

- all 400 prompt-chunk logical hashes match their metadata and their ordered
  arrays reproduce the aggregate trace byte-for-byte;
- aggregate event indices, sequence/token/layer ordering, atomic top-8 uniqueness,
  expert ranges, router weights, and the full logical trace hash validate;
- calibration LFU-decay totals, selected alpha, Static Hotset matrix, and Static
  Hotset hash reproduce exactly from the raw calibration requests;
- decision-driving LFU-decay and oracle simulations replay exactly from raw
  requests across all ten workloads and five capacities, matching events,
  requests, hits, misses, admissions, evictions, and maximum occupancy for 100
  saved conditions;
- generalized farthest-future matches exact DP on 6,690 exhaustive plus 500
  fixed-seed random tiny atomic traces across lambdas 0/0.25/0.5/1.0 with maximum
  cost difference 0.0;
- oracle dominance, cache monotonicity, top-k capacity equivalence, zero- and
  unlimited-cache limits, deterministic evaluation replay, event accounting,
  calibration/evaluation separation, byte proportionality, per-sequence
  aggregation, and five-seed Random coverage all pass.

### Main comparison and decision

Primary oracle headroom over the strongest eligible simple policy is:

| Regime | C=8 | C=12 | C=16 | C=24 | C=32 |
|---|---:|---:|---:|---:|---:|
| stationary | 0.00% | 17.75% | 26.55% | 37.59% | 45.48% |
| abrupt | 0.00% | 17.56% | 26.46% | 37.67% | 45.67% |
| repeated | 0.00% | 17.66% | 26.42% | 37.44% | 45.65% |
| mixed | 0.00% | 18.24% | 27.27% | 38.64% | 46.73% |

The preregistered STRONG-GO rule passes at capacities 12/16/24/32 in every
regime, so the final decision is `RACE_STAGE0_STRONG_GO`. Capacity 8 equals
atomic top-k and therefore offers no policy freedom. Abrupt-shift headroom is
similar to stationary headroom; the result demonstrates general future-aware
residency headroom rather than a uniquely shift-driven effect.

### Archival and claim limitations

The pilot frozen configuration, evaluation, and audit outputs remain archived,
but `stage3_residency/traces/pilot/` is absent. The pilot therefore cannot be
independently replayed from the current checkout. This is an archival limitation,
not a failure of the validated full run: the complete full raw trace is present,
logically hashed, reconstructed from all 400 chunks, and independently replayed.

Bootstrap intervals are conditional on the frozen workload ordering and reweight
per-sequence contributions; stateful cache trajectories are not regenerated under
reordered bootstrap workloads.

Results concern simulated expert residency/miss counts; no end-to-end latency
improvement is claimed.

No defensible host-device transfer calibration was collected. All experts have
equal parameter size, so byte-weighted cost is proportional to miss count and is
not an independent latency model. The evidence is conditional on one checkpoint,
one greedy decode panel, sequential prompt workloads, and independent per-layer
caches.

Next action: proceed to design RACE as a separately authorized stage. Do not claim
that RACE works until an online method is implemented and evaluated, and do not
claim inference speedup without end-to-end runtime measurement.

## Fresh-session checklist

A new Codex session should:

1. Read `AGENTS.md`, `README.md`, and this file.
2. Run `git status --short` and preserve local result artifacts.
3. Inspect the relevant run's `collection_config.json`, metadata, `results.json`,
   and NPZ arrays before making new numerical claims.
4. Treat `results/expert_domain_causal_validation/` as the current validated
   controlled run; validate its raw NPZ/JSON/CSV artifacts for exact claims.
5. Treat `results/expert_domain_balanced_causal_validation/` as the completed,
   independently audited balanced causal run; preserve its frozen preregistration,
   masking checkpoints, final tables, and figures.
6. Keep all rankings layer-wise; expert IDs are not comparable across layers.
7. Keep “routing utilization,” “gate mass,” and “functional contribution proxy”
   terminology unless an intervention supports a stronger claim.
8. Describe selected-route masking as a causal loss-sensitivity intervention, not
   as expert deletion or a quantization simulation.
9. Treat `results/expert_quantization_pilot/pilot_panel_preregistered.json` as
   frozen. Never replace experts using masking or quantization outcomes.
10. Treat `results/expert_quantization_pilot/` as the completed, independently
    audited Stage-1 `GO` run. Preserve its raw checkpoints, decision, summary, and
    `independent_audit.json`.
11. Treat Stage-2A as complete and independently audited with final decision
    `SURROGATE_NO_GO`. Preserve its small result package and raw capture artifacts;
    do not reinterpret the failed AOD/GQS gates.
12. The full cost matrix was not authorized or generated. Any optimizer that
    predicts per-expert quantization delta NLL remains scientifically blocked.
13. Treat Stage 2B development as complete and independently audited with final
    decision `ROBUST_PRESERVATION_NO_GO`. Preserve its frozen allocations, splits,
    development checkpoints, and negative result without modification. The final
    seed-44 split may be evaluated only by an authorized Stage 2C final
    confirmation after a preregistered seed-45 `FINAL_CONFIRMATION_GO`.
14. Treat Stage 2C development as complete and independently audited with final
    decision `FRAGILITY_ROBUST_NO_GO`. Preserve its frozen fragility record,
    allocations, seed-45 split, development checkpoints, and negative result
    without modification. No alternative fragility weighting may be searched.
15. Treat `results/measured_damage_preservation/` as the Stage 3 workspace.
    Stage 3 measures per-expert damage; it never predicts it. Seeds 43 and 45
    are contaminated for Stage 3 and must never be used; the damage matrix,
    allocations, and preregistration must be produced only by the documented
    commands, in the preregistered order, and never edited afterward; the
    additivity gate decides whether seed 46 may be evaluated; seed 44 may be
    evaluated only after a preregistered `FINAL_CONFIRMATION_GO` plus a
    passing independent audit. Stage 3 is on hold: it may not run until
    Stage 3D Sweep A returns `HEADROOM`, and its seed-46 split build fails as
    written because the coding pool is exhausted.
16. Treat `prereg/stage3d.md` and
    `results/stage3d_diagnostics/allocations/allocation_registry.json` as
    frozen. Stage 3D is diagnosis only: no allocator, no 1024-expert sweep, no
    new domain, no changed budget, no threshold edited after seeing numbers.
    Step 5 is applied to the relative worst-domain definition; the raw one is
    reported but never decides. Seeds 46-65 select protection sets there and
    build no data split, so split seed 46 stays available. Report a surprising
    sweep result; do not investigate it further without asking.
17. Treat `stage3_residency/configs/full_preregistered.json`, the complete raw
    trace under `stage3_residency/traces/full/`, the frozen evaluation, final
    reports, and `stage3_residency/reports/final_archive_manifest.json` as the
    immutable audited RACE Stage 0 archive. The decision is
    `RACE_STAGE0_STRONG_GO`. The missing raw pilot trace is an archival limitation,
    not a failure of the validated full run. Do not implement RACE as part of
    Stage 0; any RACE design is a separately authorized next stage.
18. Update this handoff after every new validated run.
