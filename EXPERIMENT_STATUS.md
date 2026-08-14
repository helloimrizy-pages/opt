# Expert-domain importance experiment: persistent handoff

Last updated: 2026-08-14

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
modified checkpoint has been saved, fine-tuned, or compressed. The newly
implemented Stage-1 code applies only temporary, exactly restored expert QDQ.

Current decision: the mandatory Stage-1 reversible quantization-sensitivity pilot
is **GO** and complete after production execution and independent raw-artifact audit
on the NVIDIA A40. Four-bit QDQ passed all four preregistered gates; the 3-bit
fallback was not triggered. This scientifically justifies a separately designed
distributionally robust mixed-precision Stage 2, which remains pending and was not
implemented in this run. Prompt-only, controlled causal, and balanced causal
validation remain complete and independently audited.

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

The balanced-panel implementation adds deterministic baseline-only
preregistration, strict integrity gates, intervention-level resume support, paired
control/aggregate bootstraps, figures, and reporting. The new Stage-1 pilot adds
temporary expert-only fake quantization/QDQ with exact restoration; it does not
save modified model weights or implement a mixed-precision allocator.

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
11. Robust mixed-precision Stage 2 is scientifically justified but remains pending.
    Do not implement it unless the user explicitly authorizes that later stage.
12. Update this handoff after every new validated run.
