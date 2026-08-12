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

Current decision: **strong-support GO for controlled validation, not yet for
quantization or compression**. The domain-dependence signal survives removal of
reference answers, but length, prompt-format, dataset, and proxy-metric controls
remain necessary before using these rankings for bit allocation.

The controlled causal-validation code is implemented but has not yet been run on
the full OLMoE checkpoint. Do not infer its outcome from the implementation tests.

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

## Required next experiment before compression

The required controlled experiment is now implemented in
`scripts/run_causal_validation.py`. Its exact pinned RunPod command is in the
`Controlled causal validation` section of `README.md`.

The implementation performs the following controls:

1. Uses the identical separately tokenized neutral prefix `Input:\n` for every
   example, with no domain label or reference answer.
2. Selects examples from deterministic shuffled candidate pools and retains exactly
   64 measured content source positions plus one look-ahead label per example.
3. Gives every domain the exact same length distribution and 6,400 measured-token
   budget at 100 examples.
4. Aligns expert statistics, expert zeroing, and next-token cross-entropy to the
   same source positions; prefix and look-ahead positions are excluded.
5. Computes repeated same-domain split-half Spearman correlations and
   Spearman–Brown full-sample estimates for all three metrics.
6. Repeats all cross-domain rankings, bootstraps, top-k comparisons, specialization,
   and routing-versus-functional analyses.
7. Pre-registers layer 11/expert 27, layer 10/expert 56, and layer 1/expert 25 from
   the prior prompt-only run. Their high-versus-low contrasts are pre-registered as
   Coding–Reasoning, Coding–General, and Coding–General, respectively. It then
   zeroes their selected gate weights without rerouting and measures paired
   per-example next-token loss changes.
8. Produces paired-bootstrap loss intervals, high-versus-low domain contrasts, and
   activation-proxy versus causal-loss alignment diagnostics.

The masking smoke test dynamically finds a routed expert, verifies finite baseline
and masked losses, verifies that the loss actually changes, and checks hook cleanup.
Every full intervention also verifies that its zeroed route counts exactly equal the
independently collected routing counts. No parameters are modified.

Local validation completed with all 14 tests passing in the project environment.
This includes a tiny current Hugging Face `OlmoeForCausalLM`, bitwise checks that
parameters remain unchanged, hook-leak checks, end-to-end masking artifacts, causal
summary generation, and PNG/PDF figure creation. The actual checkpoint tokenizer
was also loaded from the local cache under Transformers 5.15: `Input:\n` maps to
token IDs `[8982, 27, 187]`, and the controlled 64-position construction produces a
68-token model sequence with one excluded look-ahead token. These implementation
checks are not a substitute for the pending full-checkpoint RunPod run.

Because fixed-length control requires at least 65 content tokens, the experiment
conditions on longer examples. The generated summary reports candidate-pool size,
eligible count, selected count, and selected original-token range for every domain;
interpret the results as applying to these length-matched subsets.

After the RunPod output is copied back, audit the raw NPZ arrays, exact token-budget
invariants, split-half reliability, loss route-count equality, bootstrap intervals,
and generated report. Only then should the project decide whether to expand causal
validation or begin a small reversible mixed-precision pilot.

## Fresh-session checklist

A new Codex session should:

1. Read `AGENTS.md`, `README.md`, and this file.
2. Run `git status --short` and preserve local result artifacts.
3. Inspect the relevant run's `collection_config.json`, metadata, `results.json`,
   and NPZ arrays before making new numerical claims.
4. Treat the prompt-only result as the current primary diagnostic and the
   with-answer result as a sensitivity comparison.
5. Treat `results/expert_domain_causal_validation/` as pending until a real RunPod
   run is present and independently audited; unit tests do not constitute a result.
6. Keep all rankings layer-wise; expert IDs are not comparable across layers.
7. Keep “routing utilization,” “gate mass,” and “functional contribution proxy”
   terminology unless an intervention supports a stronger claim.
8. Describe selected-route masking as a causal loss-sensitivity intervention, not
   as expert deletion or a quantization simulation.
9. Do not begin quantization unless the user explicitly advances the project stage.
10. Update this handoff after every new validated run.
