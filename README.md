# OLMoE expert importance across domains

This repository implements a diagnostic experiment for the question:

> Do per-layer expert-importance rankings in OLMoE change across general text,
> mathematics, coding, and reasoning inputs?

The latest validated runs, cross-run comparison, current go/no-go decision, and
next experimental gate are recorded in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).
Future Codex sessions are directed there by the repository-level `AGENTS.md`.

It collects routing utilization, selected gate mass, and a functional-contribution
proxy. It does not quantize, compress, fine-tune, generate from, or modify the model.

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
top-k overlap, causal report generation, and PNG/PDF figure creation.

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
