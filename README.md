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

## Outputs

The analysis creates:

- expert_importance_by_domain.csv
- cross_domain_correlations.csv
- topk_overlap.csv
- routing_vs_functional_correlation.csv
- domain_specialized_experts.csv
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

The tests cover current tensorized OLMoE-style experts, older ModuleList-style
experts, padding exclusion, optional gradient attribution, bootstrapping, ranking,
top-k overlap, and complete analysis/report artifact generation.

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
