#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis import DEFAULT_MODEL, DOMAINS
from expert_analysis.analysis import analyze_results
from expert_analysis.collection import (
    collect_prepared_domain,
    collection_fingerprint,
    load_resumable_domain,
    run_smoke_validation,
    save_domain_result,
)
from expert_analysis.controlled import PreparedDomainExamples, prepare_controlled_domains
from expert_analysis.datasets import load_domain_examples
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions, read_json
from expert_analysis.masking import (
    MaskTarget,
    run_masking_validation,
    validate_masking_mechanism,
)
from expert_analysis.metrics import DomainStatistics
from expert_analysis.modeling import (
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)
from expert_analysis.plotting import create_all_figures
from expert_analysis.report import write_summary


DEFAULT_MASK_TARGETS = (
    "11:27:coding:reasoning",
    "10:56:coding:general",
    "1:25:coding:general",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled OLMoE domain validation with exact token matching, "
            "split-half reliability, and reversible expert masking."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument(
        "--tokens-per-example",
        type=int,
        default=64,
        help=(
            "Exact measured source positions per example. One additional look-ahead "
            "token is retained as the final next-token label."
        ),
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=1000,
        help="Maximum shuffled non-empty candidates loaded per domain before length filtering.",
    )
    parser.add_argument(
        "--neutral-prefix",
        default="Input:\n",
        help="Literal shared prefix prepended as separately tokenized IDs in every domain.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dataset-revision",
        action="append",
        default=[],
        metavar="DOMAIN=REVISION",
    )
    parser.add_argument(
        "--allow-dataset-substitution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mask-expert",
        action="append",
        default=None,
        metavar="LAYER:EXPERT[:HIGH:LOW]",
        help=(
            "Pre-registered expert intervention; repeat for multiple targets. "
            f"Default: {', '.join(DEFAULT_MASK_TARGETS)}."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--split-half-replicates", type=int, default=100)
    parser.add_argument("--mask-bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-smoke-validation", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.num_examples < 2:
        parser.error("num-examples must be at least 2")
    if args.tokens_per_example < 2:
        parser.error("tokens-per-example must be at least 2")
    if args.candidate_pool_size < args.num_examples:
        parser.error("candidate-pool-size must be at least num-examples")
    if args.max_length < args.tokens_per_example + 2:
        parser.error("max-length is too small for the prefix, measured tokens, and look-ahead")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if min(
        args.bootstrap_replicates,
        args.split_half_replicates,
        args.mask_bootstrap_replicates,
    ) < 0:
        parser.error("all replicate counts must be nonnegative")
    try:
        args.mask_targets = [
            MaskTarget.parse(value)
            for value in (args.mask_expert or DEFAULT_MASK_TARGETS)
        ]
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    set_reproducible_seed(args.seed, deterministic=args.deterministic)
    runtime = resolve_runtime(args.device, args.dtype)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_revisions = _parse_dataset_revisions(args.dataset_revision)

    print(
        f"Loading {args.model} on {runtime.description} as "
        f"{str(runtime.dtype).replace('torch.', '')}",
        flush=True,
    )
    bundle = load_model_and_tokenizer(
        checkpoint=args.model,
        runtime=runtime,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    layer_specs = discover_moe_layers(bundle.model)
    architecture = architecture_metadata(bundle.model, layer_specs)
    print(
        f"Discovered {len(layer_specs)} MoE layers, "
        f"{layer_specs[0].num_experts} experts/layer, top-k values "
        f"{sorted({spec.top_k for spec in layer_specs})}",
        flush=True,
    )

    candidates = {}
    for domain in args.domains:
        print(f"[{domain}] loading neutral candidate pool...", flush=True)
        candidates[domain] = load_domain_examples(
            domain=domain,
            num_examples=args.candidate_pool_size,
            seed=args.seed,
            cache_dir=args.cache_dir,
            revision=dataset_revisions.get(domain),
            include_answers=False,
            allow_substitution=args.allow_dataset_substitution,
            format_style="neutral_content",
        )
    prepared, corpus_manifest = prepare_controlled_domains(
        candidates,
        tokenizer=bundle.tokenizer,
        num_examples=args.num_examples,
        measured_tokens_per_example=args.tokens_per_example,
        neutral_prefix=args.neutral_prefix,
        max_length=args.max_length,
    )
    print(
        f"Controlled corpus validated: {args.num_examples} examples/domain, "
        f"exactly {args.tokens_per_example} measured positions/example, "
        f"{corpus_manifest['measured_tokens_per_domain']} measured tokens/domain, "
        f"model sequence length {corpus_manifest['model_sequence_length']}.",
        flush=True,
    )

    fingerprint_basis = _fingerprint_basis(
        args,
        bundle.resolved_revision,
        str(runtime.dtype).replace("torch.", ""),
        dataset_revisions,
        prepared,
    )
    fingerprint = collection_fingerprint(fingerprint_basis)
    config_path = output_dir / "collection_config.json"
    configuration_changed = False
    if config_path.exists():
        previous = read_json(config_path)
        configuration_changed = previous.get("collection_fingerprint") != fingerprint
        has_artifacts = any((output_dir / "domains").glob("*.npz")) or (
            output_dir / "masking"
        ).exists()
        if configuration_changed and has_artifacts and not args.overwrite:
            raise RuntimeError(
                f"{config_path} describes a different completed run. Use --overwrite "
                "or a different --output-dir."
            )

    config: dict[str, Any] = {
        **fingerprint_basis,
        "collection_fingerprint": fingerprint,
        "quick": args.num_examples <= 100,
        "batch_size": args.batch_size,
        "device": str(runtime.device),
        "device_description": runtime.description,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "deterministic": args.deterministic,
        "allow_dataset_substitution": args.allow_dataset_substitution,
        "candidate_pool_size": args.candidate_pool_size,
        "controlled_input": {
            key: value for key, value in corpus_manifest.items() if key != "domains"
        },
        "mask_targets": [
            {
                "layer": item.model_layer_index,
                "expert_id": item.expert_id,
                "expected_high_domain": item.expected_high_domain,
                "expected_low_domain": item.expected_low_domain,
            }
            for item in args.mask_targets
        ],
        "package_versions": package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(config_path, config)
    atomic_write_json(output_dir / "architecture.json", architecture)
    atomic_write_json(output_dir / "controlled_corpus.json", corpus_manifest)
    _save_or_validate_inputs(
        output_dir,
        prepared,
        resume=args.resume and not configuration_changed,
    )

    smoke: dict[str, Any] | None = None
    masking_smoke: dict[str, Any] | None = None
    if not args.skip_smoke_validation:
        print("Running instrumentation and masking smoke validation...", flush=True)
        smoke = run_smoke_validation(
            bundle,
            layer_specs,
            max_length=min(args.max_length, 128),
            compute_gradient_attribution=False,
        )
        smoke["controlled_input_validation"] = {
            "passed": True,
            "same_prefix_token_ids": corpus_manifest["same_prefix_token_ids"],
            "same_model_sequence_length": corpus_manifest[
                "same_model_sequence_length"
            ],
            "same_measurement_length_distribution": corpus_manifest[
                "same_measurement_length_distribution"
            ],
            "same_total_measurement_budget": corpus_manifest[
                "same_total_measurement_budget"
            ],
        }
        atomic_write_json(output_dir / "smoke_validation.json", smoke)
        masking_smoke = validate_masking_mechanism(
            bundle, layer_specs[0], prepared[args.domains[0]]
        )
        atomic_write_json(
            output_dir / "masking_smoke_validation.json", masking_smoke
        )
        print(
            f"Mask validation passed by zeroing {masking_smoke['zeroed_routes']} "
            f"routes for L{masking_smoke['layer']}/E{masking_smoke['expert_id']}; "
            "no hooks leaked.",
            flush=True,
        )
    elif not (output_dir / "smoke_validation.json").exists():
        print("Warning: smoke validation explicitly skipped.", flush=True)

    if args.smoke_only:
        print(f"Controlled smoke-only run complete: {output_dir}", flush=True)
        return 0

    completed: list[str] = []
    domain_summaries: dict[str, Any] = {}
    collected_statistics: dict[str, DomainStatistics] = {}
    for domain in args.domains:
        existing = (
            load_resumable_domain(output_dir, domain, fingerprint, layer_specs)
            if args.resume and not configuration_changed
            else None
        )
        if existing is not None:
            result = existing
            print(
                f"[{domain}] resume: {result.statistics.num_examples} examples, "
                f"{result.statistics.token_counts.sum()} measured tokens",
                flush=True,
            )
        else:
            result = collect_prepared_domain(
                bundle,
                layer_specs,
                prepared[domain],
                batch_size=args.batch_size,
            )
            save_domain_result(output_dir, domain, result, fingerprint)
        completed.append(domain)
        domain_summaries[domain] = result.metadata
        collected_statistics[domain] = result.statistics
        atomic_write_json(
            output_dir / "collection_manifest.json",
            {
                "collection_fingerprint": fingerprint,
                "completed_domains": completed,
                "domain_summaries": domain_summaries,
            },
        )

    print("Computing cross-domain statistics and split-half reliability...", flush=True)
    results = analyze_results(
        input_dir=output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed,
        specialized_per_layer=10,
        split_half_replicates=args.split_half_replicates,
    )
    print("Running pre-registered expert-masking interventions...", flush=True)
    masking = run_masking_validation(
        bundle,
        layer_specs,
        prepared,
        collected_statistics,
        args.mask_targets,
        output_dir=output_dir,
        collection_fingerprint=fingerprint,
        batch_size=args.batch_size,
        bootstrap_replicates=args.mask_bootstrap_replicates,
        seed=args.seed + 15_487,
        resume=args.resume and not configuration_changed,
    )
    results["controlled_corpus"] = corpus_manifest
    results["masking_smoke_validation"] = masking_smoke
    results["expert_masking"] = masking
    results["expert_masking_loss"] = masking["loss_rows"]
    results["expert_masking_domain_contrasts"] = masking["domain_contrasts"]
    atomic_write_json(output_dir / "results.json", results)
    write_summary(results, output_dir / "SUMMARY.md")
    figure_paths = [] if args.skip_plots else create_all_figures(results, output_dir)

    print(f"Controlled causal validation complete: {output_dir}", flush=True)
    print(f"Summary: {output_dir / 'SUMMARY.md'}", flush=True)
    print(f"Machine-readable results: {output_dir / 'results.json'}", flush=True)
    print(f"Created {len(figure_paths)} figure files.", flush=True)
    return 0


def _fingerprint_basis(
    args: argparse.Namespace,
    resolved_model_revision: str | None,
    dtype: str,
    dataset_revisions: dict[str, str],
    prepared: dict[str, PreparedDomainExamples],
) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "domains": list(args.domains),
        "num_examples": args.num_examples,
        "max_length": args.max_length,
        "seed": args.seed,
        "include_reference_answers": False,
        "compute_gradient_attribution": False,
        "dtype": dtype,
        "dataset_revisions": dataset_revisions,
        "prompt_style": "neutral_fixed_token_control",
        "neutral_prefix": args.neutral_prefix,
        "tokens_per_example": args.tokens_per_example,
        "lookahead_tokens_per_example": 1,
        "candidate_pool_size": args.candidate_pool_size,
        "selected_inputs": {
            domain: {
                "repository": item.metadata["repository"],
                "config": item.metadata.get("config"),
                "split": item.metadata["split"],
                "resolved_revision": item.metadata.get("resolved_revision"),
                "dataset_fingerprint": item.metadata.get("dataset_fingerprint"),
                "selected_example_ids": item.metadata["selected_example_ids"],
                "input_ids_sha256": item.metadata["control"]["input_ids_sha256"],
            }
            for domain, item in prepared.items()
        },
    }


def _save_or_validate_inputs(
    output_dir: Path,
    prepared: dict[str, PreparedDomainExamples],
    resume: bool,
) -> None:
    for domain, item in prepared.items():
        path = output_dir / "controlled_inputs" / f"{domain}.npz"
        if resume and path.exists():
            existing = PreparedDomainExamples.load(path, domain, item.metadata)
            if not (
                np.array_equal(existing.input_ids, item.input_ids)
                and np.array_equal(existing.attention_mask, item.attention_mask)
                and np.array_equal(existing.measurement_mask, item.measurement_mask)
            ):
                raise RuntimeError(
                    f"Existing controlled input artifact for {domain!r} differs despite "
                    "a matching collection fingerprint"
                )
        else:
            item.save(path)


def _parse_dataset_revisions(values: list[str]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --dataset-revision {value!r}; expected DOMAIN=REVISION"
            )
        domain, revision = value.split("=", 1)
        if domain not in DOMAINS or not revision:
            raise ValueError(
                f"Invalid --dataset-revision {value!r}; domain must be one of {DOMAINS}"
            )
        revisions[domain] = revision
    return revisions


if __name__ == "__main__":
    raise SystemExit(main())
