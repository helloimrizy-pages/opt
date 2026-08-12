#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis import DEFAULT_MODEL, DOMAINS
from expert_analysis.collection import (
    collect_domain,
    collection_fingerprint,
    freeze_parameters_for_gradient_attribution,
    load_resumable_domain,
    run_smoke_validation,
    save_domain_result,
)
from expert_analysis.datasets import load_domain_examples
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import (
    atomic_write_json,
    package_versions,
    read_json,
)
from expert_analysis.modeling import (
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect domain-conditioned OLMoE expert statistics."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS)
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Examples per domain (default: 500, or 100 with --quick).",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expert_domain_importance"),
    )
    parser.add_argument("--quick", action="store_true")
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
        help="Pin a dataset revision; may be repeated.",
    )
    parser.add_argument(
        "--allow-dataset-substitution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Exclude benchmark reference answers/solutions from measured text.",
    )
    parser.add_argument("--compute-gradient-attribution", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing artifacts made with a different collection configuration.",
    )
    parser.add_argument("--skip-smoke-validation", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Validate model loading/instrumentation on four fixed examples, then stop.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    args.num_examples = (
        args.num_examples if args.num_examples is not None else (100 if args.quick else 500)
    )
    if args.num_examples < 1 or args.max_length < 2 or args.batch_size < 1:
        parser.error("num-examples and batch-size must be positive; max-length must be >= 2")
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

    fingerprint_basis = {
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": bundle.resolved_revision,
        "domains": list(args.domains),
        "num_examples": args.num_examples,
        "max_length": args.max_length,
        "seed": args.seed,
        "include_reference_answers": not args.prompts_only,
        "compute_gradient_attribution": args.compute_gradient_attribution,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "dataset_revisions": dataset_revisions,
    }
    fingerprint = collection_fingerprint(fingerprint_basis)
    config_path = output_dir / "collection_config.json"
    configuration_changed = False
    if config_path.exists():
        previous = read_json(config_path)
        configuration_changed = previous.get("collection_fingerprint") != fingerprint
        has_completed_domains = any((output_dir / "domains").glob("*.npz"))
        if configuration_changed and has_completed_domains and not args.overwrite:
            raise RuntimeError(
                f"{config_path} describes a different completed run. Use --overwrite "
                "or a different --output-dir."
            )
        if configuration_changed and not has_completed_domains:
            print(
                "Replacing smoke-only configuration because no completed domain "
                "artifacts exist.",
                flush=True,
            )
    config: dict[str, Any] = dict(fingerprint_basis)
    config.update(
        {
            "collection_fingerprint": fingerprint,
            "quick": args.quick,
            "batch_size": args.batch_size,
            "device": str(runtime.device),
            "device_description": runtime.description,
            "dtype": str(runtime.dtype).replace("torch.", ""),
            "deterministic": args.deterministic,
            "allow_dataset_substitution": args.allow_dataset_substitution,
            "package_versions": package_versions(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_json(config_path, config)
    atomic_write_json(output_dir / "architecture.json", architecture)

    if args.compute_gradient_attribution:
        print(
            "Gradient attribution enabled: freezing parameter gradients and using "
            "differentiable input embeddings. This mode is substantially slower.",
            flush=True,
        )
        freeze_parameters_for_gradient_attribution(bundle.model)

    if not args.skip_smoke_validation:
        print("Running four-example instrumentation validation...", flush=True)
        smoke = run_smoke_validation(
            bundle,
            layer_specs,
            max_length=min(args.max_length, 128),
            compute_gradient_attribution=args.compute_gradient_attribution,
        )
        atomic_write_json(output_dir / "smoke_validation.json", smoke)
        print(
            f"Validation passed. Diagnostic ranking for layer "
            f"{smoke['diagnostic_layer']}:",
            flush=True,
        )
        for item in smoke["top_experts_by_contribution"]:
            print(
                f"  expert {item['expert_id']:>2}: contribution="
                f"{item['normalized_contribution']:.5f}, routing="
                f"{item['routing_frequency']:.5f}, gate={item['gate_mass']:.5f}",
                flush=True,
            )
    elif not (output_dir / "smoke_validation.json").exists():
        print("Warning: smoke validation explicitly skipped.", flush=True)

    if args.smoke_only:
        print(f"Smoke-only run complete: {output_dir}", flush=True)
        return 0

    completed: list[str] = []
    domain_summaries: dict[str, Any] = {}
    for domain in args.domains:
        existing = (
            load_resumable_domain(output_dir, domain, fingerprint, layer_specs)
            if args.resume and not configuration_changed
            else None
        )
        if existing is not None:
            print(
                f"[{domain}] resume: using completed artifact "
                f"({existing.statistics.num_examples} examples, "
                f"{existing.statistics.token_counts.sum()} tokens)",
                flush=True,
            )
            completed.append(domain)
            domain_summaries[domain] = existing.metadata
            continue
        print(f"[{domain}] loading dataset...", flush=True)
        examples = load_domain_examples(
            domain=domain,
            num_examples=args.num_examples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            revision=dataset_revisions.get(domain),
            include_answers=not args.prompts_only,
            allow_substitution=args.allow_dataset_substitution,
        )
        if examples.metadata.get("substituted"):
            print(
                f"[{domain}] primary dataset unavailable; using "
                f"{examples.metadata['repository']} ({examples.metadata['config']})",
                flush=True,
            )
        result = collect_domain(
            bundle,
            layer_specs,
            examples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            compute_gradient_attribution=args.compute_gradient_attribution,
        )
        save_domain_result(output_dir, domain, result, fingerprint)
        completed.append(domain)
        domain_summaries[domain] = result.metadata
        atomic_write_json(
            output_dir / "collection_manifest.json",
            {
                "collection_fingerprint": fingerprint,
                "completed_domains": completed,
                "domain_summaries": domain_summaries,
            },
        )
    atomic_write_json(
        output_dir / "collection_manifest.json",
        {
            "collection_fingerprint": fingerprint,
            "completed_domains": completed,
            "domain_summaries": domain_summaries,
        },
    )
    print(f"Collection complete: {output_dir}", flush=True)
    print(
        f"Next: python scripts/analyze_expert_importance.py --input-dir {output_dir}",
        flush=True,
    )
    return 0


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
