#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.balanced import (  # noqa: E402
    BALANCED_DOMAINS,
    EXPECTED_MODEL,
    EXPECTED_MODEL_REVISION,
    EXPECTED_PREFIX,
    EXPECTED_PREFIX_IDS,
    canonical_sha256,
    file_sha256,
    load_controlled_source,
)
from expert_analysis.balanced_analysis import (  # noqa: E402
    analyze_balanced_interventions,
    create_balanced_figures,
    intervention_panel,
    write_balanced_outputs,
    write_balanced_summary,
)
from expert_analysis.collection import collect_prepared_domain  # noqa: E402
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed  # noqa: E402
from expert_analysis.io_utils import (  # noqa: E402
    atomic_write_json,
    package_versions,
    read_json,
)
from expert_analysis.masking import (  # noqa: E402
    LossStatistics,
    evaluate_next_token_loss,
    validate_masking_mechanism,
)
from expert_analysis.modeling import (  # noqa: E402
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)


MASKING_METHOD = "zero_selected_gate_weight_without_rerouting"
MASKING_SCOPE = "measured_next_token_source_positions_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen balanced causal panel. Expert identities are read only from "
            "selected_experts_preregistered.json; this script never selects or replaces them."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expert_domain_balanced_causal_validation"),
    )
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-existing-interventions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--baseline-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reproducing source per-example baseline loss sums.",
    )
    parser.add_argument(
        "--baseline-rtol",
        type=float,
        default=1e-7,
        help="Relative tolerance for reproducing source per-example baseline loss sums.",
    )
    args = parser.parse_args()
    if args.model != EXPECTED_MODEL or args.model_revision != EXPECTED_MODEL_REVISION:
        parser.error("The balanced run is pinned to the validated OLMoE checkpoint/revision")
    if args.batch_size != 1:
        parser.error("The frozen balanced run requires batch size 1")
    if args.seed != 42:
        parser.error("The frozen balanced run requires seed 42")
    if args.bootstrap_replicates != 1000:
        parser.error("The frozen balanced run requires exactly 1,000 bootstrap replicates")
    if args.baseline_atol < 0 or args.baseline_rtol < 0:
        parser.error("Baseline tolerances must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    preregistration_path = output_dir / "selected_experts_preregistered.json"
    if not preregistration_path.is_file():
        raise RuntimeError(
            "The frozen preregistration is missing. Run "
            "scripts/preregister_balanced_causal_panel.py before masking."
        )
    preregistration = read_json(preregistration_path)
    source = load_controlled_source(args.source_dir)
    _validate_preregistration(preregistration, source.input_fingerprint)
    panel = intervention_panel(preregistration)
    run_config = _run_config(args, source, preregistration, panel)
    run_config = _save_or_validate_run_config(output_dir, run_config)
    source_baselines, source_masking = _load_source_baselines(source.root, source)
    regression = _validate_regression_anchors(source.root, source_baselines, source)
    source_input_hashes = _source_controlled_input_hashes(source)

    if args.analysis_only:
        integrity = read_json(output_dir / "integrity_validation.json")
        if not integrity.get("passed"):
            raise RuntimeError("A completed full integrity validation is required")
        baselines = _load_output_baselines(output_dir, run_config["inference_fingerprint"])
        masked, provenance = _load_completed_interventions(
            output_dir,
            run_config["inference_fingerprint"],
            panel,
            source,
            source_input_hashes,
        )
        return _finish_analysis(
            output_dir,
            run_config,
            preregistration,
            integrity,
            regression,
            baselines,
            masked,
            provenance,
            args.skip_plots,
        )

    set_reproducible_seed(args.seed, deterministic=True)
    runtime = resolve_runtime(args.device, args.dtype)
    if "A40" not in runtime.description.upper():
        raise RuntimeError(
            f"The frozen experiment requires an NVIDIA A40; found {runtime.description}"
        )
    print(
        f"Loading {args.model}@{args.model_revision} on {runtime.description} as BF16...",
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
    runtime_architecture = architecture_metadata(bundle.model, layer_specs)
    _validate_runtime_model(bundle, runtime_architecture, source.architecture)
    prefix_ids = _encode_prefix(bundle.tokenizer, EXPECTED_PREFIX)
    if tuple(prefix_ids) != EXPECTED_PREFIX_IDS:
        raise RuntimeError(
            f"Neutral prefix tokenization changed: expected {EXPECTED_PREFIX_IDS}, "
            f"found {tuple(prefix_ids)}"
        )
    print("Running reversible-mask smoke validation and hook audit...", flush=True)
    smoke = validate_masking_mechanism(
        bundle, layer_specs[0], source.prepared[BALANCED_DOMAINS[0]]
    )
    if not smoke.get("passed") or smoke.get("hooks_before") != smoke.get("hooks_after"):
        raise RuntimeError("Masking smoke validation or hook cleanup failed")

    print("Reproducing all four source baselines before any panel masking...", flush=True)
    baselines, baseline_checks = _reproduce_baselines(
        bundle,
        source,
        source_baselines,
        output_dir,
        run_config["inference_fingerprint"],
        args.batch_size,
        args.baseline_atol,
        args.baseline_rtol,
        args.resume,
    )
    print("Reproducing baseline routing tensors before panel masking...", flush=True)
    routing_reproduction = _reproduce_routing_tensors(
        bundle, layer_specs, source, args.batch_size
    )
    current_versions = package_versions()
    reusable_environment = current_versions == source.config.get("package_versions")
    source_baselines_exact = all(row["bitwise_equal"] for row in baseline_checks.values())
    reuse_allowed = bool(
        args.reuse_existing_interventions
        and reusable_environment
        and source_baselines_exact
    )
    integrity: dict[str, Any] = {
        "passed": False,
        "pre_masking_integrity_passed": True,
        "completed_at_utc": None,
        "collection_fingerprint": source.config["collection_fingerprint"],
        "selection_input_fingerprint": source.input_fingerprint,
        "preregistration_fingerprint": preregistration[
            "preregistration_fingerprint"
        ],
        "source_static_audit_passed": source.audit["passed"],
        "model_revision_exact": bundle.resolved_revision == EXPECTED_MODEL_REVISION,
        "runtime": {
            "device": str(runtime.device),
            "description": runtime.description,
            "dtype": str(runtime.dtype).replace("torch.", ""),
            "batch_size": args.batch_size,
            "package_versions": current_versions,
            "matches_source_package_versions": reusable_environment,
        },
        "architecture_exact": True,
        "neutral_prefix_token_ids_exact": True,
        "baseline_reproduction_passed": all(
            row["within_tolerance"] for row in baseline_checks.values()
        ),
        "baseline_reproduction": baseline_checks,
        "routing_tensor_reproduction_passed": True,
        "routing_tensor_reproduction": routing_reproduction,
        "source_baselines_bitwise_equal": source_baselines_exact,
        "masking_smoke_validation": smoke,
        "hook_checks_passed": True,
        "routing_match_passed": False,
        "reuse_existing_interventions_requested": args.reuse_existing_interventions,
        "reuse_existing_interventions_allowed": reuse_allowed,
        "reuse_requirements": {
            "source_package_versions_exact": reusable_environment,
            "fresh_baselines_bitwise_equal_to_source": source_baselines_exact,
            "source_masking_method": source_masking["method"],
            "source_masking_scope": source_masking["scope"],
        },
        "regression_anchors": regression,
    }
    atomic_write_json(output_dir / "integrity_validation.json", integrity)
    print(
        "Pre-masking integrity gate passed. The frozen panel will now be evaluated; "
        "failed outcomes will not be replaced.",
        flush=True,
    )

    masked, provenance = _run_interventions(
        bundle,
        layer_specs,
        source,
        panel,
        output_dir,
        run_config["inference_fingerprint"],
        args.batch_size,
        args.resume,
        reuse_allowed,
        source_masking,
        source_input_hashes,
    )
    integrity["routing_match_passed"] = True
    integrity["hook_checks_passed"] = all(
        item.get("hooks_before") == item.get("hooks_after")
        for item in provenance.values()
        if item["source"] == "new_inference"
    ) and smoke["hooks_before"] == smoke["hooks_after"]
    if not integrity["hook_checks_passed"]:
        raise RuntimeError("A hook-cleanup diagnostic failed")
    integrity["passed"] = True
    integrity["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output_dir / "integrity_validation.json", integrity)
    return _finish_analysis(
        output_dir,
        run_config,
        preregistration,
        integrity,
        regression,
        baselines,
        masked,
        provenance,
        args.skip_plots,
    )


def _run_config(
    args: argparse.Namespace,
    source: Any,
    preregistration: Mapping[str, Any],
    panel: list[dict[str, Any]],
) -> dict[str, Any]:
    inference_basis = {
        "model": args.model,
        "model_revision": args.model_revision,
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "selection_input_fingerprint": source.input_fingerprint,
        "preregistration_fingerprint": preregistration[
            "preregistration_fingerprint"
        ],
        "method": MASKING_METHOD,
        "scope": MASKING_SCOPE,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "interventions": [
            {
                "intervention_id": row["intervention_id"],
                "pair_id": row["pair_id"],
                "role": row["role"],
                "target_domain": row["target_domain"],
                "layer": row["layer"],
                "expert_id": row["expert_id"],
            }
            for row in panel
        ],
    }
    return {
        **inference_basis,
        "inference_fingerprint": canonical_sha256(inference_basis),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "resume": args.resume,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_or_validate_run_config(
    output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    path = output_dir / "run_config.json"
    if path.exists():
        existing = read_json(path)
        if existing.get("inference_fingerprint") != config["inference_fingerprint"]:
            raise RuntimeError("Existing balanced artifacts use a different run configuration")
        return existing
    atomic_write_json(path, config)
    return config


def _validate_preregistration(
    preregistration: Mapping[str, Any], input_fingerprint: str
) -> None:
    if preregistration.get("status") != "FROZEN_BEFORE_MASKING":
        raise RuntimeError("Preregistration does not declare a pre-masking frozen state")
    if preregistration.get("masking_outcomes_used_for_selection") is not False:
        raise RuntimeError("Preregistration does not exclude masking outcomes")
    if preregistration.get("selection_input_fingerprint") != input_fingerprint:
        raise RuntimeError("Preregistration input fingerprint no longer matches the source")
    deterministic = {
        "source_collection_fingerprint": preregistration["source"][
            "collection_fingerprint"
        ],
        "selection_input_fingerprint": preregistration["selection_input_fingerprint"],
        "selection_algorithm_version": preregistration[
            "selection_algorithm_version"
        ],
        "control_algorithm_version": preregistration["control_algorithm_version"],
        "selection_algorithm": preregistration["selection_algorithm"],
        "domain_selection_tiers": preregistration["domain_selection_tiers"],
        "control_matching_tiers": preregistration["control_matching_tiers"],
        "ranked_candidate_pool": preregistration["ranked_candidate_pool"],
        "selected_experts": preregistration["selected_experts"],
        "matched_controls": preregistration["matched_controls"],
        "analysis_preregistration": preregistration["analysis_preregistration"],
    }
    observed = canonical_sha256(deterministic)
    if observed != preregistration.get("preregistration_fingerprint"):
        raise RuntimeError("Frozen preregistration content or fingerprint was modified")
    selected = preregistration["selected_experts"]
    controls = preregistration["matched_controls"]
    if len(selected) != 12 or len(controls) != 12:
        raise RuntimeError("Frozen panel must contain 12 specialists and 12 controls")
    for domain in BALANCED_DOMAINS:
        if sum(row["target_domain"] == domain for row in selected) != 3:
            raise RuntimeError(f"Frozen panel is not balanced for {domain}")
    if any(pair["fallback_used"] for pair in controls):
        raise RuntimeError("The frozen run unexpectedly contains a relaxed control match")


def _load_source_baselines(
    source_dir: Path, source: Any
) -> tuple[dict[str, LossStatistics], dict[str, Any]]:
    config_path = source_dir / "masking" / "masking_config.json"
    config = read_json(config_path)
    if config.get("collection_fingerprint") != source.config["collection_fingerprint"]:
        raise RuntimeError("Source baseline fingerprint does not match the collection")
    if config.get("method") != MASKING_METHOD or config.get("scope") != MASKING_SCOPE:
        raise RuntimeError("Source masking semantics differ from the frozen intervention")
    masking_fingerprint = config.get("masking_fingerprint")
    baselines: dict[str, LossStatistics] = {}
    for domain in BALANCED_DOMAINS:
        path = source_dir / "masking" / "baseline" / f"{domain}.npz"
        metadata_path = path.with_suffix(".metadata.json")
        metadata = read_json(metadata_path)
        if metadata.get("masking_fingerprint") != masking_fingerprint:
            raise RuntimeError(f"Source baseline metadata mismatch for {domain}")
        result = LossStatistics.load(path)
        if len(result.loss_sums) != 100 or not np.all(result.token_counts == 64):
            raise RuntimeError(f"Source baseline artifact is invalid for {domain}")
        baselines[domain] = result
    return baselines, config


def _validate_regression_anchors(
    source_dir: Path,
    baselines: Mapping[str, LossStatistics],
    source: Any,
) -> dict[str, Any]:
    anchors = [
        (11, 27, "coding", "reasoning", 0.0342, 0.0355),
        (10, 56, "coding", "general", 0.0288, 0.0312),
        (1, 25, "coding", "general", 0.0222, 0.0307),
    ]
    rows = []
    for layer, expert_id, high_domain, low_domain, expected_delta, expected_contrast in anchors:
        values = {}
        for domain in BALANCED_DOMAINS:
            path = (
                source_dir
                / "masking"
                / f"layer_{layer}_expert_{expert_id}"
                / f"{domain}.npz"
            )
            result = LossStatistics.load(path)
            expected_routes = source.statistics[domain].routing_counts[
                :, layer, expert_id
            ]
            if result.route_counts is None or not np.array_equal(
                result.route_counts, expected_routes
            ):
                raise RuntimeError(
                    f"Regression-anchor routes mismatch for L{layer}/E{expert_id}/{domain}"
                )
            values[domain] = result.per_token_nll - baselines[domain].per_token_nll
        high_delta = float(values[high_domain].mean())
        contrast = high_delta - float(values[low_domain].mean())
        passed = abs(high_delta - expected_delta) <= 0.002 and abs(
            contrast - expected_contrast
        ) <= 0.002
        if not passed:
            raise RuntimeError(
                f"Regression anchor L{layer}/E{expert_id} no longer matches validated values"
            )
        rows.append(
            {
                "layer": layer,
                "expert_id": expert_id,
                "high_domain": high_domain,
                "low_domain": low_domain,
                "observed_high_delta_nll": high_delta,
                "expected_high_delta_nll_approx": expected_delta,
                "observed_high_minus_low": contrast,
                "expected_high_minus_low_approx": expected_contrast,
                "tolerance": 0.002,
                "passed": True,
            }
        )
    return {"passed": True, "anchors": rows}


def _validate_runtime_model(
    bundle: Any, runtime: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    if bundle.resolved_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            f"Resolved model revision mismatch: {bundle.resolved_revision!r}"
        )
    keys = ("model_class", "config_model_type", "num_moe_layers", "num_experts", "top_k")
    for key in keys:
        if runtime.get(key) != source.get(key):
            raise RuntimeError(f"Runtime architecture mismatch for {key}")
    runtime_layers = [
        (row["model_layer_index"], row["num_experts"], row["top_k"])
        for row in runtime["layers"]
    ]
    source_layers = [
        (row["model_layer_index"], row["num_experts"], row["top_k"])
        for row in source["layers"]
    ]
    if runtime_layers != source_layers:
        raise RuntimeError("Runtime layer specifications differ from the source run")


def _encode_prefix(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _reproduce_baselines(
    bundle: Any,
    source: Any,
    source_baselines: Mapping[str, LossStatistics],
    output_dir: Path,
    fingerprint: str,
    batch_size: int,
    atol: float,
    rtol: float,
    resume: bool,
) -> tuple[dict[str, LossStatistics], dict[str, Any]]:
    baselines: dict[str, LossStatistics] = {}
    checks: dict[str, Any] = {}
    for domain in BALANCED_DOMAINS:
        path = output_dir / "masking" / "baseline" / f"{domain}.npz"
        metadata_path = path.with_suffix(".metadata.json")
        if resume and path.is_file() and metadata_path.is_file():
            metadata = read_json(metadata_path)
            if metadata.get("inference_fingerprint") != fingerprint:
                raise RuntimeError(f"Balanced baseline fingerprint mismatch for {domain}")
            fresh = LossStatistics.load(path)
            diagnostics = metadata.get("diagnostics", {})
            print(f"[{domain}] resume: reproduced baseline", flush=True)
        else:
            fresh, diagnostics = evaluate_next_token_loss(
                bundle, source.prepared[domain], batch_size=batch_size
            )
            fresh.save(path)
            atomic_write_json(
                metadata_path,
                {
                    "inference_fingerprint": fingerprint,
                    "source": "fresh_baseline_reproduction",
                    "domain": domain,
                    "diagnostics": diagnostics,
                },
            )
        reference = source_baselines[domain]
        if not np.array_equal(fresh.token_counts, reference.token_counts):
            raise RuntimeError(f"Baseline token counts failed reproduction for {domain}")
        differences = np.abs(fresh.loss_sums - reference.loss_sums)
        within = bool(
            np.allclose(fresh.loss_sums, reference.loss_sums, atol=atol, rtol=rtol)
        )
        if not within:
            raise RuntimeError(
                f"Baseline loss reproduction failed for {domain}: max absolute loss-sum "
                f"difference {differences.max():.9g}"
            )
        checks[domain] = {
            "within_tolerance": True,
            "bitwise_equal": bool(np.array_equal(fresh.loss_sums, reference.loss_sums)),
            "max_absolute_loss_sum_difference": float(differences.max()),
            "mean_absolute_loss_sum_difference": float(differences.mean()),
            "fresh_mean_nll": float(fresh.per_token_nll.mean()),
            "source_mean_nll": float(reference.per_token_nll.mean()),
            "atol": atol,
            "rtol": rtol,
            "hooks_before": diagnostics.get("hooks_before"),
            "hooks_after": diagnostics.get("hooks_after"),
        }
        if diagnostics.get("hooks_before") != diagnostics.get("hooks_after"):
            raise RuntimeError(f"Baseline evaluation leaked hooks for {domain}")
        baselines[domain] = fresh
    return baselines, checks


def _reproduce_routing_tensors(
    bundle: Any,
    layer_specs: list[Any],
    source: Any,
    batch_size: int,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for domain in BALANCED_DOMAINS:
        print(f"[{domain}] fresh routing-tensor reproduction...", flush=True)
        fresh = collect_prepared_domain(
            bundle,
            layer_specs,
            source.prepared[domain],
            batch_size=batch_size,
            compute_gradient_attribution=False,
        )
        reference = source.statistics[domain]
        routing_exact = np.array_equal(
            fresh.statistics.routing_counts, reference.routing_counts
        )
        token_counts_exact = np.array_equal(
            fresh.statistics.token_counts, reference.token_counts
        )
        if not routing_exact or not token_counts_exact:
            mismatches = int(
                np.count_nonzero(
                    fresh.statistics.routing_counts != reference.routing_counts
                )
            )
            raise RuntimeError(
                f"Baseline routing-tensor reproduction failed for {domain}: "
                f"{mismatches} differing cells"
            )
        checks[domain] = {
            "routing_counts_bitwise_equal": True,
            "token_counts_bitwise_equal": True,
            "routing_counts_sha256": _array_sha256(
                fresh.statistics.routing_counts
            ),
            "elapsed_seconds": fresh.metadata["elapsed_seconds"],
            "hook_diagnostics_present": bool(fresh.diagnostics),
        }
    return checks


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _run_interventions(
    bundle: Any,
    layer_specs: list[Any],
    source: Any,
    panel: list[dict[str, Any]],
    output_dir: Path,
    fingerprint: str,
    batch_size: int,
    resume: bool,
    reuse_allowed: bool,
    source_masking: Mapping[str, Any],
    source_input_hashes: Mapping[str, str],
) -> tuple[
    dict[tuple[int, int, str], LossStatistics],
    dict[tuple[int, int], dict[str, Any]],
]:
    specs = {spec.model_layer_index: spec for spec in layer_specs}
    source_targets = {
        (row["layer"], row["expert_id"]) for row in source_masking["targets"]
    }
    masked: dict[tuple[int, int, str], LossStatistics] = {}
    provenance: dict[tuple[int, int], dict[str, Any]] = {}
    completed: list[str] = []
    for intervention_index, intervention in enumerate(panel, 1):
        layer = int(intervention["layer"])
        expert_id = int(intervention["expert_id"])
        identity = (layer, expert_id)
        can_reuse = reuse_allowed and identity in source_targets
        diagnostics_by_domain: dict[str, Any] = {}
        sources = set()
        print(
            f"[{intervention_index}/{len(panel)}] {intervention['role']} "
            f"{intervention['target_domain']} L{layer}/E{expert_id}",
            flush=True,
        )
        for domain in BALANCED_DOMAINS:
            result, metadata = _load_or_run_intervention_domain(
                bundle=bundle,
                spec=specs[layer],
                expert_id=expert_id,
                domain=domain,
                examples=source.prepared[domain],
                expected_routes=source.statistics[domain].routing_counts[
                    :, specs[layer].ordinal, expert_id
                ],
                output_dir=output_dir,
                fingerprint=fingerprint,
                batch_size=batch_size,
                resume=resume,
                reuse_source_dir=source.root if can_reuse else None,
                source_masking_fingerprint=source_masking["masking_fingerprint"],
                controlled_input_sha256=source_input_hashes[domain],
            )
            masked[(layer, expert_id, domain)] = result
            diagnostics_by_domain[domain] = metadata.get("diagnostics", {})
            sources.add(metadata["source"])
        source_label = (
            "reused_validated_source"
            if sources == {"reused_validated_source"}
            else "new_inference"
        )
        hooks_before = [
            value.get("hooks_before")
            for value in diagnostics_by_domain.values()
            if value.get("hooks_before") is not None
        ]
        hooks_after = [
            value.get("hooks_after")
            for value in diagnostics_by_domain.values()
            if value.get("hooks_after") is not None
        ]
        provenance[identity] = {
            "source": source_label,
            "domain_sources": sorted(sources),
            "hooks_before": hooks_before[0] if hooks_before else 0,
            "hooks_after": hooks_after[-1] if hooks_after else 0,
            "diagnostics_by_domain": diagnostics_by_domain,
        }
        completed.append(intervention["intervention_id"])
        atomic_write_json(
            output_dir / "progress.json",
            {
                "inference_fingerprint": fingerprint,
                "completed_interventions": completed,
                "completed_count": len(completed),
                "total_interventions": len(panel),
                "last_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    return masked, provenance


def _load_or_run_intervention_domain(
    bundle: Any,
    spec: Any,
    expert_id: int,
    domain: str,
    examples: Any,
    expected_routes: np.ndarray,
    output_dir: Path,
    fingerprint: str,
    batch_size: int,
    resume: bool,
    reuse_source_dir: Path | None,
    source_masking_fingerprint: str,
    controlled_input_sha256: str,
) -> tuple[LossStatistics, dict[str, Any]]:
    directory = output_dir / "masking" / f"layer_{spec.model_layer_index}_expert_{expert_id}"
    path = directory / f"{domain}.npz"
    metadata_path = path.with_suffix(".metadata.json")
    if resume and path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("inference_fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint fingerprint mismatch for {path}")
        if metadata.get("controlled_input_sha256") != controlled_input_sha256:
            raise RuntimeError(f"Controlled-input fingerprint mismatch for {path}")
        result = LossStatistics.load(path)
        print(f"[{domain}] resume: L{spec.model_layer_index}/E{expert_id}", flush=True)
    elif reuse_source_dir is not None:
        source_path = (
            reuse_source_dir
            / "masking"
            / f"layer_{spec.model_layer_index}_expert_{expert_id}"
            / f"{domain}.npz"
        )
        source_metadata_path = source_path.with_suffix(".metadata.json")
        source_metadata = read_json(source_metadata_path)
        if source_metadata.get("masking_fingerprint") != source_masking_fingerprint:
            raise RuntimeError(f"Reusable source metadata mismatch for {source_path}")
        result = LossStatistics.load(source_path)
        metadata = {
            "inference_fingerprint": fingerprint,
            "source": "reused_validated_source",
            "source_path": str(source_path),
            "source_file_sha256": file_sha256(source_path),
            "source_metadata_sha256": file_sha256(source_metadata_path),
            "source_masking_fingerprint": source_masking_fingerprint,
            "domain": domain,
            "controlled_input_sha256": controlled_input_sha256,
            "layer": spec.model_layer_index,
            "expert_id": expert_id,
            "diagnostics": source_metadata.get("diagnostics", {}),
        }
        result.save(path)
        atomic_write_json(metadata_path, metadata)
        print(f"[{domain}] reused validated L{spec.model_layer_index}/E{expert_id}", flush=True)
    else:
        result, diagnostics = evaluate_next_token_loss(
            bundle,
            examples,
            batch_size=batch_size,
            mask_spec=spec,
            expert_id=expert_id,
        )
        metadata = {
            "inference_fingerprint": fingerprint,
            "source": "new_inference",
            "domain": domain,
            "controlled_input_sha256": controlled_input_sha256,
            "layer": spec.model_layer_index,
            "expert_id": expert_id,
            "diagnostics": diagnostics,
        }
        result.save(path)
        atomic_write_json(metadata_path, metadata)
    if result.route_counts is None or not np.array_equal(
        result.route_counts, expected_routes
    ):
        raise RuntimeError(
            f"Masked route counts do not match baseline routing for "
            f"L{spec.model_layer_index}/E{expert_id}/{domain}"
        )
    if result.zeroed_gate_mass is None or not np.all(np.isfinite(result.zeroed_gate_mass)):
        raise RuntimeError(
            f"Missing or invalid zeroed gate mass for L{spec.model_layer_index}/E"
            f"{expert_id}/{domain}"
        )
    diagnostics = metadata.get("diagnostics", {})
    if diagnostics.get("hooks_before") != diagnostics.get("hooks_after"):
        raise RuntimeError(
            f"Hook cleanup failed for L{spec.model_layer_index}/E{expert_id}/{domain}"
        )
    return result, metadata


def _load_output_baselines(output_dir: Path, fingerprint: str) -> dict[str, LossStatistics]:
    output = {}
    for domain in BALANCED_DOMAINS:
        path = output_dir / "masking" / "baseline" / f"{domain}.npz"
        metadata = read_json(path.with_suffix(".metadata.json"))
        if metadata.get("inference_fingerprint") != fingerprint:
            raise RuntimeError(f"Balanced baseline fingerprint mismatch for {domain}")
        output[domain] = LossStatistics.load(path)
    return output


def _load_completed_interventions(
    output_dir: Path,
    fingerprint: str,
    panel: list[dict[str, Any]],
    source: Any,
    source_input_hashes: Mapping[str, str],
) -> tuple[
    dict[tuple[int, int, str], LossStatistics],
    dict[tuple[int, int], dict[str, Any]],
]:
    masked = {}
    provenance = {}
    for intervention in panel:
        layer = intervention["layer"]
        expert_id = intervention["expert_id"]
        identity = (layer, expert_id)
        diagnostics = {}
        sources = set()
        for domain in BALANCED_DOMAINS:
            path = (
                output_dir
                / "masking"
                / f"layer_{layer}_expert_{expert_id}"
                / f"{domain}.npz"
            )
            metadata = read_json(path.with_suffix(".metadata.json"))
            if metadata.get("inference_fingerprint") != fingerprint:
                raise RuntimeError(f"Checkpoint fingerprint mismatch for {path}")
            if metadata.get("controlled_input_sha256") != source_input_hashes[domain]:
                raise RuntimeError(f"Controlled-input fingerprint mismatch for {path}")
            result = LossStatistics.load(path)
            expected = source.statistics[domain].routing_counts[:, layer, expert_id]
            if result.route_counts is None or not np.array_equal(result.route_counts, expected):
                raise RuntimeError(f"Routing checkpoint mismatch for {path}")
            masked[(layer, expert_id, domain)] = result
            diagnostics[domain] = metadata.get("diagnostics", {})
            sources.add(metadata["source"])
        source_label = (
            "reused_validated_source"
            if sources == {"reused_validated_source"}
            else "new_inference"
        )
        provenance[identity] = {
            "source": source_label,
            "domain_sources": sorted(sources),
            "hooks_before": 0,
            "hooks_after": 0,
            "diagnostics_by_domain": diagnostics,
        }
    return masked, provenance


def _source_controlled_input_hashes(source: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for domain in BALANCED_DOMAINS:
        key = f"controlled_inputs/{domain}.npz"
        observed = file_sha256(source.root / key)
        expected = source.file_sha256[key]
        if observed != expected:
            raise RuntimeError(f"Controlled input changed after source audit for {domain}")
        hashes[domain] = observed
    return hashes


def _finish_analysis(
    output_dir: Path,
    run_config: dict[str, Any],
    preregistration: dict[str, Any],
    integrity: dict[str, Any],
    regression: dict[str, Any],
    baselines: Mapping[str, LossStatistics],
    masked: Mapping[tuple[int, int, str], LossStatistics],
    provenance: Mapping[tuple[int, int], Mapping[str, Any]],
    skip_plots: bool,
) -> int:
    balanced_analysis, arrays = analyze_balanced_interventions(
        preregistration,
        baselines,
        masked,
        provenance,
        bootstrap_replicates=1000,
        seed=42,
    )
    write_balanced_outputs(balanced_analysis, arrays, output_dir)
    results: dict[str, Any] = {
        "schema_version": 1,
        "run_config": run_config,
        "preregistration": preregistration,
        "integrity_validation": integrity,
        "regression_anchors": regression,
        "balanced_analysis": balanced_analysis,
        "artifact_manifest": {},
    }
    figure_paths = [] if skip_plots else create_balanced_figures(results, output_dir)
    results["artifact_manifest"]["figures"] = [
        str(path.relative_to(output_dir)) for path in figure_paths
    ]
    write_balanced_summary(results, output_dir / "SUMMARY.md")
    atomic_write_json(output_dir / "results.json", results)
    audit = _audit_generated_outputs(output_dir, len(figure_paths))
    results["artifact_manifest"].update(audit)
    atomic_write_json(output_dir / "results.json", results)
    write_balanced_summary(results, output_dir / "SUMMARY.md")
    print(f"Balanced causal validation complete: {output_dir}", flush=True)
    print(f"Decision: {balanced_analysis['decision']['label']}", flush=True)
    print(f"Summary: {output_dir / 'SUMMARY.md'}", flush=True)
    return 0


def _audit_generated_outputs(output_dir: Path, figure_count: int) -> dict[str, Any]:
    required = [
        "selected_experts_preregistered.json",
        "candidate_experts.csv",
        "matched_controls.csv",
        "masking_results.csv",
        "pairwise_domain_contrasts.csv",
        "specialized_vs_control.csv",
        "aggregate_results.csv",
        "per_example_loss_changes.npz",
        "results.json",
        "SUMMARY.md",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("Generated output audit found missing files: " + ", ".join(missing))
    with np.load(output_dir / "per_example_loss_changes.npz", allow_pickle=False) as data:
        if data["per_example_loss_changes"].shape != (24, 4, 100):
            raise RuntimeError("Per-example loss-change array has the wrong shape")
        for key in (
            "baseline_per_example_nll",
            "masked_per_example_nll",
            "per_example_loss_changes",
        ):
            if not np.all(np.isfinite(data[key])):
                raise RuntimeError(f"Generated {key} contains non-finite values")
    if figure_count not in (0, 10):
        raise RuntimeError(f"Expected 10 figure files, found {figure_count}")
    hashable = [name for name in required if name not in ("results.json", "SUMMARY.md")]
    paths = [output_dir / name for name in hashable]
    paths.extend(sorted((output_dir / "figures").glob("figure_*")))
    return {
        "output_audit_passed": True,
        "required_files": required,
        "file_sha256": {
            str(path.relative_to(output_dir)): file_sha256(path)
            for path in paths
            if path.is_file()
        },
        "per_example_loss_changes_shape": [24, 4, 100],
        "figure_file_count": figure_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
