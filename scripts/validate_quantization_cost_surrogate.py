#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from expert_analysis.activation_quantization_cost import (
    ACTIVATION_COST_EPSILON,
    evaluate_activation_surrogates_for_panel,
)
from expert_analysis.balanced import (
    BALANCED_DOMAINS,
    EXPECTED_EXAMPLES,
    EXPECTED_EXPERTS,
    EXPECTED_MODEL,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MOE_LAYERS,
    EXPECTED_TOP_K,
    canonical_sha256,
    file_sha256,
    load_controlled_source,
)
from expert_analysis.expert_replay import (
    DEFAULT_REPLAY_ATOL,
    DEFAULT_REPLAY_RTOL,
    capture_replay_dataset,
    validate_replay_captures,
)
from expert_analysis.gradient_quantization_cost import (
    capture_gradient_dataset,
    evaluate_gradient_surrogate_for_panel,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions, read_json
from expert_analysis.modeling import (
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)
from expert_analysis.quantization import ExpertWeightLayout
from expert_analysis.quantization_pilot import pilot_intervention_panel
from expert_analysis.surrogate_validation import (
    analyze_fixed_surrogates,
    create_surrogate_figures,
    decision_from_analyses,
    load_stage1_qdq_fingerprints,
    load_stage1_validation_data,
    write_pilot_raw_npz,
    write_surrogate_summary,
    write_validation_tables,
)


SCRIPT_VERSION = "stage2a_surrogate_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fixed activation-aware OLMoE quantization-cost surrogates. "
            "This command does not build a mixed-precision optimizer."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("results/expert_quantization_pilot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/quantization_cost_surrogate"),
    )
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--primary-bits", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-chunk-size", type=int, default=512)
    parser.add_argument("--replay-validation-layers", type=int, default=4)
    parser.add_argument("--replay-validation-samples-per-layer", type=int, default=3)
    parser.add_argument("--replay-atol", type=float, default=DEFAULT_REPLAY_ATOL)
    parser.add_argument("--replay-rtol", type=float, default=DEFAULT_REPLAY_RTOL)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen artifacts and print the planned run without loading OLMoE.",
    )
    args = parser.parse_args()
    if args.model != EXPECTED_MODEL or args.model_revision != EXPECTED_MODEL_REVISION:
        parser.error("Stage 2A is pinned to the validated OLMoE checkpoint/revision")
    if args.device != "cuda":
        parser.error("The real Stage-2A validation requires --device cuda")
    if args.dtype.lower() not in ("bfloat16", "bf16"):
        parser.error("The real Stage-2A validation requires BF16")
    if args.batch_size != 1:
        parser.error("The frozen validation requires batch size 1")
    if args.group_size != 128 or args.primary_bits != 4:
        parser.error("Stage 2A must reuse Stage-1 4-bit/group-size-128 QDQ")
    if args.bootstrap_replicates != 1000 or args.seed != 42:
        parser.error("Stage 2A requires 1,000 grouped replicates and seed 42")
    if args.replay_chunk_size < 1:
        parser.error("--replay-chunk-size must be positive")
    if not 3 <= args.replay_validation_layers <= EXPECTED_MOE_LAYERS:
        parser.error("--replay-validation-layers must select at least three layers")
    if args.replay_validation_samples_per_layer < 1:
        parser.error("Replay validation requires at least one sample per selected layer")
    if args.replay_atol < 0 or args.replay_rtol < 0:
        parser.error("Replay tolerances must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    source = load_controlled_source(args.source_dir.resolve())
    stage1 = load_stage1_validation_data(args.stage1_dir.resolve())
    if source.input_fingerprint != stage1.stage1_metadata["source_input_fingerprint"]:
        raise RuntimeError("Frozen Stage-1 inputs differ from the controlled source")
    qdq_fingerprints = load_stage1_qdq_fingerprints(args.stage1_dir.resolve())
    panel_payload = read_json(args.stage1_dir / "pilot_panel_preregistered.json")
    panel = pilot_intervention_panel(panel_payload)
    input_hashes = panel_payload["source"]["controlled_input_file_sha256"]
    for domain in BALANCED_DOMAINS:
        path = source.root / "controlled_inputs" / f"{domain}.npz"
        if input_hashes[domain] != file_sha256(path):
            raise RuntimeError(f"Stage-1 controlled input hash changed for {domain}")

    validation_layers = _validation_layers(args.seed, args.replay_validation_layers)
    capture_basis = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "model_revision": args.model_revision,
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "source_input_fingerprint": source.input_fingerprint,
        "stage1_panel_fingerprint": panel_payload["pilot_panel_fingerprint"],
        "controlled_input_file_sha256": input_hashes,
        "domains": list(BALANCED_DOMAINS),
        "examples_per_domain": EXPECTED_EXAMPLES,
        "measured_positions_per_example": 64,
        "batch_size": args.batch_size,
        "dtype": "bfloat16",
        "seed": args.seed,
        "validation_layers": validation_layers,
        "validation_samples_per_layer": args.replay_validation_samples_per_layer,
        "hidden_storage": "exact_native_dtype_bits",
    }
    capture_fingerprint = canonical_sha256(capture_basis)
    run_basis = {
        **capture_basis,
        "capture_fingerprint": capture_fingerprint,
        "quantization_method": (
            "Stage-1 deterministic symmetric group-wise expert-only QDQ"
        ),
        "primary_bits": args.primary_bits,
        "group_size": args.group_size,
        "scale_storage_dtype": "float16",
        "aod_epsilon": ACTIVATION_COST_EPSILON,
        "replay_chunk_size": args.replay_chunk_size,
        "replay_validation_atol": args.replay_atol,
        "replay_validation_rtol": args.replay_rtol,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "expert_with_four_domains_grouped",
        "no_fitting_rule": True,
        "gradient_fallback": "GQS_primary_GQS2_diagnostic",
    }
    run_fingerprint = canonical_sha256(run_basis)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "stage1_decision": "GO",
                    "stage1_observations": 64,
                    "stage1_qdq_fingerprints": len(qdq_fingerprints),
                    "source_input_fingerprint": source.input_fingerprint,
                    "capture_fingerprint": capture_fingerprint,
                    "run_fingerprint": run_fingerprint,
                    "validation_layers": validation_layers,
                    "cuda_available": torch.cuda.is_available(),
                    "execution_performed": False,
                },
                indent=2,
            )
        )
        return 0

    runtime = resolve_runtime(args.device, args.dtype)
    _require_a40(runtime)
    set_reproducible_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **run_basis,
        "run_fingerprint": run_fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "runtime_description": runtime.description,
        "device": str(runtime.device),
        "package_versions": package_versions(),
        "output_dir": str(output_dir),
        "optimizer_implemented": False,
    }
    _write_or_validate_config(output_dir / "run_config.json", run_config)

    bundle = load_model_and_tokenizer(
        args.model,
        runtime,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    specs = discover_moe_layers(bundle.model)
    architecture = architecture_metadata(bundle.model, specs)
    _validate_runtime(bundle, architecture)
    layouts = {spec.model_layer_index: ExpertWeightLayout(spec) for spec in specs}
    layout_metadata = _validate_layouts(layouts)
    capture_manifest = capture_replay_dataset(
        bundle,
        specs,
        source.prepared,
        output_dir / "capture",
        capture_fingerprint=capture_fingerprint,
        controlled_input_file_sha256=input_hashes,
        source_statistics=source.statistics,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_layer_indices=validation_layers,
        validation_samples_per_layer=args.replay_validation_samples_per_layer,
        resume=args.resume,
    )
    capture_metadata = {
        "passed": True,
        "run_fingerprint": run_fingerprint,
        "capture_fingerprint": capture_fingerprint,
        "capture_basis": capture_basis,
        "runtime": {
            "description": runtime.description,
            "device": str(runtime.device),
            "dtype": str(runtime.dtype).replace("torch.", ""),
        },
        "architecture": architecture,
        "expert_weight_layout": layout_metadata,
        "controlled_source_audit": source.audit,
        "capture_manifest": capture_manifest,
    }
    atomic_write_json(output_dir / "capture_metadata.json", capture_metadata)

    replay_validation = validate_replay_captures(
        specs,
        output_dir / "capture",
        BALANCED_DOMAINS,
        capture_fingerprint=capture_fingerprint,
        validation_layer_indices=validation_layers,
        atol=args.replay_atol,
        rtol=args.replay_rtol,
    )
    expected_replay_samples = (
        len(BALANCED_DOMAINS)
        * len(validation_layers)
        * args.replay_validation_samples_per_layer
    )
    replay_coverage_passed = bool(
        replay_validation["sample_count"] == expected_replay_samples
        and replay_validation["validated_layers"] == validation_layers
        and replay_validation["validated_expert_count"] >= 3
    )
    replay_validation["coverage_gate"] = {
        "passed": replay_coverage_passed,
        "expected_sample_count": expected_replay_samples,
        "required_unique_experts_at_least": 3,
        "required_layers": validation_layers,
    }
    replay_validation["passed"] = bool(
        replay_validation["passed"] and replay_coverage_passed
    )
    atomic_write_json(output_dir / "replay_validation.json", replay_validation)
    if replay_validation["passed"] is not True:
        raise RuntimeError("Critical expert replay validation failed; stopping before costs")

    activation_raw, qdq_validation = evaluate_activation_surrogates_for_panel(
        specs,
        layouts,
        output_dir / "capture",
        BALANCED_DOMAINS,
        panel,
        [4],
        capture_fingerprint=capture_fingerprint,
        group_size=args.group_size,
        chunk_size=args.replay_chunk_size,
        expected_qdq_fingerprints=qdq_fingerprints,
        verify_unrelated_experts=True,
    )
    if not all(row.get("exact_stage1_fingerprint_match") for row in qdq_validation):
        raise RuntimeError("Pilot QDQ did not reproduce every Stage-1 expert fingerprint")
    fixed_scores = {
        "weight_risk_functional": stage1.weight_risk_functional,
        "weight_risk_routing": stage1.weight_risk_routing,
        "functional_importance": stage1.functional_importance,
        "routing_importance": stage1.routing_importance,
        "uod": activation_raw["uod"][0],
        "reod": activation_raw["reod"][0],
        "apd": activation_raw["apd"][0],
        "aod": activation_raw["aod"][0],
    }
    aod_analysis = analyze_fixed_surrogates(
        stage1,
        {
            key: value
            for key, value in fixed_scores.items()
            if key
            not in (
                "weight_risk_functional",
                "weight_risk_routing",
                "functional_importance",
                "routing_importance",
            )
        },
        primary_name="aod",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    gradient_metadata: dict[str, Any] | None = None
    gradient_raw: dict[str, np.ndarray] | None = None
    gradient_qdq_validation: list[dict[str, Any]] | None = None
    gqs_analysis: dict[str, Any] | None = None
    final_scores = dict(fixed_scores)
    final_analysis = aod_analysis
    if not aod_analysis["primary_passed"]:
        gradient_basis = {
            "script_version": SCRIPT_VERSION,
            "run_fingerprint": run_fingerprint,
            "capture_fingerprint": capture_fingerprint,
            "loss": "per-example mean NLL over 64 frozen measured positions",
            "one_backward_per_example": True,
            "model_frozen": True,
            "seed": args.seed,
        }
        gradient_fingerprint = canonical_sha256(gradient_basis)
        gradient_metadata = capture_gradient_dataset(
            bundle,
            specs,
            source.prepared,
            output_dir / "capture",
            output_dir / "gradient_capture",
            capture_fingerprint=capture_fingerprint,
            gradient_fingerprint=gradient_fingerprint,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        gradient_metadata["gradient_basis"] = gradient_basis
        atomic_write_json(output_dir / "gradient_capture_metadata.json", gradient_metadata)
        gradient_raw, gradient_qdq_validation = evaluate_gradient_surrogate_for_panel(
            specs,
            layouts,
            output_dir / "capture",
            output_dir / "gradient_capture",
            BALANCED_DOMAINS,
            panel,
            [4],
            capture_fingerprint=capture_fingerprint,
            gradient_fingerprint=gradient_fingerprint,
            num_examples=EXPECTED_EXAMPLES,
            group_size=args.group_size,
            chunk_size=args.replay_chunk_size,
            expected_qdq_fingerprints=qdq_fingerprints,
        )
        final_scores["gqs"] = gradient_raw["gqs"][0]
        gqs_analysis = analyze_fixed_surrogates(
            stage1,
            {
                "uod": activation_raw["uod"][0],
                "reod": activation_raw["reod"][0],
                "apd": activation_raw["apd"][0],
                "aod": activation_raw["aod"][0],
                "gqs": gradient_raw["gqs"][0],
            },
            primary_name="gqs",
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        final_analysis = gqs_analysis

    decision = decision_from_analyses(aod_analysis, gqs_analysis, audit_passed=None)
    primary_internal = (
        "aod" if decision["provisional_metric_decision"] == "AOD_GO" else "gqs"
    )
    write_pilot_raw_npz(
        output_dir / "pilot_surrogate_raw.npz",
        stage1,
        activation_raw,
        gradient_raw=gradient_raw,
    )
    write_validation_tables(output_dir, stage1, final_scores, final_analysis)
    bootstrap_payload = {
        "aod": aod_analysis["bootstrap_summaries"],
        "aod_weight_proxy_improvement": aod_analysis[
            "weight_proxy_improvement_bootstrap"
        ],
        "gqs": gqs_analysis["bootstrap_summaries"] if gqs_analysis else None,
        "method": aod_analysis["bootstrap"],
    }
    atomic_write_json(output_dir / "bootstrap_results.json", bootstrap_payload)
    atomic_write_json(output_dir / "surrogate_decision.json", decision)
    figures = (
        []
        if args.skip_plots
        else create_surrogate_figures(
            output_dir,
            stage1,
            final_scores,
            final_analysis,
            primary_name=primary_internal,
        )
    )
    results = {
        "schema_version": 1,
        "run_config": run_config,
        "stage1_validation": stage1.stage1_metadata,
        "capture_metadata": capture_metadata,
        "replay_validation": replay_validation,
        "qdq_reproduction": {
            "passed": True,
            "validated_pilot_experts": 16,
            "exact_stage1_fingerprint_matches": 16,
            "rows": qdq_validation,
        },
        "aod_analysis": aod_analysis,
        "gradient_fallback": {
            "triggered": gradient_metadata is not None,
            "capture_metadata": gradient_metadata,
            "analysis": gqs_analysis,
            "qdq_reproduction": gradient_qdq_validation,
            "gqs_is_primary_if_triggered": True,
            "gqs2_is_diagnostic_only": True,
        },
        "surrogate_decision": decision,
        "full_cost_matrix": None,
        "mixed_precision_optimizer": {"implemented": False, "status": "not_in_scope"},
        "artifact_manifest": _artifact_manifest(output_dir, figures),
    }
    atomic_write_json(output_dir / "results.json", results)
    write_surrogate_summary(output_dir / "SUMMARY.md", results)

    audit_passed = _run_independent_audit(args, output_dir)
    decision = decision_from_analyses(
        aod_analysis, gqs_analysis, audit_passed=audit_passed
    )
    atomic_write_json(output_dir / "surrogate_decision.json", decision)
    results["surrogate_decision"] = decision
    results["independent_audit"] = read_json(output_dir / "independent_audit.json")
    results["artifact_manifest"] = _artifact_manifest(output_dir, figures)
    atomic_write_json(output_dir / "results.json", results)
    write_surrogate_summary(output_dir / "SUMMARY.md", results)
    print(json.dumps(decision, indent=2))
    return 0 if audit_passed else 2


def _validation_layers(seed: int, count: int) -> list[int]:
    digest = hashlib.sha256(
        f"olmoe-stage2a-replay-layer-selection-v1\0{seed}".encode("utf-8")
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16) % (2**63 - 1))
    return sorted(
        int(value)
        for value in rng.choice(EXPECTED_MOE_LAYERS, size=count, replace=False)
    )


def _require_a40(runtime: Any) -> None:
    if runtime.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real Stage-2A execution requires CUDA")
    if runtime.dtype != torch.bfloat16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real Stage-2A execution requires CUDA BF16 support")
    name = torch.cuda.get_device_name(runtime.device)
    if "A40" not in name.upper():
        raise RuntimeError(f"Stage-2A production is pinned to NVIDIA A40, found {name!r}")


def _validate_runtime(bundle: Any, architecture: Mapping[str, Any]) -> None:
    if bundle.resolved_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Resolved checkpoint revision differs from the frozen revision")
    if architecture.get("num_moe_layers") != EXPECTED_MOE_LAYERS:
        raise RuntimeError("Runtime model does not have 16 MoE layers")
    if architecture.get("num_experts") != EXPECTED_EXPERTS:
        raise RuntimeError("Runtime model does not have 64 experts/layer")
    if architecture.get("top_k") != [EXPECTED_TOP_K]:
        raise RuntimeError("Runtime model does not use top-8 routing")
    layers = architecture.get("layers", [])
    if [row.get("model_layer_index") for row in layers] != list(range(16)):
        raise RuntimeError("Runtime MoE layer indexing differs from 0..15")


def _validate_layouts(layouts: Mapping[int, ExpertWeightLayout]) -> dict[str, Any]:
    if set(layouts) != set(range(16)):
        raise RuntimeError("Expert layouts do not cover all 16 MoE layers")
    rows = {}
    for layer, layout in layouts.items():
        metadata = layout.metadata()
        shapes = {tuple(row["expert_slice_shape"]) for row in metadata["tensors"]}
        axes = {row["expert_axis"] for row in metadata["tensors"]}
        if metadata["num_experts"] != 64 or shapes != {(2048, 2048), (2048, 1024)}:
            raise RuntimeError(f"Unexpected expert layout at layer {layer}")
        if axes != {0}:
            raise RuntimeError(f"Tensorized expert axis is not zero at layer {layer}")
        rows[str(layer)] = metadata
    return {
        "passed": True,
        "tensorized_expert_axis": 0,
        "expert_slice_shapes": [[2048, 1024], [2048, 2048]],
        "layers": rows,
    }


def _write_or_validate_config(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_file():
        observed = read_json(path)
        if observed.get("run_fingerprint") != expected.get("run_fingerprint"):
            raise RuntimeError(
                "Existing Stage-2A output uses a different configuration; choose a "
                "different directory"
            )
        return
    atomic_write_json(path, dict(expected))


def _artifact_manifest(output_dir: Path, figures: list[Path]) -> dict[str, Any]:
    names = [
        "run_config.json",
        "capture_metadata.json",
        "replay_validation.json",
        "pilot_surrogate_raw.npz",
        "pilot_surrogate_values.csv",
        "surrogate_comparison.csv",
        "surrogate_specificity.csv",
        "within_expert_domain_rankings.csv",
        "domain_specific_correlations.csv",
        "bootstrap_results.json",
        "surrogate_decision.json",
        "SUMMARY.md",
    ]
    if (output_dir / "gradient_capture_metadata.json").is_file():
        names.append("gradient_capture_metadata.json")
    hashes = {
        name: file_sha256(output_dir / name)
        for name in names
        if (output_dir / name).is_file()
    }
    hashes.update(
        {
            str(path.relative_to(output_dir)): file_sha256(path)
            for path in figures
            if path.is_file()
        }
    )
    return {
        "required_small_outputs": names,
        "figure_files": [str(path.relative_to(output_dir)) for path in figures],
        "file_sha256": hashes,
        "large_capture_files_excluded_from_manifest": True,
        "optimizer_artifacts_present": False,
    }


def _run_independent_audit(args: argparse.Namespace, output_dir: Path) -> bool:
    script = Path(__file__).with_name("audit_quantization_cost_surrogate.py")
    command = [
        sys.executable,
        str(script),
        "--stage1-dir",
        str(args.stage1_dir.resolve()),
        "--surrogate-dir",
        str(output_dir),
        "--output",
        str(output_dir / "independent_audit.json"),
    ]
    completed = subprocess.run(command, check=False)
    if not (output_dir / "independent_audit.json").is_file():
        return False
    audit = read_json(output_dir / "independent_audit.json")
    return completed.returncode == 0 and audit.get("passed") is True


if __name__ == "__main__":
    raise SystemExit(main())
