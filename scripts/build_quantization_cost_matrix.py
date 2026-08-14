#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from expert_analysis.balanced import (
    BALANCED_DOMAINS,
    EXPECTED_EXPERTS,
    EXPECTED_MODEL,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MOE_LAYERS,
    EXPECTED_TOP_K,
    canonical_sha256,
    file_sha256,
)
from expert_analysis.cost_matrix import (
    DEFAULT_BIT_WIDTHS,
    build_full_cost_matrix,
    build_memory_matrix,
    create_full_cost_map_figures,
    verify_pilot_reproduction,
    write_full_matrix_outputs,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions, read_json
from expert_analysis.modeling import (
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)
from expert_analysis.quantization import ExpertWeightLayout
from expert_analysis.surrogate_validation import (
    load_stage1_validation_data,
    write_surrogate_summary,
)


SCRIPT_VERSION = "stage2a_full_cost_matrix_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full OLMoE expert × domain × precision cost matrix after an "
            "audited Stage-2A surrogate GO. No optimizer is implemented."
        )
    )
    parser.add_argument(
        "--surrogate-dir",
        type=Path,
        default=Path("results/quantization_cost_surrogate"),
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("results/expert_quantization_pilot"),
    )
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bit-widths", type=int, nargs=4, default=list(DEFAULT_BIT_WIDTHS))
    parser.add_argument("--replay-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.model != EXPECTED_MODEL or args.model_revision != EXPECTED_MODEL_REVISION:
        parser.error("The full matrix is pinned to the validated OLMoE revision")
    if args.device != "cuda" or args.dtype.lower() not in ("bfloat16", "bf16"):
        parser.error("The production full matrix requires CUDA/BF16")
    if args.batch_size != 1 or args.group_size != 128 or args.seed != 42:
        parser.error("The frozen experiment requires batch=1, group-size=128, seed=42")
    if tuple(args.bit_widths) != DEFAULT_BIT_WIDTHS:
        parser.error("The precision axis must be exactly: 3 4 8 16")
    if args.replay_chunk_size < 1:
        parser.error("--replay-chunk-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    root = args.surrogate_dir.resolve()
    decision_path = root / "surrogate_decision.json"
    audit_path = root / "independent_audit.json"
    if not decision_path.is_file() or not audit_path.is_file():
        raise RuntimeError("Full matrix requires a completed independently audited validation")
    decision = read_json(decision_path)
    audit = read_json(audit_path)
    if decision.get("decision") not in ("AOD_GO", "SURROGATE_GO_GRADIENT"):
        raise RuntimeError("No surrogate GO authorizes the full cost matrix")
    if decision.get("independent_audit_passed") is not True or audit.get("passed") is not True:
        raise RuntimeError("Independent pilot audit failure blocks the full cost matrix")
    selected = str(decision["selected_surrogate"])
    capture_metadata = read_json(root / "capture_metadata.json")
    run_config = read_json(root / "run_config.json")
    capture_fingerprint = str(capture_metadata["capture_fingerprint"])
    gradient_fingerprint = None
    if selected == "GQS":
        gradient_metadata = read_json(root / "gradient_capture_metadata.json")
        gradient_fingerprint = str(gradient_metadata["gradient_fingerprint"])

    basis = {
        "script_version": SCRIPT_VERSION,
        "surrogate_run_fingerprint": run_config["run_fingerprint"],
        "capture_fingerprint": capture_fingerprint,
        "gradient_fingerprint": gradient_fingerprint,
        "selected_surrogate": selected,
        "model": args.model,
        "model_revision": args.model_revision,
        "domains": list(BALANCED_DOMAINS),
        "bit_widths": list(DEFAULT_BIT_WIDTHS),
        "group_size": args.group_size,
        "device": "cuda",
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "replay_chunk_size": args.replay_chunk_size,
        "seed": args.seed,
        "posthoc_cost_renormalization": False,
    }
    matrix_fingerprint = canonical_sha256(basis)
    config = {
        **basis,
        "matrix_fingerprint": matrix_fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "package_versions": package_versions(),
        "optimizer_implemented": False,
    }
    _write_or_validate_config(root / "full_matrix_run_config.json", config)

    runtime = resolve_runtime(args.device, args.dtype)
    _require_a40(runtime)
    set_reproducible_seed(args.seed)
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

    matrix = build_full_cost_matrix(
        specs,
        layouts,
        root / "capture",
        root,
        BALANCED_DOMAINS,
        selected_surrogate=selected,
        matrix_fingerprint=matrix_fingerprint,
        capture_fingerprint=capture_fingerprint,
        gradient_capture_dir=(root / "gradient_capture" if selected == "GQS" else None),
        gradient_fingerprint=gradient_fingerprint,
        bit_widths=DEFAULT_BIT_WIDTHS,
        group_size=args.group_size,
        chunk_size=args.replay_chunk_size,
        num_examples=100,
        resume=args.resume,
    )
    memory = build_memory_matrix(
        specs, layouts, DEFAULT_BIT_WIDTHS, group_size=args.group_size
    )
    metadata = write_full_matrix_outputs(
        root,
        matrix,
        memory,
        selected_surrogate=selected,
        matrix_fingerprint=matrix_fingerprint,
        group_size=args.group_size,
    )
    stage1 = load_stage1_validation_data(args.stage1_dir.resolve())
    with np.load(root / "pilot_surrogate_raw.npz", allow_pickle=False) as data:
        pilot_values = (
            data["activation_aod"][0].astype(np.float64)
            if selected == "AOD"
            else data["gradient_gqs"][0].astype(np.float64)
        )
    reproduction = verify_pilot_reproduction(
        matrix["cost"],
        matrix["layer_indices"],
        matrix["expert_ids"],
        matrix["domain_names"],
        matrix["bit_widths"],
        stage1.layers,
        stage1.expert_ids,
        pilot_values,
        atol=1e-12,
    )
    if not reproduction["passed"]:
        raise RuntimeError("Full matrix failed to reproduce pilot 4-bit surrogate values")
    figures = (
        []
        if args.skip_plots
        else create_full_cost_map_figures(
            root,
            matrix["cost"],
            BALANCED_DOMAINS,
            DEFAULT_BIT_WIDTHS,
            bit_width=4,
        )
    )
    metadata["pilot_reproduction"] = reproduction
    metadata["qdq_manifest"] = matrix["qdq_manifest"]
    metadata["figures"] = [str(path.relative_to(root)) for path in figures]
    metadata["file_sha256"].update(
        {
            str(path.relative_to(root)): file_sha256(path)
            for path in figures
            if path.is_file()
        }
    )
    if selected == "GQS":
        metadata["raw_gqs_scale_by_layer_domain"] = {
            domain: {
                str(layer): {
                    "minimum": float(matrix["cost"][layer, :, domain_index, :3].min()),
                    "median": float(
                        np.median(matrix["cost"][layer, :, domain_index, :3])
                    ),
                    "maximum": float(matrix["cost"][layer, :, domain_index, :3].max()),
                }
                for layer in range(16)
            }
            for domain_index, domain in enumerate(BALANCED_DOMAINS)
        }
        metadata["gqs_normalization_investigation"] = {
            "performed_before_optimizer_outcomes": True,
            "raw_scale_reported": True,
            "normalization_applied": False,
            "conclusion": (
                "No deterministic layer normalization is imposed in Stage 2A; the raw "
                "pre-registered GQS scale is preserved for a separately designed optimizer."
            ),
        }
    atomic_write_json(root / "full_matrix_metadata.json", metadata)

    results = read_json(root / "results.json")
    results["full_cost_matrix"] = {
        "status": "complete_pending_final_independent_audit",
        "selected_surrogate": selected,
        "matrix_fingerprint": matrix_fingerprint,
        "metadata": metadata,
        "pilot_reproduction": reproduction,
    }
    results["mixed_precision_optimizer"] = {
        "implemented": False,
        "status": "pending_separate_authorization_and_design",
    }
    atomic_write_json(root / "results.json", results)
    write_surrogate_summary(root / "SUMMARY.md", results)
    _append_matrix_summary(root / "SUMMARY.md", metadata, reproduction)

    audit_passed = _run_audit(args, root)
    final_audit = read_json(root / "independent_audit.json")
    if not audit_passed:
        decision["decision"] = "SURROGATE_NO_GO"
        decision["selected_surrogate"] = None
        decision["independent_audit_passed"] = False
        decision["full_cost_matrix_authorized"] = False
        decision["rationale"] = "Full-matrix independent audit failure blocks GO."
        atomic_write_json(decision_path, decision)
        results["surrogate_decision"] = decision
        results["full_cost_matrix"]["status"] = "audit_failed"
        results["independent_audit"] = final_audit
        atomic_write_json(root / "results.json", results)
        return 2
    results["full_cost_matrix"]["status"] = "complete_and_independently_audited"
    results["independent_audit"] = final_audit
    atomic_write_json(root / "results.json", results)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "selected_surrogate": selected,
                "matrix_shape": list(matrix["cost"].shape),
                "bit_widths": list(DEFAULT_BIT_WIDTHS),
                "pilot_reproduction": reproduction,
                "independent_audit_passed": True,
                "optimizer_implemented": False,
            },
            indent=2,
        )
    )
    return 0


def _write_or_validate_config(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_file():
        observed = read_json(path)
        if observed.get("matrix_fingerprint") != expected.get("matrix_fingerprint"):
            raise RuntimeError("Existing full-matrix output uses another configuration")
        return
    atomic_write_json(path, dict(expected))


def _require_a40(runtime: Any) -> None:
    if runtime.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full cost matrix requires CUDA")
    if runtime.dtype != torch.bfloat16 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full cost matrix requires BF16")
    name = torch.cuda.get_device_name(runtime.device)
    if "A40" not in name.upper():
        raise RuntimeError(f"Full matrix is pinned to NVIDIA A40, found {name!r}")


def _validate_runtime(bundle: Any, architecture: Mapping[str, Any]) -> None:
    if bundle.resolved_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Resolved checkpoint revision differs")
    if architecture.get("num_moe_layers") != EXPECTED_MOE_LAYERS:
        raise RuntimeError("Runtime model does not contain 16 MoE layers")
    if architecture.get("num_experts") != EXPECTED_EXPERTS:
        raise RuntimeError("Runtime model does not contain 64 experts/layer")
    if architecture.get("top_k") != [EXPECTED_TOP_K]:
        raise RuntimeError("Runtime model does not use top-8 routing")


def _append_matrix_summary(
    path: Path, metadata: Mapping[str, Any], reproduction: Mapping[str, Any]
) -> None:
    text = path.read_text(encoding="utf-8")
    lines = [
        "## Full cost matrix",
        "",
        f"- Selected fixed cost: {metadata['selected_surrogate']}",
        f"- Shape: {metadata['shape']}",
        f"- Precision order: {metadata['bit_widths']}",
        f"- Unobserved expert-domain cells: "
        f"{metadata['route_coverage']['unobserved_cells']}",
        f"- Pilot reproduction maximum absolute difference: "
        f"{reproduction['maximum_absolute_difference']:.3g}",
        f"- Diagnostic monotonicity violations: "
        f"{metadata['validation']['monotonicity_violation_count']}",
        "- Mixed-precision optimizer: not implemented (separate future stage)",
        "",
    ]
    path.write_text(text.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def _run_audit(args: argparse.Namespace, root: Path) -> bool:
    script = Path(__file__).with_name("audit_quantization_cost_surrogate.py")
    command = [
        sys.executable,
        str(script),
        "--stage1-dir",
        str(args.stage1_dir.resolve()),
        "--surrogate-dir",
        str(root),
        "--output",
        str(root / "independent_audit.json"),
    ]
    completed = subprocess.run(command, check=False)
    audit = read_json(root / "independent_audit.json")
    return completed.returncode == 0 and audit.get("passed") is True


if __name__ == "__main__":
    raise SystemExit(main())
