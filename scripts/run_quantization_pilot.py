#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.balanced import (  # noqa: E402
    BALANCED_DOMAINS,
    EXPECTED_EXPERTS,
    EXPECTED_MEASURED_POSITIONS,
    EXPECTED_MODEL,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MOE_LAYERS,
    EXPECTED_TOP_K,
    canonical_sha256,
    file_sha256,
    load_controlled_source,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed  # noqa: E402
from expert_analysis.io_utils import (  # noqa: E402
    atomic_write_json,
    package_versions,
    read_json,
)
from expert_analysis.masking import LossStatistics, evaluate_next_token_loss  # noqa: E402
from expert_analysis.modeling import (  # noqa: E402
    MoeLayerSpec,
    architecture_metadata,
    discover_moe_layers,
    load_model_and_tokenizer,
)
from expert_analysis.quantization import (  # noqa: E402
    ExpertWeightLayout,
    ReversibleExpertQuantization,
    load_or_compute_loss_checkpoint,
    module_hook_count,
    projected_expert_storage,
    symmetric_groupwise_qdq,
)
from expert_analysis.quantization_pilot import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_FALLBACK_BITS,
    DEFAULT_GROUP_SIZE,
    DEFAULT_PRIMARY_BITS,
    DEFAULT_SEED,
    analyze_quantization_pilot,
    build_pilot_preregistration,
    create_quantization_figures,
    pilot_intervention_panel,
    validate_pilot_preregistration,
    write_or_validate_pilot_preregistration,
    write_quantization_outputs,
    write_quantization_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage-1 OLMoE expert quantization-sensitivity pilot. "
            "This script does not implement mixed-precision allocation or Stage 2."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
        help="Frozen controlled inputs and baseline importance tensors.",
    )
    parser.add_argument(
        "--balanced-results-dir",
        type=Path,
        default=Path("results/expert_domain_balanced_causal_validation"),
        help="Frozen balanced preregistration, matched controls, and masking results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expert_quantization_pilot"),
    )
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--primary-bits", type=int, default=DEFAULT_PRIMARY_BITS)
    parser.add_argument("--fallback-bits", type=int, default=DEFAULT_FALLBACK_BITS)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--real-checkpoint-smoke",
        action="store_true",
        help="With --smoke-only, also load the full checkpoint and test a real expert.",
    )
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--baseline-atol", type=float, default=1e-6)
    parser.add_argument("--baseline-rtol", type=float, default=1e-7)
    args = parser.parse_args()
    if args.model != EXPECTED_MODEL or args.model_revision != EXPECTED_MODEL_REVISION:
        parser.error("Stage 1 is pinned to the validated OLMoE checkpoint and revision")
    if args.group_size < 1:
        parser.error("--group-size must be positive")
    if args.primary_bits != 4 or args.fallback_bits != 3:
        parser.error("The preregistered Stage-1 precisions are 4-bit primary and 3-bit fallback")
    if args.bootstrap_replicates != 1000:
        parser.error("The preregistered Stage-1 analysis requires exactly 1,000 replicates")
    if args.seed != 42:
        parser.error("The frozen controlled experiment requires seed 42")
    if args.batch_size != 1:
        parser.error("The frozen controlled experiment requires batch size 1")
    if args.dtype.lower() not in ("bfloat16", "bf16"):
        parser.error("The production pilot requires BF16 model weights")
    if args.baseline_atol < 0 or args.baseline_rtol < 0:
        parser.error("Baseline reproduction tolerances must be nonnegative")
    if args.analysis_only and args.smoke_only:
        parser.error("--analysis-only and --smoke-only are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    balanced_dir = args.balanced_results_dir.resolve()
    panel_path = output_dir / "pilot_panel_preregistered.json"

    # Freeze selection before reading results.json, masked losses, or any QDQ output.
    expected_panel = build_pilot_preregistration(
        balanced_dir / "selected_experts_preregistered.json",
        balanced_dir / "matched_controls.csv",
        group_size=args.group_size,
        primary_bits=args.primary_bits,
        fallback_bits=args.fallback_bits,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    preregistration = write_or_validate_pilot_preregistration(
        expected_panel, panel_path
    )
    validate_pilot_preregistration(preregistration)
    panel = pilot_intervention_panel(preregistration)

    smoke = _synthetic_smoke(args.group_size, args.cache_dir)
    atomic_write_json(output_dir / "synthetic_smoke_validation.json", smoke)
    if args.smoke_only and not args.real_checkpoint_smoke:
        print("Synthetic quantization smoke validation passed.", flush=True)
        print(f"Frozen panel: {panel_path}", flush=True)
        return 0

    source = load_controlled_source(args.source_dir)
    _validate_panel_source(preregistration, source, balanced_dir)
    balanced_results = _load_balanced_results(balanced_dir, preregistration)

    if args.analysis_only:
        run_config = _load_analysis_run_config(
            args,
            output_dir,
            source,
            preregistration,
            panel,
            balanced_results,
        )
        return _analysis_only(
            args,
            output_dir,
            source,
            balanced_results,
            preregistration,
            panel,
            run_config,
        )

    set_reproducible_seed(args.seed, deterministic=True)
    runtime = resolve_runtime(args.device, args.dtype)
    if runtime.device.type != "cuda":
        raise RuntimeError(
            "The full OLMoE pilot requires CUDA. Use --smoke-only for the cheap "
            "local synthetic validation."
        )
    if "A40" not in runtime.description.upper():
        raise RuntimeError(
            f"The frozen Stage-1 production run requires an NVIDIA A40; found "
            f"{runtime.description}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16")
    run_config = _build_run_config(
        args, source, preregistration, panel, balanced_results
    )
    run_config = _save_or_validate_run_config(output_dir, run_config)
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
    specs = discover_moe_layers(bundle.model)
    architecture = architecture_metadata(bundle.model, specs)
    _validate_runtime_model(bundle, architecture, source.architecture)
    spec_by_layer = {spec.model_layer_index: spec for spec in specs}
    layouts = {
        layer: ExpertWeightLayout(spec_by_layer[layer])
        for layer in sorted({int(row["layer"]) for row in panel})
    }
    layout_validation = _validate_runtime_expert_layouts(layouts)

    print("Running real-checkpoint reversible expert-QDQ smoke validation...", flush=True)
    real_smoke = _real_checkpoint_smoke(
        bundle, source, panel[0], layouts[panel[0]["layer"]], args.group_size
    )
    real_smoke["runtime_expert_layout_validation"] = layout_validation
    atomic_write_json(output_dir / "real_checkpoint_smoke_validation.json", real_smoke)
    if args.smoke_only:
        print("Real-checkpoint quantization smoke validation passed.", flush=True)
        return 0

    source_baselines = _load_balanced_baselines(balanced_dir)
    baselines, baseline_validation = _run_baselines(
        bundle,
        source,
        source_baselines,
        output_dir,
        run_config["inference_fingerprint"],
        preregistration,
        args,
    )
    atomic_write_json(output_dir / "baseline_reproduction.json", baseline_validation)
    noise = max(
        float(row["max_absolute_per_token_nll_difference"])
        for row in baseline_validation["domains"].values()
    )

    quantized: dict[tuple[int, int, int, str], LossStatistics] = {}
    metadata: dict[tuple[int, int, int], dict[str, Any]] = {}
    _run_bit_width(
        args.primary_bits,
        bundle,
        source,
        panel,
        layouts,
        baselines,
        output_dir,
        run_config,
        preregistration,
        args,
        quantized,
        metadata,
    )
    primary_analysis, _ = analyze_quantization_pilot(
        preregistration,
        baselines,
        quantized,
        metadata,
        balanced_results,
        [args.primary_bits],
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        baseline_reproduction_noise_nll=noise,
    )
    evaluated_bits = [args.primary_bits]
    if primary_analysis["stage1_decision"]["decision"] == "PENDING_FALLBACK":
        print(
            "4-bit perturbations meet the preregistered too-small-to-measure fallback "
            "condition; starting the frozen 3-bit fallback.",
            flush=True,
        )
        _run_bit_width(
            args.fallback_bits,
            bundle,
            source,
            panel,
            layouts,
            baselines,
            output_dir,
            run_config,
            preregistration,
            args,
            quantized,
            metadata,
        )
        evaluated_bits.append(args.fallback_bits)

    return _finish(
        args,
        output_dir,
        run_config,
        preregistration,
        balanced_results,
        baselines,
        quantized,
        metadata,
        baseline_validation,
        evaluated_bits,
        smoke,
        real_smoke,
    )


def _synthetic_smoke(group_size: int, cache_dir: str | None) -> dict[str, Any]:
    source = torch.tensor(
        [[0.0, -1.0, 0.5, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    first = symmetric_groupwise_qdq(source, 4, min(group_size, 3))
    second = symmetric_groupwise_qdq(source, 4, min(group_size, 3))
    identity = symmetric_groupwise_qdq(source, 16, group_size)
    if not torch.equal(first.dequantized, second.dequantized):
        raise RuntimeError("Synthetic QDQ is not deterministic")
    if not torch.equal(identity.dequantized, source) or identity.squared_error != 0:
        raise RuntimeError("Synthetic 16-bit QDQ is not an identity")
    if not torch.equal(first.dequantized[1], source[1]):
        raise RuntimeError("Synthetic zero-group handling failed")
    memory = projected_expert_storage([source.shape], 4, min(group_size, 3))
    return {
        "passed": True,
        "method": "synthetic_tensor_qdq",
        "deterministic": True,
        "sixteen_bit_identity": True,
        "zero_groups_safe": True,
        "distortion": first.relative_squared_error,
        "memory_accounting": memory,
        "cached_checkpoint_structure": _inspect_cached_checkpoint(
            cache_dir, group_size
        ),
    }


def _inspect_cached_checkpoint(
    cache_dir: str | None, group_size: int
) -> dict[str, Any]:
    """Inspect safetensors headers only; never materialize the 7B checkpoint."""

    cache_root = Path(cache_dir).resolve() if cache_dir else REPOSITORY_ROOT / ".hf_cache"
    model_slug = "models--" + EXPECTED_MODEL.replace("/", "--")
    snapshot = cache_root / model_slug / "snapshots" / EXPECTED_MODEL_REVISION
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        return {
            "available": False,
            "reason": "complete pinned safetensors index was not found in the selected cache",
            "cache_root": str(cache_root),
        }
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for cached-checkpoint smoke inspection"
        ) from exc
    config = read_json(config_path)
    index = read_json(index_path)
    if config.get("num_hidden_layers") != EXPECTED_MOE_LAYERS:
        raise RuntimeError("Cached OLMoE config has an unexpected layer count")
    if config.get("num_local_experts", config.get("num_experts")) != EXPECTED_EXPERTS:
        raise RuntimeError("Cached OLMoE config has an unexpected expert count")
    if config.get("num_experts_per_tok") != EXPECTED_TOP_K:
        raise RuntimeError("Cached OLMoE config has an unexpected top-k value")
    weight_map = index.get("weight_map", {})
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(str(shard), []).append(str(key))
    serialized_expert_shapes: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    serialized_expert_keys: dict[tuple[int, int], list[str]] = {}
    for shard, keys in by_shard.items():
        shard_path = snapshot / shard
        if not shard_path.is_file():
            raise RuntimeError(f"Cached checkpoint shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                shape = tuple(int(value) for value in handle.get_slice(key).get_shape())
                match = re.search(
                    r"(?:layers|blocks|h|layer)\.(\d+).*?"
                    r"(?:experts|expert_modules)\.(\d+)\.",
                    key,
                )
                if match and len(shape) == 2:
                    identity = (int(match.group(1)), int(match.group(2)))
                    serialized_expert_shapes.setdefault(identity, []).append(shape)
                    serialized_expert_keys.setdefault(identity, []).append(key)
    expected_identities = {
        (layer, expert_id)
        for layer in range(EXPECTED_MOE_LAYERS)
        for expert_id in range(EXPECTED_EXPERTS)
    }
    if set(serialized_expert_shapes) != expected_identities:
        raise RuntimeError("Cached checkpoint expert matrices do not cover all experts")
    expected_shapes = [(1024, 2048), (1024, 2048), (2048, 1024)]
    for identity, shapes in serialized_expert_shapes.items():
        if sorted(shapes) != sorted(expected_shapes):
            raise RuntimeError(f"Cached checkpoint expert shapes differ at {identity}")
    real_weight_smoke = _cached_real_weight_qdq_smoke(
        snapshot,
        weight_map,
        serialized_expert_keys,
        group_size,
        safe_open,
    )
    return {
        "available": True,
        "passed": True,
        "inspection": "safetensors_headers_only_no_weight_materialization",
        "snapshot": str(snapshot),
        "num_moe_layers": EXPECTED_MOE_LAYERS,
        "num_experts_per_layer": EXPECTED_EXPERTS,
        "serialized_matrices_per_expert": 3,
        "serialized_expert_matrix_shapes": [list(shape) for shape in expected_shapes],
        "runtime_layout_note": (
            "Transformers converts separate gate/up/down checkpoint matrices into "
            "tensorized gate_up_proj/down_proj parameters with expert axis 0"
        ),
        "runtime_expert_axis": 0,
        "runtime_matrices_per_expert": 2,
        "runtime_expert_matrix_shapes": [[2048, 2048], [2048, 1024]],
        "grouping_axis_within_expert_slice": -1,
        "grouping_axis_semantics": "input_feature_dimension",
        "real_weight_slice_qdq": real_weight_smoke,
    }


def _cached_real_weight_qdq_smoke(
    snapshot: Path,
    weight_map: Mapping[str, str],
    keys_by_identity: Mapping[tuple[int, int], Sequence[str]],
    group_size: int,
    safe_open: Any,
) -> dict[str, Any]:
    """Load two real expert slices, fuse them as Transformers does, and restore one."""

    source_identities = [(0, 0), (0, 1)]
    fused_gate_up = []
    down = []
    for identity in source_identities:
        matrices = []
        for key in sorted(keys_by_identity[identity]):
            shard = snapshot / weight_map[key]
            with safe_open(shard, framework="pt", device="cpu") as handle:
                matrices.append((key, handle.get_tensor(key)))
        if any(tensor.dtype != torch.bfloat16 for _, tensor in matrices):
            raise RuntimeError(f"Cached expert {identity} is not stored as BF16")
        input_matrices = [
            tensor for _, tensor in matrices if tuple(tensor.shape) == (1024, 2048)
        ]
        output_matrices = [
            tensor for _, tensor in matrices if tuple(tensor.shape) == (2048, 1024)
        ]
        if len(input_matrices) != 2 or len(output_matrices) != 1:
            raise RuntimeError(f"Could not reconstruct cached expert {identity}")
        fused_gate_up.append(torch.cat(input_matrices, dim=0))
        down.append(output_matrices[0])

    class CachedExpertContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.gate_up_proj = nn.Parameter(torch.stack(fused_gate_up))
            self.down_proj = nn.Parameter(torch.stack(down))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values

    class CachedBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(1, 2, bias=False)
            self.experts = CachedExpertContainer()

    block = CachedBlock()
    spec = MoeLayerSpec(
        ordinal=0,
        model_layer_index=0,
        block_name="cached.layers.0.mlp",
        router_name="cached.layers.0.mlp.gate",
        experts_name="cached.layers.0.mlp.experts",
        num_experts=2,
        top_k=1,
        contribution_backend="tensorized_gate_up",
        capture_point="experts_pre",
        block=block,
        router=block.gate,
        experts=block.experts,
    )
    layout = ExpertWeightLayout(spec)
    before = layout.all_fingerprints()
    context = ReversibleExpertQuantization(
        layout, expert_id=0, bits=4, group_size=group_size
    )
    with context:
        during = layout.all_fingerprints()
        if during[0] == before[0] or during[1] != before[1]:
            raise RuntimeError("Cached real-weight QDQ failed expert isolation")
    if not context.restoration_verified or layout.all_fingerprints() != before:
        raise RuntimeError("Cached real-weight QDQ failed exact restoration")
    diagnostic = context.diagnostics()
    return {
        "passed": True,
        "source_experts": [
            {"layer": layer, "expert_id": expert_id}
            for layer, expert_id in source_identities
        ],
        "selected_source_expert": {"layer": 0, "expert_id": 0},
        "unrelated_source_expert": {"layer": 0, "expert_id": 1},
        "runtime_tensor_shapes": layout.metadata()["tensors"],
        "quantization_distortion": diagnostic["quantization_distortion"],
        "original_expert_fingerprint": diagnostic["original_expert_fingerprint"],
        "quantized_expert_fingerprint": diagnostic["quantized_expert_fingerprint"],
        "unrelated_expert_unchanged": True,
        "exact_restoration_verified": diagnostic["exact_restoration_verified"],
        "hooks_before": diagnostic["hooks_before"],
        "hooks_after": diagnostic["hooks_after"],
        "memory_accounting": diagnostic["memory_accounting"],
    }


def _build_run_config(
    args: argparse.Namespace,
    source: Any,
    preregistration: Mapping[str, Any],
    panel: Sequence[Mapping[str, Any]],
    balanced_results: Mapping[str, Any],
) -> dict[str, Any]:
    versions = package_versions()
    basis = {
        "model": args.model,
        "model_revision": args.model_revision,
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "source_input_fingerprint": source.input_fingerprint,
        "pilot_panel_fingerprint": preregistration["pilot_panel_fingerprint"],
        "balanced_masking_raw_sha256": balanced_results[
            "_pilot_raw_masking_audit"
        ]["source_file_sha256"],
        "method": "symmetric_groupwise_weight_only_fake_quantization_qdq",
        "expert_scope": "one_expert_ffn_at_a_time",
        "group_axis": "last_dimension_input_features",
        "group_size": args.group_size,
        "scale_storage_dtype": "float16",
        "primary_bits": args.primary_bits,
        "fallback_bits": args.fallback_bits,
        "fallback_trigger": "only_preregistered_too_small_to_measure_condition",
        "device": args.device,
        "required_hardware": "NVIDIA A40",
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "package_versions": versions,
        "interventions": [
            {
                key: row[key]
                for key in (
                    "intervention_id",
                    "pair_id",
                    "role",
                    "target_domain",
                    "layer",
                    "expert_id",
                )
            }
            for row in panel
        ],
    }
    return {
        **basis,
        "inference_fingerprint": canonical_sha256(basis),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_or_validate_run_config(
    output_dir: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    path = output_dir / "run_config.json"
    if path.exists():
        observed = read_json(path)
        if observed.get("inference_fingerprint") != expected["inference_fingerprint"]:
            raise RuntimeError("Existing quantization artifacts use a different configuration")
        return observed
    atomic_write_json(path, dict(expected))
    return dict(expected)


def _load_analysis_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    source: Any,
    preregistration: Mapping[str, Any],
    panel: Sequence[Mapping[str, Any]],
    balanced_results: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_dir / "run_config.json"
    if not path.is_file():
        raise RuntimeError("Analysis-only mode requires a completed production run_config.json")
    config = read_json(path)
    basis_keys = (
        "model",
        "model_revision",
        "source_collection_fingerprint",
        "source_input_fingerprint",
        "pilot_panel_fingerprint",
        "balanced_masking_raw_sha256",
        "method",
        "expert_scope",
        "group_axis",
        "group_size",
        "scale_storage_dtype",
        "primary_bits",
        "fallback_bits",
        "fallback_trigger",
        "device",
        "required_hardware",
        "dtype",
        "batch_size",
        "seed",
        "package_versions",
        "interventions",
    )
    try:
        basis = {key: config[key] for key in basis_keys}
    except KeyError as exc:
        raise RuntimeError(f"Production run config is missing {exc.args[0]}") from exc
    if canonical_sha256(basis) != config.get("inference_fingerprint"):
        raise RuntimeError("Production run configuration fingerprint is invalid")
    expected = {
        "model": args.model,
        "model_revision": args.model_revision,
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "source_input_fingerprint": source.input_fingerprint,
        "pilot_panel_fingerprint": preregistration["pilot_panel_fingerprint"],
        "balanced_masking_raw_sha256": balanced_results[
            "_pilot_raw_masking_audit"
        ]["source_file_sha256"],
        "group_size": args.group_size,
        "primary_bits": args.primary_bits,
        "fallback_bits": args.fallback_bits,
        "device": args.device,
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "interventions": [
            {
                key: row[key]
                for key in (
                    "intervention_id",
                    "pair_id",
                    "role",
                    "target_domain",
                    "layer",
                    "expert_id",
                )
            }
            for row in panel
        ],
    }
    _require_metadata(config, expected, path)
    if config.get("bootstrap_replicates") != args.bootstrap_replicates:
        raise RuntimeError("Production bootstrap replicate count differs")
    return config


def _validate_panel_source(
    preregistration: Mapping[str, Any], source: Any, balanced_dir: Path
) -> None:
    frozen = preregistration["source"]
    if frozen["balanced_selection_input_fingerprint"] != source.input_fingerprint:
        raise RuntimeError("Pilot panel input fingerprint no longer matches controlled source")
    for domain in BALANCED_DOMAINS:
        path = source.root / "controlled_inputs" / f"{domain}.npz"
        if file_sha256(path) != frozen["controlled_input_file_sha256"][domain]:
            raise RuntimeError(f"Frozen controlled input changed for {domain}")
    if file_sha256(balanced_dir / "selected_experts_preregistered.json") != frozen[
        "balanced_preregistration_sha256"
    ]:
        raise RuntimeError("Balanced preregistration file hash changed")
    if file_sha256(balanced_dir / "matched_controls.csv") != frozen[
        "balanced_matched_controls_sha256"
    ]:
        raise RuntimeError("Balanced matched-control file hash changed")


def _load_balanced_results(
    balanced_dir: Path, preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    path = balanced_dir / "results.json"
    if not path.is_file():
        raise RuntimeError(
            "Balanced results.json is required for the quantization-vs-masking analysis"
        )
    results = read_json(path)
    if results.get("integrity_validation", {}).get("passed") is not True:
        raise RuntimeError("Balanced masking results did not pass their integrity audit")
    observed = results.get("preregistration", {}).get("preregistration_fingerprint")
    expected = preregistration["source"]["balanced_preregistration_fingerprint"]
    if observed != expected:
        raise RuntimeError("Balanced results use a different preregistered panel")
    raw_path = balanced_dir / "per_example_loss_changes.npz"
    if not raw_path.is_file():
        raise RuntimeError(
            "Balanced per_example_loss_changes.npz is required to reconstruct masking contrasts"
        )
    raw_sha256 = file_sha256(raw_path)
    expected_raw_sha256 = results.get("artifact_manifest", {}).get(
        "file_sha256", {}
    ).get("per_example_loss_changes.npz")
    if expected_raw_sha256 != raw_sha256:
        raise RuntimeError("Balanced raw per-example array failed its recorded hash")
    with np.load(raw_path, allow_pickle=False) as data:
        required = {
            "intervention_ids",
            "roles",
            "target_domains",
            "layers",
            "expert_ids",
            "domain_names",
            "per_example_loss_changes",
        }
        if not required.issubset(data.files):
            raise RuntimeError("Balanced raw per-example array is missing required fields")
        intervention_ids = [str(value) for value in data["intervention_ids"]]
        roles = [str(value) for value in data["roles"]]
        target_domains = [str(value) for value in data["target_domains"]]
        layers = data["layers"].astype(int)
        expert_ids = data["expert_ids"].astype(int)
        domain_names = [str(value) for value in data["domain_names"]]
        changes = data["per_example_loss_changes"].astype(np.float64)
    if domain_names != list(BALANCED_DOMAINS) or changes.shape != (24, 4, 100):
        raise RuntimeError("Balanced raw per-example array has unexpected geometry")
    if not np.all(np.isfinite(changes)):
        raise RuntimeError("Balanced raw per-example masking changes are non-finite")
    published = {
        row["intervention_id"]: float(row["target_minus_mean_other_contrast"])
        for row in results["balanced_analysis"]["intervention_contrasts"]
    }
    raw_rows = []
    maximum_difference = 0.0
    for index, intervention_id in enumerate(intervention_ids):
        target_index = domain_names.index(target_domains[index])
        other_indices = [value for value in range(4) if value != target_index]
        contrast = float(
            changes[index, target_index].mean()
            - changes[index, other_indices].mean()
        )
        if intervention_id not in published:
            raise RuntimeError(f"Balanced results omit {intervention_id}")
        difference = abs(contrast - published[intervention_id])
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-12:
            raise RuntimeError(
                f"Balanced masking contrast failed raw reconstruction for {intervention_id}"
            )
        raw_rows.append(
            {
                "intervention_id": intervention_id,
                "role": roles[index],
                "target_domain": target_domains[index],
                "layer": int(layers[index]),
                "expert_id": int(expert_ids[index]),
                "target_minus_mean_other_contrast": contrast,
            }
        )
    panel_identities = {
        (row["role"], int(row["layer"]), int(row["expert_id"]))
        for row in pilot_intervention_panel(preregistration)
    }
    raw_identities = {
        (
            "specialist" if row["role"] == "specialized" else row["role"],
            row["layer"],
            row["expert_id"],
        )
        for row in raw_rows
    }
    if not panel_identities.issubset(raw_identities):
        raise RuntimeError("Balanced raw masking array does not cover the pilot panel")
    results["_pilot_raw_masking_contrasts"] = raw_rows
    results["_pilot_raw_masking_audit"] = {
        "passed": True,
        "source_file": str(raw_path),
        "source_file_sha256": raw_sha256,
        "shape": list(changes.shape),
        "reconstructed_interventions": len(raw_rows),
        "maximum_absolute_published_contrast_difference": maximum_difference,
    }
    return results


def _validate_runtime_model(
    bundle: Any, runtime: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    if bundle.resolved_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError(f"Resolved model revision mismatch: {bundle.resolved_revision!r}")
    if runtime.get("num_moe_layers") != EXPECTED_MOE_LAYERS:
        raise RuntimeError("Runtime model does not contain 16 MoE layers")
    if runtime.get("num_experts") != EXPECTED_EXPERTS:
        raise RuntimeError("Runtime model does not contain 64 experts/layer")
    if runtime.get("top_k") != [EXPECTED_TOP_K]:
        raise RuntimeError("Runtime model does not use top-8 routing")
    for key in ("model_class", "config_model_type", "num_moe_layers", "num_experts", "top_k"):
        if runtime.get(key) != source.get(key):
            raise RuntimeError(f"Runtime architecture differs from frozen source for {key}")


def _validate_runtime_expert_layouts(
    layouts: Mapping[int, ExpertWeightLayout]
) -> dict[str, Any]:
    expected_shapes = {(2048, 2048), (2048, 1024)}
    rows = {}
    for layer, layout in layouts.items():
        metadata = layout.metadata()
        shapes = {
            tuple(row["expert_slice_shape"]) for row in metadata["tensors"]
        }
        axes = {row["expert_axis"] for row in metadata["tensors"]}
        if metadata["num_experts"] != EXPECTED_EXPERTS:
            raise RuntimeError(f"Runtime expert count differs at layer {layer}")
        if metadata["tensor_count_per_expert"] != 2 or shapes != expected_shapes:
            raise RuntimeError(f"Runtime expert tensor shapes differ at layer {layer}")
        if axes != {0}:
            raise RuntimeError(f"Runtime expert axis is not zero at layer {layer}")
        rows[str(layer)] = metadata
    return {
        "passed": True,
        "validated_layers": sorted(layouts),
        "expected_expert_slice_shapes": [
            list(shape) for shape in sorted(expected_shapes)
        ],
        "expected_expert_axis": 0,
        "grouping_axis_within_expert_slice": -1,
        "layers": rows,
    }


def _real_checkpoint_smoke(
    bundle: Any,
    source: Any,
    intervention: Mapping[str, Any],
    layout: ExpertWeightLayout,
    group_size: int,
) -> dict[str, Any]:
    before_hooks = module_hook_count(bundle.model)
    original = layout.fingerprint(int(intervention["expert_id"]))
    context = ReversibleExpertQuantization(
        layout, int(intervention["expert_id"]), 4, group_size
    )
    with context:
        if context.quantized_fingerprint == original:
            raise RuntimeError("Real-checkpoint 4-bit smoke did not change expert weights")
        one = _first_example(source.prepared[intervention["target_domain"]])
        result, loss_diagnostics = evaluate_next_token_loss(bundle, one, batch_size=1)
        if not np.all(np.isfinite(result.loss_sums)):
            raise RuntimeError("Real-checkpoint QDQ smoke produced non-finite loss")
    diagnostics = context.diagnostics()
    if not context.restoration_verified or layout.fingerprint(
        int(intervention["expert_id"])
    ) != original:
        raise RuntimeError("Real-checkpoint QDQ smoke did not restore weights exactly")
    after_hooks = module_hook_count(bundle.model)
    if before_hooks != after_hooks:
        raise RuntimeError("Real-checkpoint QDQ smoke leaked model hooks")
    return {
        "passed": True,
        "runtime_description": bundle.runtime.description,
        "intervention_id": intervention["intervention_id"],
        "loss_nll": float(result.per_token_nll.mean()),
        "loss_diagnostics": loss_diagnostics,
        "quantization": diagnostics,
        "hooks_before": before_hooks,
        "hooks_after": after_hooks,
    }


def _first_example(examples: Any) -> Any:
    from expert_analysis.controlled import PreparedDomainExamples

    result = PreparedDomainExamples(
        domain=examples.domain,
        input_ids=examples.input_ids[:1].copy(),
        attention_mask=examples.attention_mask[:1].copy(),
        measurement_mask=examples.measurement_mask[:1].copy(),
        metadata=dict(examples.metadata),
    )
    result.validate()
    return result


def _load_balanced_baselines(balanced_dir: Path) -> dict[str, LossStatistics]:
    run_config = read_json(balanced_dir / "run_config.json")
    integrity = read_json(balanced_dir / "integrity_validation.json")
    if integrity.get("passed") is not True:
        raise RuntimeError("Balanced baseline source did not pass integrity validation")
    fingerprint = run_config.get("inference_fingerprint")
    output = {}
    for domain in BALANCED_DOMAINS:
        path = balanced_dir / "masking" / "baseline" / f"{domain}.npz"
        metadata_path = path.with_suffix(".metadata.json")
        metadata = read_json(metadata_path)
        if metadata.get("inference_fingerprint") != fingerprint:
            raise RuntimeError(f"Balanced baseline metadata mismatch for {domain}")
        output[domain] = LossStatistics.load(path)
    return output


def _run_baselines(
    bundle: Any,
    source: Any,
    reference: Mapping[str, LossStatistics],
    output_dir: Path,
    fingerprint: str,
    preregistration: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, LossStatistics], dict[str, Any]]:
    baselines = {}
    checks = {}
    for domain in BALANCED_DOMAINS:
        path = output_dir / "quantization" / "baseline" / f"{domain}.npz"
        expected = {
            "inference_fingerprint": fingerprint,
            "pilot_panel_fingerprint": preregistration["pilot_panel_fingerprint"],
            "controlled_input_sha256": preregistration["source"][
                "controlled_input_file_sha256"
            ][domain],
            "kind": "fresh_bf16_baseline",
            "domain": domain,
        }
        checkpoint = load_or_compute_loss_checkpoint(
            path,
            expected,
            source.prepared[domain].measurement_mask.sum(axis=1),
            lambda domain=domain: evaluate_next_token_loss(
                bundle, source.prepared[domain], batch_size=args.batch_size
            ),
            resume=args.resume,
        )
        fresh = checkpoint.statistics
        old = reference[domain]
        if not np.array_equal(fresh.token_counts, old.token_counts):
            raise RuntimeError(f"BF16 baseline token counts changed for {domain}")
        difference = np.abs(fresh.loss_sums - old.loss_sums)
        if not np.allclose(
            fresh.loss_sums, old.loss_sums, atol=args.baseline_atol, rtol=args.baseline_rtol
        ):
            raise RuntimeError(
                f"BF16 baseline reproduction failed for {domain}: max loss-sum "
                f"difference {difference.max():.9g}"
            )
        diagnostics = checkpoint.metadata.get("diagnostics", {})
        if diagnostics.get("hooks_before") != diagnostics.get("hooks_after"):
            raise RuntimeError(f"BF16 baseline evaluation leaked hooks for {domain}")
        checks[domain] = {
            "passed": True,
            "resumed": checkpoint.resumed,
            "bitwise_equal": bool(np.array_equal(fresh.loss_sums, old.loss_sums)),
            "max_absolute_loss_sum_difference": float(difference.max()),
            "max_absolute_per_token_nll_difference": float(
                np.max(difference / fresh.token_counts)
            ),
            "mean_absolute_loss_sum_difference": float(difference.mean()),
            "fresh_mean_nll": float(fresh.per_token_nll.mean()),
            "balanced_mean_nll": float(old.per_token_nll.mean()),
            "atol": args.baseline_atol,
            "rtol": args.baseline_rtol,
        }
        baselines[domain] = fresh
    return baselines, {
        "passed": True,
        "reference": "balanced causal validation fresh BF16 baselines",
        "domains": checks,
    }


def _run_bit_width(
    bit_width: int,
    bundle: Any,
    source: Any,
    panel: Sequence[Mapping[str, Any]],
    layouts: Mapping[int, ExpertWeightLayout],
    baselines: Mapping[str, LossStatistics],
    output_dir: Path,
    run_config: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    args: argparse.Namespace,
    quantized: dict[tuple[int, int, int, str], LossStatistics],
    metadata: dict[tuple[int, int, int], dict[str, Any]],
) -> None:
    for intervention_index, intervention in enumerate(panel, 1):
        layer = int(intervention["layer"])
        expert_id = int(intervention["expert_id"])
        layout = layouts[layer]
        print(
            f"[{intervention_index}/{len(panel)}] {bit_width}-bit "
            f"{intervention['role']} {intervention['target_domain']} L{layer}/E{expert_id}",
            flush=True,
        )
        context = ReversibleExpertQuantization(
            layout,
            expert_id,
            bit_width,
            args.group_size,
            verify_unrelated_experts=True,
        )
        domain_paths: list[Path] = []
        with context:
            static_metadata = {
                "inference_fingerprint": run_config["inference_fingerprint"],
                "pilot_panel_fingerprint": preregistration[
                    "pilot_panel_fingerprint"
                ],
                "intervention_id": intervention["intervention_id"],
                "pair_id": intervention["pair_id"],
                "role": intervention["role"],
                "target_domain": intervention["target_domain"],
                "layer": layer,
                "expert_id": expert_id,
                "bit_width": bit_width,
                "group_size": args.group_size,
                "original_expert_fingerprint": context.original_fingerprint,
                "quantized_expert_fingerprint": context.quantized_fingerprint,
                "quantization_distortion": context.distortion,
            }
            for domain in BALANCED_DOMAINS:
                path = (
                    output_dir
                    / "quantization"
                    / f"bit_{bit_width}"
                    / f"layer_{layer}_expert_{expert_id}"
                    / f"{domain}.npz"
                )
                expected = {
                    **static_metadata,
                    "domain": domain,
                    "controlled_input_sha256": preregistration["source"][
                        "controlled_input_file_sha256"
                    ][domain],
                }
                checkpoint = load_or_compute_loss_checkpoint(
                    path,
                    expected,
                    baselines[domain].token_counts,
                    lambda domain=domain: evaluate_next_token_loss(
                        bundle, source.prepared[domain], batch_size=args.batch_size
                    ),
                    resume=args.resume,
                )
                diagnostics = checkpoint.metadata.get("diagnostics", {})
                if diagnostics.get("hooks_before") != diagnostics.get("hooks_after"):
                    raise RuntimeError(
                        f"Quantized loss evaluation leaked hooks for {bit_width}-bit "
                        f"L{layer}/E{expert_id}/{domain}"
                    )
                quantized[(bit_width, layer, expert_id, domain)] = checkpoint.statistics
                domain_paths.append(path)
                print(
                    f"[{domain}] {'resume' if checkpoint.resumed else 'saved'}: "
                    f"{bit_width}-bit L{layer}/E{expert_id}",
                    flush=True,
                )
        diagnostic = context.diagnostics()
        if not diagnostic["exact_restoration_verified"]:
            raise RuntimeError(f"Exact QDQ restoration failed for L{layer}/E{expert_id}")
        quantization_path = (
            output_dir
            / "quantization"
            / f"bit_{bit_width}"
            / f"layer_{layer}_expert_{expert_id}"
            / "quantization.metadata.json"
        )
        quantization_payload = {
            "inference_fingerprint": run_config["inference_fingerprint"],
            "pilot_panel_fingerprint": preregistration["pilot_panel_fingerprint"],
            "intervention_id": intervention["intervention_id"],
            "pair_id": intervention["pair_id"],
            "role": intervention["role"],
            "target_domain": intervention["target_domain"],
            **diagnostic,
        }
        atomic_write_json(quantization_path, quantization_payload)
        metadata[(bit_width, layer, expert_id)] = quantization_payload
        completed = sum(
            1
            for width, *_ in quantized
            if width == bit_width
        )
        atomic_write_json(
            output_dir / "progress.json",
            {
                "inference_fingerprint": run_config["inference_fingerprint"],
                "pilot_panel_fingerprint": preregistration[
                    "pilot_panel_fingerprint"
                ],
                "completed_expert_domain_bit_width_evaluations": completed,
                "total_primary_evaluations": len(panel) * len(BALANCED_DOMAINS),
                "current_bit_width": bit_width,
                "last_completed_intervention_id": intervention["intervention_id"],
                "last_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


def _analysis_only(
    args: argparse.Namespace,
    output_dir: Path,
    source: Any,
    balanced_results: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    panel: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> int:
    baselines = _load_quantization_baselines(
        output_dir, source, preregistration, run_config
    )
    baseline_validation = read_json(output_dir / "baseline_reproduction.json")
    quantized: dict[tuple[int, int, int, str], LossStatistics] = {}
    metadata: dict[tuple[int, int, int], dict[str, Any]] = {}
    evaluated_bits = []
    for bit_width in (args.primary_bits, args.fallback_bits):
        if _bit_width_complete(
            bit_width, output_dir, panel, preregistration, run_config
        ):
            _load_bit_width(
                bit_width,
                output_dir,
                panel,
                baselines,
                preregistration,
                run_config,
                quantized,
                metadata,
            )
            evaluated_bits.append(bit_width)
    if not evaluated_bits or evaluated_bits[0] != args.primary_bits:
        raise RuntimeError("A complete 4-bit pilot is required for analysis-only mode")
    smoke = read_json(output_dir / "synthetic_smoke_validation.json")
    real_smoke = read_json(output_dir / "real_checkpoint_smoke_validation.json")
    return _finish(
        args,
        output_dir,
        run_config,
        preregistration,
        balanced_results,
        baselines,
        quantized,
        metadata,
        baseline_validation,
        evaluated_bits,
        smoke,
        real_smoke,
    )


def _load_quantization_baselines(
    output_dir: Path,
    source: Any,
    preregistration: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, LossStatistics]:
    output = {}
    for domain in BALANCED_DOMAINS:
        path = output_dir / "quantization" / "baseline" / f"{domain}.npz"
        metadata = read_json(path.with_suffix(".metadata.json"))
        expected = {
            "inference_fingerprint": run_config["inference_fingerprint"],
            "pilot_panel_fingerprint": preregistration["pilot_panel_fingerprint"],
            "controlled_input_sha256": preregistration["source"][
                "controlled_input_file_sha256"
            ][domain],
            "kind": "fresh_bf16_baseline",
            "domain": domain,
        }
        _require_metadata(metadata, expected, path)
        result = LossStatistics.load(path)
        expected_tokens = source.prepared[domain].measurement_mask.sum(axis=1)
        if not np.array_equal(result.token_counts, expected_tokens):
            raise RuntimeError(f"Baseline token geometry mismatch for {domain}")
        output[domain] = result
    return output


def _bit_width_complete(
    bit_width: int,
    output_dir: Path,
    panel: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> bool:
    del preregistration, run_config
    for row in panel:
        directory = (
            output_dir
            / "quantization"
            / f"bit_{bit_width}"
            / f"layer_{row['layer']}_expert_{row['expert_id']}"
        )
        if not (directory / "quantization.metadata.json").is_file():
            return False
        for domain in BALANCED_DOMAINS:
            path = directory / f"{domain}.npz"
            if not path.is_file() or not path.with_suffix(".metadata.json").is_file():
                return False
    return True


def _load_bit_width(
    bit_width: int,
    output_dir: Path,
    panel: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, LossStatistics],
    preregistration: Mapping[str, Any],
    run_config: Mapping[str, Any],
    quantized: dict[tuple[int, int, int, str], LossStatistics],
    metadata: dict[tuple[int, int, int], dict[str, Any]],
) -> None:
    for row in panel:
        layer = int(row["layer"])
        expert_id = int(row["expert_id"])
        directory = (
            output_dir
            / "quantization"
            / f"bit_{bit_width}"
            / f"layer_{layer}_expert_{expert_id}"
        )
        quantization_metadata = read_json(directory / "quantization.metadata.json")
        expected_quantization = {
            "inference_fingerprint": run_config["inference_fingerprint"],
            "pilot_panel_fingerprint": preregistration["pilot_panel_fingerprint"],
            "intervention_id": row["intervention_id"],
            "pair_id": row["pair_id"],
            "role": row["role"],
            "target_domain": row["target_domain"],
            "layer": layer,
            "expert_id": expert_id,
            "bits": bit_width,
            "group_size": run_config["group_size"],
        }
        _require_metadata(
            quantization_metadata,
            expected_quantization,
            directory / "quantization.metadata.json",
        )
        if not quantization_metadata.get("exact_restoration_verified"):
            raise RuntimeError(f"QDQ restoration is not verified for {directory}")
        metadata[(bit_width, layer, expert_id)] = quantization_metadata
        for domain in BALANCED_DOMAINS:
            path = directory / f"{domain}.npz"
            domain_metadata = read_json(path.with_suffix(".metadata.json"))
            expected_domain = {
                "inference_fingerprint": run_config["inference_fingerprint"],
                "pilot_panel_fingerprint": preregistration[
                    "pilot_panel_fingerprint"
                ],
                "intervention_id": row["intervention_id"],
                "pair_id": row["pair_id"],
                "role": row["role"],
                "target_domain": row["target_domain"],
                "layer": layer,
                "expert_id": expert_id,
                "bit_width": bit_width,
                "group_size": run_config["group_size"],
                "domain": domain,
                "controlled_input_sha256": preregistration["source"][
                    "controlled_input_file_sha256"
                ][domain],
            }
            _require_metadata(domain_metadata, expected_domain, path)
            result = LossStatistics.load(path)
            if not np.array_equal(result.token_counts, baselines[domain].token_counts):
                raise RuntimeError(f"Quantized token geometry mismatch for {path}")
            quantized[(bit_width, layer, expert_id, domain)] = result


def _finish(
    args: argparse.Namespace,
    output_dir: Path,
    run_config: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    balanced_results: Mapping[str, Any],
    baselines: Mapping[str, LossStatistics],
    quantized: Mapping[tuple[int, int, int, str], LossStatistics],
    metadata: Mapping[tuple[int, int, int], Mapping[str, Any]],
    baseline_validation: Mapping[str, Any],
    evaluated_bits: Sequence[int],
    synthetic_smoke: Mapping[str, Any],
    real_smoke: Mapping[str, Any],
) -> int:
    noise = max(
        float(row["max_absolute_per_token_nll_difference"])
        for row in baseline_validation["domains"].values()
    )
    analysis, arrays = analyze_quantization_pilot(
        preregistration,
        baselines,
        quantized,
        metadata,
        balanced_results,
        evaluated_bits,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        baseline_reproduction_noise_nll=noise,
    )
    if analysis["stage1_decision"]["decision"] == "PENDING_FALLBACK":
        raise RuntimeError("Final analysis still requires the preregistered 3-bit fallback")
    if analysis["stage1_decision"]["decision"] not in ("GO", "NO_GO"):
        raise RuntimeError("Final Stage-1 decision must be GO or NO_GO")
    write_quantization_outputs(analysis, arrays, output_dir)
    results: dict[str, Any] = {
        "schema_version": 1,
        "run_config": {
            **dict(run_config),
            "runtime_description": real_smoke.get("runtime_description"),
        },
        "pilot_panel_preregistration": dict(preregistration),
        "synthetic_smoke_validation": dict(synthetic_smoke),
        "real_checkpoint_smoke_validation": dict(real_smoke),
        "baseline_reproduction": dict(baseline_validation),
        "balanced_masking_input_audit": dict(
            balanced_results.get("_pilot_raw_masking_audit", {})
        ),
        "quantization_analysis": analysis,
        "artifact_manifest": {},
    }
    figure_paths = [] if args.skip_plots else create_quantization_figures(results, output_dir)
    results["artifact_manifest"]["figures"] = [
        str(path.relative_to(output_dir)) for path in figure_paths
    ]
    atomic_write_json(output_dir / "results.json", results)
    write_quantization_summary(results, output_dir / "SUMMARY.md")
    audit = _audit_outputs(output_dir, len(evaluated_bits), len(figure_paths))
    results["artifact_manifest"].update(audit)
    atomic_write_json(output_dir / "results.json", results)
    write_quantization_summary(results, output_dir / "SUMMARY.md")
    print(f"Stage-1 quantization pilot complete: {output_dir}", flush=True)
    print(f"Decision: {analysis['stage1_decision']['decision']}", flush=True)
    print(f"Summary: {output_dir / 'SUMMARY.md'}", flush=True)
    return 0


def _audit_outputs(output_dir: Path, bit_count: int, figure_count: int) -> dict[str, Any]:
    required = [
        "pilot_panel_preregistered.json",
        "quantization_pilot_results.csv",
        "quantization_pilot_pairwise.csv",
        "specialist_vs_control.csv",
        "quantization_vs_masking.csv",
        "quantization_distortion.csv",
        "per_example_quantization_losses.npz",
        "results.json",
        "stage1_decision.json",
        "SUMMARY.md",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("Generated pilot output audit found: " + ", ".join(missing))
    with np.load(output_dir / "per_example_quantization_losses.npz", allow_pickle=False) as data:
        expected = (bit_count, 16, 4, 100)
        if data["per_example_loss_changes"].shape != expected:
            raise RuntimeError("Per-example quantization array has the wrong shape")
        if not np.all(np.isfinite(data["per_example_loss_changes"])):
            raise RuntimeError("Per-example quantization losses contain non-finite values")
    if figure_count not in (0, 10):
        raise RuntimeError(f"Expected zero or ten figure files, found {figure_count}")
    paths = [output_dir / name for name in required if name not in ("results.json", "SUMMARY.md")]
    paths.extend(sorted((output_dir / "figures").glob("figure_*")))
    return {
        "output_audit_passed": True,
        "required_files": required,
        "per_example_loss_changes_shape": [bit_count, 16, 4, 100],
        "figure_file_count": figure_count,
        "file_sha256": {
            str(path.relative_to(output_dir)): file_sha256(path)
            for path in paths
            if path.is_file()
        },
    }


def _require_metadata(
    observed: Mapping[str, Any], expected: Mapping[str, Any], path: Path
) -> None:
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"Checkpoint metadata mismatch for {path}: " + ", ".join(mismatches)
        )


if __name__ == "__main__":
    raise SystemExit(main())
