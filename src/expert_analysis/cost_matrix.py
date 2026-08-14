from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .activation_quantization_cost import (
    calculate_perturbation_sums,
    finalize_activation_surrogates,
    replay_expert_outputs,
    selected_routes,
)
from .balanced import array_sha256, file_sha256
from .expert_replay import ReplayCapture
from .gradient_quantization_cost import GradientCapture, calculate_gqs
from .io_utils import atomic_save_npz, atomic_write_json, read_json, write_csv
from .modeling import MoeLayerSpec
from .quantization import (
    ExpertWeightLayout,
    ReversibleExpertQuantization,
    projected_expert_storage,
)


COST_MATRIX_SCHEMA_VERSION = 1
DEFAULT_BIT_WIDTHS = (3, 4, 8, 16)


@dataclass(frozen=True)
class CostChunk:
    cost: np.ndarray
    route_counts: np.ndarray
    unobserved: np.ndarray
    diagnostics: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validate(self, num_experts: int) -> None:
        if self.cost.shape != (num_experts,):
            raise ValueError("Cost chunk has the wrong expert dimension")
        if self.route_counts.shape != (num_experts,):
            raise ValueError("Route-count chunk has the wrong expert dimension")
        if self.unobserved.shape != (num_experts,):
            raise ValueError("Unobserved chunk has the wrong expert dimension")
        if not np.all(np.isfinite(self.cost)) or np.any(self.cost < 0):
            raise ValueError("Cost chunk contains invalid values")
        if np.any(self.route_counts < 0):
            raise ValueError("Route-count chunk contains negative values")
        if not np.array_equal(self.unobserved, self.route_counts == 0):
            raise ValueError("Unobserved flags do not match zero route counts")
        for name, values in self.diagnostics.items():
            if values.shape != (num_experts,):
                raise ValueError(f"Diagnostic chunk {name} has the wrong shape")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Diagnostic chunk {name} contains invalid values")


def save_cost_chunk(
    path: Path,
    chunk: CostChunk,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    chunk.validate(len(chunk.cost))
    arrays: dict[str, Any] = {
        "cost": chunk.cost.astype(np.float64, copy=False),
        "route_counts": chunk.route_counts.astype(np.int64, copy=False),
        "unobserved": chunk.unobserved.astype(np.bool_, copy=False),
    }
    arrays.update(
        {
            f"diagnostic_{name}": values.astype(np.float64, copy=False)
            for name, values in chunk.diagnostics.items()
        }
    )
    atomic_save_npz(path, **arrays)
    payload = {
        **dict(metadata),
        "schema_version": COST_MATRIX_SCHEMA_VERSION,
        "num_experts": len(chunk.cost),
        "array_sha256": {
            key: array_sha256(np.asarray(value)) for key, value in arrays.items()
        },
        "npz_sha256": file_sha256(path),
    }
    atomic_write_json(path.with_suffix(".metadata.json"), payload)
    return payload


def load_cost_chunk(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    num_experts: int,
) -> CostChunk:
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Cost chunk is incomplete: {path}")
    metadata = read_json(metadata_path)
    mismatches = [
        key for key, value in expected_metadata.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Cost chunk fingerprint mismatch for {path}: " + ", ".join(mismatches)
        )
    if metadata.get("npz_sha256") != file_sha256(path):
        raise RuntimeError(f"Cost chunk file hash mismatch for {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for key, values in arrays.items():
        if metadata.get("array_sha256", {}).get(key) != array_sha256(values):
            raise RuntimeError(f"Cost chunk array hash mismatch for {path}:{key}")
    required = {"cost", "route_counts", "unobserved"}
    if not required.issubset(arrays):
        raise RuntimeError(f"Cost chunk arrays are incomplete in {path}")
    result = CostChunk(
        cost=arrays["cost"].astype(np.float64),
        route_counts=arrays["route_counts"].astype(np.int64),
        unobserved=arrays["unobserved"].astype(np.bool_),
        diagnostics={
            key.removeprefix("diagnostic_"): value.astype(np.float64)
            for key, value in arrays.items()
            if key.startswith("diagnostic_")
        },
        metadata=metadata,
    )
    result.validate(num_experts)
    return result


def compute_activation_layer_costs(
    spec: MoeLayerSpec,
    layout: ExpertWeightLayout,
    captures: Mapping[str, ReplayCapture],
    domains: Sequence[str],
    bit_widths: Sequence[int],
    *,
    group_size: int = 128,
    chunk_size: int = 512,
    requested_keys: set[tuple[str, int]] | None = None,
) -> tuple[dict[tuple[str, int], CostChunk], list[dict[str, Any]]]:
    """Compute every expert in one layer, exploiting selected-route sparsity."""

    num_experts = spec.num_experts
    bits = [int(value) for value in bit_widths]
    all_keys = {(domain, bit) for domain in domains for bit in bits}
    requested = all_keys if requested_keys is None else set(requested_keys)
    if not requested or not requested.issubset(all_keys):
        raise ValueError("requested_keys must be a nonempty subset of domain/bit pairs")
    active_domains = [
        domain for domain in domains if any((domain, bit) in requested for bit in bits)
    ]
    active_bits = [
        bit for bit in bits if any((domain, bit) in requested for domain in domains)
    ]
    fields = (
        "aod",
        "reod",
        "apd",
        "uod",
        "gated_delta_squared",
        "gated_baseline_squared",
        "ungated_delta_squared",
    )
    values = {
        (domain, bit): {
            name: np.zeros(num_experts, dtype=np.float64) for name in fields
        }
        for domain, bit in requested
    }
    counts = {
        domain: np.zeros(num_experts, dtype=np.int64) for domain in active_domains
    }
    qdq_rows: list[dict[str, Any]] = []
    layer_before = layout.all_fingerprints()
    for expert_id in range(num_experts):
        routed: dict[str, tuple[Any, Any, Any]] = {}
        baseline: dict[str, Any] = {}
        for domain in active_domains:
            hidden, gates, examples = selected_routes(captures[domain], expert_id)
            routed[domain] = (hidden, gates, examples)
            counts[domain][expert_id] = int(hidden.shape[0])
            baseline[domain] = replay_expert_outputs(
                spec, expert_id, hidden, chunk_size=chunk_size
            )
        for bit_width in active_bits:
            if bit_width == 16:
                continue
            requested_domains = [
                domain
                for domain in active_domains
                if (domain, bit_width) in requested
            ]
            context = ReversibleExpertQuantization(
                layout,
                expert_id,
                bit_width,
                group_size,
                verify_unrelated_experts=False,
            )
            with context:
                for domain in requested_domains:
                    hidden, gates, _ = routed[domain]
                    quantized = replay_expert_outputs(
                        spec, expert_id, hidden, chunk_size=chunk_size
                    )
                    sums = calculate_perturbation_sums(
                        baseline[domain], quantized, gates
                    )
                    metrics = finalize_activation_surrogates(
                        sums,
                        layer_energy=captures[domain].layer_energy,
                        domain_token_count=captures[domain].num_tokens,
                    )
                    target = values[(domain, bit_width)]
                    target["aod"][expert_id] = metrics.aod
                    target["reod"][expert_id] = metrics.reod
                    target["apd"][expert_id] = metrics.apd
                    target["uod"][expert_id] = metrics.uod
                    target["gated_delta_squared"][
                        expert_id
                    ] = sums.gated_delta_squared
                    target["gated_baseline_squared"][
                        expert_id
                    ] = sums.gated_baseline_squared
                    target["ungated_delta_squared"][
                        expert_id
                    ] = sums.ungated_delta_squared
            diagnostics = context.diagnostics()
            qdq_rows.append(
                {
                    "layer": spec.model_layer_index,
                    "expert_id": expert_id,
                    "bit_width": bit_width,
                    "original_expert_fingerprint": diagnostics[
                        "original_expert_fingerprint"
                    ],
                    "quantized_expert_fingerprint": diagnostics[
                        "quantized_expert_fingerprint"
                    ],
                    "quantization_distortion": diagnostics[
                        "quantization_distortion"
                    ],
                    "exact_restoration_verified": diagnostics[
                        "exact_restoration_verified"
                    ],
                }
            )
    layer_after = layout.all_fingerprints()
    if layer_before != layer_after:
        raise RuntimeError(
            f"Model weights changed while building layer {spec.model_layer_index} costs"
        )
    chunks: dict[tuple[str, int], CostChunk] = {}
    for domain in domains:
        for bit_width in bits:
            if (domain, bit_width) not in requested:
                continue
            metric = values[(domain, bit_width)]
            chunks[(domain, bit_width)] = CostChunk(
                cost=metric["aod"],
                route_counts=counts[domain].copy(),
                unobserved=counts[domain] == 0,
                diagnostics={key: value for key, value in metric.items() if key != "aod"},
                metadata={},
            )
            chunks[(domain, bit_width)].validate(num_experts)
    return chunks, qdq_rows


def compute_gradient_layer_costs(
    spec: MoeLayerSpec,
    layout: ExpertWeightLayout,
    replay_captures: Mapping[str, ReplayCapture],
    gradient_captures: Mapping[str, GradientCapture],
    domains: Sequence[str],
    bit_widths: Sequence[int],
    *,
    num_examples: int,
    group_size: int = 128,
    chunk_size: int = 512,
    requested_keys: set[tuple[str, int]] | None = None,
) -> tuple[dict[tuple[str, int], CostChunk], list[dict[str, Any]]]:
    num_experts = spec.num_experts
    bits = [int(value) for value in bit_widths]
    all_keys = {(domain, bit) for domain in domains for bit in bits}
    requested = all_keys if requested_keys is None else set(requested_keys)
    if not requested or not requested.issubset(all_keys):
        raise ValueError("requested_keys must be a nonempty subset of domain/bit pairs")
    active_domains = [
        domain for domain in domains if any((domain, bit) in requested for bit in bits)
    ]
    active_bits = [
        bit for bit in bits if any((domain, bit) in requested for domain in domains)
    ]
    values = {
        (domain, bit): {
            "gqs": np.zeros(num_experts, dtype=np.float64),
            "gqs2": np.zeros(num_experts, dtype=np.float64),
        }
        for domain, bit in requested
    }
    counts = {
        domain: np.zeros(num_experts, dtype=np.int64) for domain in active_domains
    }
    qdq_rows: list[dict[str, Any]] = []
    layer_before = layout.all_fingerprints()
    for expert_id in range(num_experts):
        routed: dict[str, tuple[Any, Any, np.ndarray, Any]] = {}
        baseline: dict[str, Any] = {}
        for domain in active_domains:
            replay = replay_captures[domain]
            gradient = gradient_captures[domain]
            hidden, gates, examples = selected_routes(replay, expert_id)
            rows, _ = replay.route_rows(expert_id)
            routed[domain] = (hidden, gates, examples, gradient.gradients[rows])
            counts[domain][expert_id] = len(examples)
            baseline[domain] = replay_expert_outputs(
                spec, expert_id, hidden, chunk_size=chunk_size
            )
        for bit_width in active_bits:
            if bit_width == 16:
                continue
            requested_domains = [
                domain
                for domain in active_domains
                if (domain, bit_width) in requested
            ]
            context = ReversibleExpertQuantization(
                layout,
                expert_id,
                bit_width,
                group_size,
                verify_unrelated_experts=False,
            )
            with context:
                for domain in requested_domains:
                    hidden, gates, examples, gradients = routed[domain]
                    quantized = replay_expert_outputs(
                        spec, expert_id, hidden, chunk_size=chunk_size
                    )
                    gqs, gqs2, _ = calculate_gqs(
                        baseline[domain],
                        quantized,
                        gates,
                        gradients,
                        examples,
                        num_examples=num_examples,
                    )
                    values[(domain, bit_width)]["gqs"][expert_id] = gqs
                    values[(domain, bit_width)]["gqs2"][expert_id] = gqs2
            diagnostics = context.diagnostics()
            qdq_rows.append(
                {
                    "layer": spec.model_layer_index,
                    "expert_id": expert_id,
                    "bit_width": bit_width,
                    "original_expert_fingerprint": diagnostics[
                        "original_expert_fingerprint"
                    ],
                    "quantized_expert_fingerprint": diagnostics[
                        "quantized_expert_fingerprint"
                    ],
                    "exact_restoration_verified": diagnostics[
                        "exact_restoration_verified"
                    ],
                }
            )
    if layer_before != layout.all_fingerprints():
        raise RuntimeError(
            f"Model weights changed while building layer {spec.model_layer_index} GQS"
        )
    chunks: dict[tuple[str, int], CostChunk] = {}
    for domain in domains:
        for bit_width in bits:
            if (domain, bit_width) not in requested:
                continue
            chunks[(domain, bit_width)] = CostChunk(
                cost=values[(domain, bit_width)]["gqs"],
                route_counts=counts[domain].copy(),
                unobserved=counts[domain] == 0,
                diagnostics={"gqs2": values[(domain, bit_width)]["gqs2"]},
                metadata={},
            )
            chunks[(domain, bit_width)].validate(num_experts)
    return chunks, qdq_rows


def build_full_cost_matrix(
    layer_specs: Sequence[MoeLayerSpec],
    layouts: Mapping[int, ExpertWeightLayout],
    replay_capture_dir: Path,
    output_dir: Path,
    domains: Sequence[str],
    *,
    selected_surrogate: str,
    matrix_fingerprint: str,
    capture_fingerprint: str,
    gradient_capture_dir: Path | None = None,
    gradient_fingerprint: str | None = None,
    bit_widths: Sequence[int] = DEFAULT_BIT_WIDTHS,
    group_size: int = 128,
    chunk_size: int = 512,
    num_examples: int = 100,
    resume: bool = True,
) -> dict[str, Any]:
    bits = tuple(int(value) for value in bit_widths)
    if bits != DEFAULT_BIT_WIDTHS:
        raise ValueError("Full Stage-2A precision order must be [3, 4, 8, 16]")
    if selected_surrogate not in ("AOD", "GQS"):
        raise ValueError("Full matrix must use exactly AOD or GQS")
    if selected_surrogate == "GQS" and (
        gradient_capture_dir is None or gradient_fingerprint is None
    ):
        raise ValueError("GQS full matrix requires aligned gradient captures")
    sorted_specs = sorted(layer_specs, key=lambda spec: spec.model_layer_index)
    num_layers = len(sorted_specs)
    num_experts = sorted_specs[0].num_experts
    chunks_root = output_dir / "chunks" / selected_surrogate.lower()
    qdq_manifest: list[dict[str, Any]] = []
    for spec in sorted_specs:
        expected_by_chunk = {
            (domain, bit): {
                "matrix_fingerprint": matrix_fingerprint,
                "capture_fingerprint": capture_fingerprint,
                "selected_surrogate": selected_surrogate,
                "domain": domain,
                "layer": spec.model_layer_index,
                "bit_width": bit,
                "group_size": group_size,
                **(
                    {"gradient_fingerprint": gradient_fingerprint}
                    if selected_surrogate == "GQS"
                    else {}
                ),
            }
            for domain in domains
            for bit in bits
        }
        valid: set[tuple[str, int]] = set()
        if resume:
            for key, expected in expected_by_chunk.items():
                domain, bit = key
                path = (
                    chunks_root
                    / domain
                    / f"layer_{spec.model_layer_index:02d}"
                    / f"bit_{bit}.npz"
                )
                try:
                    load_cost_chunk(
                        path, expected_metadata=expected, num_experts=num_experts
                    )
                    valid.add(key)
                except FileNotFoundError:
                    pass
        if len(valid) == len(expected_by_chunk):
            print(
                f"[cost-matrix] resume complete layer {spec.model_layer_index}",
                flush=True,
            )
            continue
        missing_keys = set(expected_by_chunk) - valid
        replay = {
            domain: ReplayCapture.load(
                replay_capture_dir
                / domain
                / f"layer_{spec.model_layer_index:02d}.npz",
                expected_metadata={"capture_fingerprint": capture_fingerprint},
            )
            for domain in domains
        }
        if selected_surrogate == "AOD":
            computed, qdq_rows = compute_activation_layer_costs(
                spec,
                layouts[spec.model_layer_index],
                replay,
                domains,
                bits,
                group_size=group_size,
                chunk_size=chunk_size,
                requested_keys=missing_keys,
            )
        else:
            assert gradient_capture_dir is not None
            gradients = {
                domain: GradientCapture.load(
                    gradient_capture_dir
                    / domain
                    / f"layer_{spec.model_layer_index:02d}.npz",
                    expected_metadata={
                        "capture_fingerprint": capture_fingerprint,
                        "gradient_fingerprint": gradient_fingerprint,
                    },
                )
                for domain in domains
            }
            computed, qdq_rows = compute_gradient_layer_costs(
                spec,
                layouts[spec.model_layer_index],
                replay,
                gradients,
                domains,
                bits,
                num_examples=num_examples,
                group_size=group_size,
                chunk_size=chunk_size,
                requested_keys=missing_keys,
            )
        qdq_manifest.extend(qdq_rows)
        for key, chunk in computed.items():
            if key in valid:
                continue
            domain, bit = key
            path = (
                chunks_root
                / domain
                / f"layer_{spec.model_layer_index:02d}"
                / f"bit_{bit}.npz"
            )
            save_cost_chunk(path, chunk, expected_by_chunk[key])
        print(f"[cost-matrix] saved layer {spec.model_layer_index}", flush=True)

    cost = np.zeros((num_layers, num_experts, len(domains), len(bits)), dtype=np.float64)
    route_counts = np.zeros((num_layers, num_experts, len(domains)), dtype=np.int64)
    unobserved = np.zeros_like(route_counts, dtype=np.bool_)
    chunk_hashes: dict[str, str] = {}
    for layer_ordinal, spec in enumerate(sorted_specs):
        for domain_index, domain in enumerate(domains):
            reference_counts: np.ndarray | None = None
            for bit_index, bit in enumerate(bits):
                expected = {
                    "matrix_fingerprint": matrix_fingerprint,
                    "capture_fingerprint": capture_fingerprint,
                    "selected_surrogate": selected_surrogate,
                    "domain": domain,
                    "layer": spec.model_layer_index,
                    "bit_width": bit,
                    "group_size": group_size,
                    **(
                        {"gradient_fingerprint": gradient_fingerprint}
                        if selected_surrogate == "GQS"
                        else {}
                    ),
                }
                path = (
                    chunks_root
                    / domain
                    / f"layer_{spec.model_layer_index:02d}"
                    / f"bit_{bit}.npz"
                )
                chunk = load_cost_chunk(
                    path, expected_metadata=expected, num_experts=num_experts
                )
                cost[layer_ordinal, :, domain_index, bit_index] = chunk.cost
                if reference_counts is None:
                    reference_counts = chunk.route_counts
                    route_counts[layer_ordinal, :, domain_index] = chunk.route_counts
                    unobserved[layer_ordinal, :, domain_index] = chunk.unobserved
                elif not np.array_equal(reference_counts, chunk.route_counts):
                    raise RuntimeError("Route counts differ across precision chunks")
                chunk_hashes[str(path.relative_to(output_dir))] = file_sha256(path)
    validation = validate_full_cost_matrix(cost, route_counts, unobserved, bits)
    return {
        "cost": cost,
        "route_counts": route_counts,
        "unobserved": unobserved,
        "validation": validation,
        "chunk_hashes": chunk_hashes,
        "qdq_manifest": qdq_manifest,
        "layer_indices": np.asarray(
            [spec.model_layer_index for spec in sorted_specs], dtype=np.int16
        ),
        "expert_ids": np.arange(num_experts, dtype=np.int16),
        "domain_names": np.asarray(domains, dtype=np.str_),
        "bit_widths": np.asarray(bits, dtype=np.int8),
    }


def validate_full_cost_matrix(
    cost: np.ndarray,
    route_counts: np.ndarray,
    unobserved: np.ndarray,
    bit_widths: Sequence[int],
) -> dict[str, Any]:
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("Cost matrix must have [layer, expert, domain, precision] shape")
    if route_counts.shape != values.shape[:3] or unobserved.shape != values.shape[:3]:
        raise ValueError("Route coverage matrices have the wrong shape")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Full cost matrix contains invalid values")
    bits = list(int(value) for value in bit_widths)
    if bits != list(DEFAULT_BIT_WIDTHS):
        raise ValueError("Full matrix bit-width metadata is invalid")
    if not np.all(values[..., bits.index(16)] == 0):
        raise ValueError("16-bit reference costs are not exactly zero")
    if np.any(route_counts < 0) or not np.array_equal(unobserved, route_counts == 0):
        raise ValueError("Full route coverage matrix is invalid")
    violations: list[dict[str, Any]] = []
    for layer, expert, domain in np.ndindex(values.shape[:3]):
        row = values[layer, expert, domain]
        if not (row[0] >= row[1] >= row[2] >= row[3]):
            if len(violations) < 100:
                violations.append(
                    {
                        "layer_ordinal": layer,
                        "expert_id": expert,
                        "domain_index": domain,
                        "cost_3": row[0],
                        "cost_4": row[1],
                        "cost_8": row[2],
                        "cost_16": row[3],
                    }
                )
    violation_count = int(
        np.count_nonzero(
            (values[..., 0] < values[..., 1])
            | (values[..., 1] < values[..., 2])
            | (values[..., 2] < values[..., 3])
        )
    )
    return {
        "passed": True,
        "shape": list(values.shape),
        "all_costs_finite": True,
        "all_costs_nonnegative": True,
        "sixteen_bit_cost_exactly_zero": True,
        "route_counts_nonnegative": True,
        "unobserved_flags_match_zero_routes": True,
        "bit_widths": bits,
        "monotonicity_is_diagnostic_not_forced": True,
        "monotonicity_violation_count": violation_count,
        "monotonicity_violations_first_100": violations,
    }


def build_memory_matrix(
    layer_specs: Sequence[MoeLayerSpec],
    layouts: Mapping[int, ExpertWeightLayout],
    bit_widths: Sequence[int] = DEFAULT_BIT_WIDTHS,
    *,
    group_size: int = 128,
) -> dict[str, np.ndarray]:
    specs = sorted(layer_specs, key=lambda spec: spec.model_layer_index)
    bits = [int(value) for value in bit_widths]
    shape = (len(specs), specs[0].num_experts, len(bits))
    arrays = {
        "projected_bytes": np.zeros(shape, dtype=np.int64),
        "effective_bits_per_weight": np.zeros(shape, dtype=np.float64),
        "weight_count": np.zeros(shape, dtype=np.int64),
        "number_of_groups": np.zeros(shape, dtype=np.int64),
        "quantized_weight_payload_bits": np.zeros(shape, dtype=np.int64),
        "scale_storage_bits": np.zeros(shape, dtype=np.int64),
        "other_required_metadata_bits": np.zeros(shape, dtype=np.int64),
        "bf16_projected_bytes": np.zeros(shape, dtype=np.int64),
    }
    for layer_index, spec in enumerate(specs):
        layout = layouts[spec.model_layer_index]
        for expert_id in range(spec.num_experts):
            shapes = layout.tensor_shapes(expert_id)
            for bit_index, bit_width in enumerate(bits):
                accounting = projected_expert_storage(
                    shapes, bits=bit_width, group_size=group_size
                )
                for key in arrays:
                    arrays[key][layer_index, expert_id, bit_index] = accounting[key]
    if not np.all(arrays["effective_bits_per_weight"][..., bits.index(16)] == 16.0):
        raise RuntimeError("BF16 memory accounting is not exactly 16 bits/weight")
    return arrays


def write_full_matrix_outputs(
    output_dir: Path,
    matrix: Mapping[str, Any],
    memory: Mapping[str, np.ndarray],
    *,
    selected_surrogate: str,
    matrix_fingerprint: str,
    group_size: int,
) -> dict[str, Any]:
    cost_path = output_dir / "full_cost_matrix.npz"
    atomic_save_npz(
        cost_path,
        cost=matrix["cost"],
        layer_indices=matrix["layer_indices"],
        expert_ids=matrix["expert_ids"],
        domain_names=matrix["domain_names"],
        bit_widths=matrix["bit_widths"],
        selected_surrogate=np.asarray(selected_surrogate, dtype=np.str_),
    )
    route_path = output_dir / "route_coverage_matrix.npz"
    atomic_save_npz(
        route_path,
        route_counts=matrix["route_counts"],
        unobserved=matrix["unobserved"],
        layer_indices=matrix["layer_indices"],
        expert_ids=matrix["expert_ids"],
        domain_names=matrix["domain_names"],
    )
    memory_path = output_dir / "memory_matrix.npz"
    atomic_save_npz(
        memory_path,
        **memory,
        layer_indices=matrix["layer_indices"],
        expert_ids=matrix["expert_ids"],
        bit_widths=matrix["bit_widths"],
    )
    csv_rows = []
    for layer_index, layer in enumerate(matrix["layer_indices"]):
        for expert_index, expert_id in enumerate(matrix["expert_ids"]):
            for domain_index, domain in enumerate(matrix["domain_names"]):
                for bit_index, bit_width in enumerate(matrix["bit_widths"]):
                    csv_rows.append(
                        {
                            "layer": int(layer),
                            "expert_id": int(expert_id),
                            "domain": str(domain),
                            "bit_width": int(bit_width),
                            "cost": matrix["cost"][
                                layer_index, expert_index, domain_index, bit_index
                            ],
                            "route_count": matrix["route_counts"][
                                layer_index, expert_index, domain_index
                            ],
                            "unobserved": matrix["unobserved"][
                                layer_index, expert_index, domain_index
                            ],
                            "projected_bytes": memory["projected_bytes"][
                                layer_index, expert_index, bit_index
                            ],
                            "effective_bits_per_weight": memory[
                                "effective_bits_per_weight"
                            ][layer_index, expert_index, bit_index],
                        }
                    )
    write_csv(
        output_dir / "full_cost_matrix.csv",
        csv_rows,
        [
            "layer",
            "expert_id",
            "domain",
            "bit_width",
            "cost",
            "route_count",
            "unobserved",
            "projected_bytes",
            "effective_bits_per_weight",
        ],
    )
    metadata = {
        "schema_version": COST_MATRIX_SCHEMA_VERSION,
        "matrix_fingerprint": matrix_fingerprint,
        "selected_surrogate": selected_surrogate,
        "cost_definition": selected_surrogate,
        "shape": list(matrix["cost"].shape),
        "axis_order": ["layer", "expert", "domain", "bit_width"],
        "layer_indices": matrix["layer_indices"].tolist(),
        "expert_ids": matrix["expert_ids"].tolist(),
        "domains": matrix["domain_names"].tolist(),
        "bit_widths": matrix["bit_widths"].tolist(),
        "group_size": group_size,
        "sixteen_bit_reference_cost": 0,
        "posthoc_layer_or_expert_renormalization_applied": False,
        "gqs_layer_normalization_applied": False,
        "gqs_scale_policy": (
            "raw per-example mean absolute first-order score retained; deterministic "
            "normalization must be designed before, and independently of, optimizer outcomes"
            if selected_surrogate == "GQS"
            else None
        ),
        "validation": matrix["validation"],
        "route_coverage": {
            "observed_cells": int(np.count_nonzero(~matrix["unobserved"])),
            "unobserved_cells": int(np.count_nonzero(matrix["unobserved"])),
            "zero_route_cost_recorded_as_zero": True,
            "zero_route_is_not_evidence_of_global_irrelevance": True,
        },
        "memory_accounting": {
            "includes_quantized_weight_payload": True,
            "includes_fp16_scales": True,
            "includes_group_count": True,
            "includes_required_metadata": True,
            "symmetric_zero_point_bits": 0,
            "effective_bits_per_weight_by_bit_width": {
                str(int(bit)): sorted(
                    set(
                        float(value)
                        for value in memory["effective_bits_per_weight"][
                            ..., bit_index
                        ].reshape(-1)
                    )
                )
                for bit_index, bit in enumerate(matrix["bit_widths"])
            },
        },
        "file_sha256": {
            "full_cost_matrix.npz": file_sha256(cost_path),
            "route_coverage_matrix.npz": file_sha256(route_path),
            "memory_matrix.npz": file_sha256(memory_path),
            "full_cost_matrix.csv": file_sha256(output_dir / "full_cost_matrix.csv"),
        },
        "chunk_sha256": matrix["chunk_hashes"],
    }
    atomic_write_json(output_dir / "full_matrix_metadata.json", metadata)
    return metadata


def verify_pilot_reproduction(
    cost: np.ndarray,
    layer_indices: np.ndarray,
    expert_ids: np.ndarray,
    domain_names: np.ndarray,
    bit_widths: np.ndarray,
    pilot_layers: np.ndarray,
    pilot_expert_ids: np.ndarray,
    pilot_values: np.ndarray,
    *,
    atol: float = 1e-15,
) -> dict[str, Any]:
    bit_index = list(bit_widths.astype(int)).index(4)
    extracted = np.empty_like(pilot_values, dtype=np.float64)
    layer_lookup = {int(value): index for index, value in enumerate(layer_indices)}
    expert_lookup = {int(value): index for index, value in enumerate(expert_ids)}
    for index, (layer, expert_id) in enumerate(
        zip(pilot_layers, pilot_expert_ids, strict=True)
    ):
        extracted[index] = cost[
            layer_lookup[int(layer)], expert_lookup[int(expert_id)], :, bit_index
        ]
    difference = np.abs(extracted - np.asarray(pilot_values, dtype=np.float64))
    passed = bool(np.all(difference <= atol))
    return {
        "passed": passed,
        "pilot_experts": len(pilot_layers),
        "observations": int(extracted.size),
        "bit_width": 4,
        "domain_order": domain_names.tolist(),
        "atol": atol,
        "maximum_absolute_difference": float(difference.max(initial=0.0)),
        "exact_array_equal": bool(np.array_equal(extracted, pilot_values)),
        "extracted_values": extracted.tolist(),
    }


def create_full_cost_map_figures(
    output_dir: Path,
    cost: np.ndarray,
    domain_names: Sequence[str],
    bit_widths: Sequence[int],
    *,
    bit_width: int = 4,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bit_index = list(int(value) for value in bit_widths).index(bit_width)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for domain_index, domain in enumerate(domain_names):
        figure, axis = plt.subplots(figsize=(12.0, 4.8))
        image = axis.imshow(
            cost[:, :, domain_index, bit_index],
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
        )
        axis.set_xlabel("Expert ID")
        axis.set_ylabel("MoE layer")
        axis.set_title(f"{domain.title()} {bit_width}-bit predicted cost map")
        figure.colorbar(image, ax=axis, label="Predicted cost")
        figure.tight_layout()
        base = figure_dir / f"figure_5_full_cost_map_{domain}_bit_{bit_width}"
        png = base.with_suffix(".png")
        pdf = base.with_suffix(".pdf")
        figure.savefig(png, dpi=180, bbox_inches="tight")
        figure.savefig(pdf, bbox_inches="tight")
        plt.close(figure)
        outputs.extend([png, pdf])
    return outputs
