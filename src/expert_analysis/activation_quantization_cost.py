from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .expert_replay import ReplayCapture
from .hooks import run_expert
from .modeling import MoeLayerSpec
from .quantization import ExpertWeightLayout, ReversibleExpertQuantization


ACTIVATION_COST_SCHEMA_VERSION = 1
ACTIVATION_COST_EPSILON = 1e-30


@dataclass(frozen=True)
class PerturbationSums:
    route_count: int
    gated_delta_squared: float
    gated_baseline_squared: float
    ungated_delta_squared: float

    def validate(self) -> None:
        values = np.asarray(
            [
                self.gated_delta_squared,
                self.gated_baseline_squared,
                self.ungated_delta_squared,
            ],
            dtype=np.float64,
        )
        if self.route_count < 0:
            raise ValueError("route_count cannot be negative")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Perturbation sums must be finite and nonnegative")
        if self.route_count == 0 and np.any(values != 0):
            raise ValueError("A zero-route expert cannot have perturbation energy")


@dataclass(frozen=True)
class ActivationSurrogateValues:
    aod: float
    reod: float
    apd: float
    uod: float
    route_count: int
    unobserved: bool


def calculate_perturbation_sums(
    baseline_output: torch.Tensor | np.ndarray,
    quantized_output: torch.Tensor | np.ndarray,
    gate_weights: torch.Tensor | np.ndarray,
) -> PerturbationSums:
    baseline = torch.as_tensor(baseline_output).detach().double()
    quantized = torch.as_tensor(quantized_output).detach().double()
    gates = torch.as_tensor(gate_weights).detach().double().reshape(-1)
    if baseline.ndim != 2 or quantized.shape != baseline.shape:
        raise ValueError("Expert outputs must have matching [route, hidden] shapes")
    if gates.shape != (baseline.shape[0],):
        raise ValueError("One gate weight is required per routed expert output")
    if not bool(torch.isfinite(baseline).all()) or not bool(
        torch.isfinite(quantized).all()
    ):
        raise ValueError("Expert replay produced non-finite outputs")
    if not bool(torch.isfinite(gates).all()) or bool((gates < 0).any()):
        raise ValueError("Routed gate weights are invalid")
    if baseline.shape[0] == 0:
        result = PerturbationSums(0, 0.0, 0.0, 0.0)
        result.validate()
        return result
    delta = quantized - baseline
    gated_delta = delta * gates[:, None]
    gated_baseline = baseline * gates[:, None]
    result = PerturbationSums(
        route_count=int(baseline.shape[0]),
        gated_delta_squared=float(gated_delta.square().sum().item()),
        gated_baseline_squared=float(gated_baseline.square().sum().item()),
        ungated_delta_squared=float(delta.square().sum().item()),
    )
    result.validate()
    return result


def finalize_activation_surrogates(
    sums: PerturbationSums,
    *,
    layer_energy: float,
    domain_token_count: int,
    epsilon: float = ACTIVATION_COST_EPSILON,
) -> ActivationSurrogateValues:
    sums.validate()
    if not np.isfinite(layer_energy) or layer_energy < 0:
        raise ValueError("LayerEnergy must be finite and nonnegative")
    if domain_token_count < 1:
        raise ValueError("A domain must contain at least one measured token")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    values = ActivationSurrogateValues(
        aod=sums.gated_delta_squared / (float(layer_energy) + epsilon),
        reod=sums.gated_delta_squared
        / (sums.gated_baseline_squared + epsilon),
        apd=sums.gated_delta_squared / float(domain_token_count),
        uod=(
            sums.ungated_delta_squared / float(sums.route_count)
            if sums.route_count
            else 0.0
        ),
        route_count=sums.route_count,
        unobserved=sums.route_count == 0,
    )
    metrics = np.asarray(
        [values.aod, values.reod, values.apd, values.uod], dtype=np.float64
    )
    if not np.all(np.isfinite(metrics)) or np.any(metrics < 0):
        raise RuntimeError("Activation-aware surrogate produced an invalid value")
    return values


def selected_routes(
    capture: ReplayCapture, expert_id: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rows, positions = capture.route_rows(expert_id)
    if rows.numel() == 0:
        return (
            torch.empty((0, capture.hidden_size), dtype=capture.hidden_states.dtype),
            torch.empty(0, dtype=torch.float32),
            np.empty(0, dtype=np.int64),
        )
    hidden = capture.hidden_states[rows]
    gates = capture.selected_gate_weights[rows, positions]
    examples = capture.example_indices[rows.numpy()]
    return hidden, gates, examples


def replay_expert_outputs(
    spec: MoeLayerSpec,
    expert_id: int,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if hidden_states.ndim != 2:
        raise ValueError("Replay hidden states must have shape [route, hidden]")
    if hidden_states.shape[0] == 0:
        return torch.empty(
            (0, hidden_states.shape[1]), dtype=torch.float32, device="cpu"
        )
    parameter = next(spec.experts.parameters())
    output: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, hidden_states.shape[0], chunk_size):
            stop = min(start + chunk_size, hidden_states.shape[0])
            chunk = hidden_states[start:stop].to(
                device=parameter.device, dtype=parameter.dtype
            )
            values = run_expert(spec, int(expert_id), chunk)
            if values.shape != chunk.shape:
                raise RuntimeError(
                    f"Expert L{spec.model_layer_index}/E{expert_id} replay returned "
                    f"{tuple(values.shape)} for {tuple(chunk.shape)}"
                )
            if not bool(torch.isfinite(values.float()).all()):
                raise RuntimeError("Expert replay produced non-finite values")
            output.append(values.detach().float().cpu())
    return torch.cat(output, dim=0)


def evaluate_activation_surrogates_for_panel(
    layer_specs: Sequence[MoeLayerSpec],
    layouts: Mapping[int, ExpertWeightLayout],
    capture_dir: Path,
    domains: Sequence[str],
    panel: Sequence[Mapping[str, Any]],
    bit_widths: Sequence[int],
    *,
    capture_fingerprint: str,
    group_size: int = 128,
    chunk_size: int = 512,
    expected_qdq_fingerprints: Mapping[tuple[int, int, int], Mapping[str, str]]
    | None = None,
    verify_unrelated_experts: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Replay the fixed expert panel without rerunning the transformer."""

    if len(set((int(row["layer"]), int(row["expert_id"])) for row in panel)) != len(
        panel
    ):
        raise ValueError("Pilot panel contains duplicate layer/expert interventions")
    bits = [int(value) for value in bit_widths]
    if not bits or len(set(bits)) != len(bits):
        raise ValueError("bit_widths must be nonempty and unique")
    spec_by_layer = {spec.model_layer_index: spec for spec in layer_specs}
    shape = (len(bits), len(panel), len(domains))
    raw = {
        "gated_delta_squared": np.zeros(shape, dtype=np.float64),
        "gated_baseline_squared": np.zeros(shape, dtype=np.float64),
        "ungated_delta_squared": np.zeros(shape, dtype=np.float64),
        "route_counts": np.zeros(shape, dtype=np.int64),
        "layer_energy": np.zeros(shape, dtype=np.float64),
        "domain_token_count": np.zeros(shape, dtype=np.int64),
        "aod": np.zeros(shape, dtype=np.float64),
        "reod": np.zeros(shape, dtype=np.float64),
        "apd": np.zeros(shape, dtype=np.float64),
        "uod": np.zeros(shape, dtype=np.float64),
        "unobserved": np.zeros(shape, dtype=np.bool_),
    }
    qdq_rows: list[dict[str, Any]] = []
    capture_cache: dict[tuple[str, int], ReplayCapture] = {}

    def get_capture(domain: str, layer: int) -> ReplayCapture:
        key = (domain, layer)
        if key not in capture_cache:
            capture_cache[key] = ReplayCapture.load(
                capture_dir / domain / f"layer_{layer:02d}.npz",
                expected_metadata={"capture_fingerprint": capture_fingerprint},
            )
        return capture_cache[key]

    for panel_index, intervention in enumerate(panel):
        layer = int(intervention["layer"])
        expert_id = int(intervention["expert_id"])
        spec = spec_by_layer[layer]
        layout = layouts[layer]
        baseline_outputs: dict[str, torch.Tensor] = {}
        routed: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
        for domain_index, domain in enumerate(domains):
            capture = get_capture(domain, layer)
            hidden, gates, examples = selected_routes(capture, expert_id)
            routed[domain] = (hidden, gates, examples)
            baseline_outputs[domain] = replay_expert_outputs(
                spec, expert_id, hidden, chunk_size=chunk_size
            )
            raw["layer_energy"][:, panel_index, domain_index] = capture.layer_energy
            raw["domain_token_count"][:, panel_index, domain_index] = capture.num_tokens

        for bit_index, bit_width in enumerate(bits):
            if bit_width == 16:
                for domain_index, domain in enumerate(domains):
                    route_count = int(routed[domain][0].shape[0])
                    raw["route_counts"][bit_index, panel_index, domain_index] = route_count
                    raw["unobserved"][bit_index, panel_index, domain_index] = route_count == 0
                qdq_rows.append(
                    {
                        "layer": layer,
                        "expert_id": expert_id,
                        "bit_width": bit_width,
                        "identity_reference": True,
                        "exact_stage1_fingerprint_match": None,
                    }
                )
                continue
            context = ReversibleExpertQuantization(
                layout,
                expert_id,
                bit_width,
                group_size,
                verify_unrelated_experts=verify_unrelated_experts,
            )
            with context:
                expected = (
                    expected_qdq_fingerprints.get((layer, expert_id, bit_width))
                    if expected_qdq_fingerprints is not None
                    else None
                )
                if expected is not None:
                    if context.original_fingerprint != expected.get("original"):
                        raise RuntimeError(
                            f"Stage-1 original expert fingerprint mismatch for "
                            f"L{layer}/E{expert_id}/{bit_width}-bit"
                        )
                    if context.quantized_fingerprint != expected.get("quantized"):
                        raise RuntimeError(
                            f"Stage-1 QDQ fingerprint mismatch for "
                            f"L{layer}/E{expert_id}/{bit_width}-bit"
                        )
                for domain_index, domain in enumerate(domains):
                    hidden, gates, _ = routed[domain]
                    quantized_output = replay_expert_outputs(
                        spec, expert_id, hidden, chunk_size=chunk_size
                    )
                    sums = calculate_perturbation_sums(
                        baseline_outputs[domain], quantized_output, gates
                    )
                    capture = get_capture(domain, layer)
                    values = finalize_activation_surrogates(
                        sums,
                        layer_energy=capture.layer_energy,
                        domain_token_count=capture.num_tokens,
                    )
                    raw["gated_delta_squared"][
                        bit_index, panel_index, domain_index
                    ] = sums.gated_delta_squared
                    raw["gated_baseline_squared"][
                        bit_index, panel_index, domain_index
                    ] = sums.gated_baseline_squared
                    raw["ungated_delta_squared"][
                        bit_index, panel_index, domain_index
                    ] = sums.ungated_delta_squared
                    raw["route_counts"][
                        bit_index, panel_index, domain_index
                    ] = sums.route_count
                    raw["aod"][bit_index, panel_index, domain_index] = values.aod
                    raw["reod"][bit_index, panel_index, domain_index] = values.reod
                    raw["apd"][bit_index, panel_index, domain_index] = values.apd
                    raw["uod"][bit_index, panel_index, domain_index] = values.uod
                    raw["unobserved"][
                        bit_index, panel_index, domain_index
                    ] = values.unobserved
            diagnostics = context.diagnostics()
            expected = (
                expected_qdq_fingerprints.get((layer, expert_id, bit_width))
                if expected_qdq_fingerprints is not None
                else None
            )
            qdq_rows.append(
                {
                    "layer": layer,
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
                    "unrelated_experts_verified_unchanged": diagnostics[
                        "unrelated_experts_verified_unchanged"
                    ],
                    "exact_stage1_fingerprint_match": (
                        diagnostics["original_expert_fingerprint"]
                        == expected.get("original")
                        and diagnostics["quantized_expert_fingerprint"]
                        == expected.get("quantized")
                        if expected is not None
                        else None
                    ),
                    "memory_accounting": diagnostics["memory_accounting"],
                }
            )

    for name, values in raw.items():
        if name == "unobserved":
            continue
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Pilot activation metric {name} contains non-finite values")
        if name not in ("domain_token_count", "route_counts") and np.any(values < 0):
            raise RuntimeError(f"Pilot activation metric {name} contains negative values")
    if not all(bool(row.get("exact_restoration_verified", True)) for row in qdq_rows):
        raise RuntimeError("A pilot expert was not restored exactly after QDQ replay")
    return raw, qdq_rows
