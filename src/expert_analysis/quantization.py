from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .io_utils import atomic_write_json, read_json
from .masking import LossStatistics
from .modeling import MoeLayerSpec


SCALE_STORAGE_DTYPE = torch.float16
SCALE_STORAGE_BITS = 16


@dataclass(frozen=True)
class FakeQuantizedTensor:
    """A dequantized tensor plus the scale and distortion information used to make it."""

    dequantized: torch.Tensor
    scales: torch.Tensor
    bits: int
    group_size: int
    number_of_groups: int
    squared_error: float
    squared_norm: float

    @property
    def relative_squared_error(self) -> float:
        if self.squared_norm == 0.0:
            return 0.0
        return self.squared_error / self.squared_norm


def symmetric_groupwise_qdq(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> FakeQuantizedTensor:
    """Apply deterministic symmetric group-wise weight QDQ along the final dimension.

    The final dimension is the input-feature dimension for both ``nn.Linear`` weights
    and OLMoE's tensorized expert slices. Scales are calculated in FP32, rounded to
    FP16 to match the projected storage format, and expanded in FP32 for QDQ. The
    returned dequantized tensor always has the input tensor's dtype and shape.
    """

    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("Expert fake quantization requires a floating-point tensor")
    if weight.ndim < 2:
        raise ValueError("Expert weights must have at least output and input dimensions")
    _validate_bits(bits)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Expert weight contains non-finite values")

    rows = math.prod(weight.shape[:-1])
    input_features = int(weight.shape[-1])
    if bits == 16:
        dequantized = weight.detach().clone()
        scales = torch.empty(
            (rows, 0), dtype=SCALE_STORAGE_DTYPE, device=weight.device
        )
        squared_norm = _stable_square_sum(weight.detach().float())
        return FakeQuantizedTensor(
            dequantized=dequantized,
            scales=scales,
            bits=bits,
            group_size=group_size,
            number_of_groups=0,
            squared_error=0.0,
            squared_norm=squared_norm,
        )

    qmax = (1 << (bits - 1)) - 1
    source = weight.detach().float().reshape(rows, input_features)
    restored = torch.empty_like(source)
    group_count_per_row = math.ceil(input_features / group_size)
    stored_scales = torch.empty(
        (rows, group_count_per_row),
        dtype=SCALE_STORAGE_DTYPE,
        device=weight.device,
    )

    for group_index, start in enumerate(range(0, input_features, group_size)):
        stop = min(start + group_size, input_features)
        group = source[:, start:stop]
        exact_scale = group.abs().amax(dim=1, keepdim=True) / float(qmax)
        storage_scale = exact_scale.to(dtype=SCALE_STORAGE_DTYPE)
        if bool(torch.isinf(storage_scale).any()):
            raise ValueError("An FP16 quantization scale overflowed")
        underflow = (exact_scale > 0) & (storage_scale == 0)
        if bool(underflow.any()):
            # Smallest positive FP16 subnormal. This remains deterministic and avoids
            # dividing a nonzero group by zero after scale storage.
            storage_scale = torch.where(
                underflow,
                torch.full_like(storage_scale, 2.0**-24),
                storage_scale,
            )
        stored_scales[:, group_index] = storage_scale.squeeze(1)
        qdq_scale = storage_scale.float()
        safe_scale = torch.where(qdq_scale == 0, torch.ones_like(qdq_scale), qdq_scale)
        quantized = torch.clamp(torch.round(group / safe_scale), -qmax, qmax)
        dequantized_group = quantized * qdq_scale
        # This also explicitly keeps an all-zero group at exactly zero.
        restored[:, start:stop] = torch.where(
            qdq_scale == 0, torch.zeros_like(dequantized_group), dequantized_group
        )

    dequantized = restored.reshape(weight.shape).to(dtype=weight.dtype)
    if not bool(torch.isfinite(dequantized).all()):
        raise RuntimeError("Expert fake quantization produced non-finite values")
    # Distortion describes the exact tensor installed in the model, including the
    # final cast back to the model's storage dtype.
    difference = source - dequantized.float().reshape(rows, input_features)
    squared_error = _stable_square_sum(difference)
    squared_norm = _stable_square_sum(source)
    return FakeQuantizedTensor(
        dequantized=dequantized,
        scales=stored_scales,
        bits=bits,
        group_size=group_size,
        number_of_groups=rows * group_count_per_row,
        squared_error=squared_error,
        squared_norm=squared_norm,
    )


def projected_expert_storage(
    tensor_shapes: Sequence[Sequence[int]], bits: int, group_size: int = 128
) -> dict[str, Any]:
    """Return exact packed-payload and FP16-scale accounting for one expert."""

    _validate_bits(bits)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    normalized_shapes = [tuple(int(value) for value in shape) for shape in tensor_shapes]
    if not normalized_shapes or any(len(shape) < 2 for shape in normalized_shapes):
        raise ValueError("At least one matrix-shaped expert tensor is required")
    if any(any(value <= 0 for value in shape) for shape in normalized_shapes):
        raise ValueError("Expert tensor dimensions must be positive")

    tensor_rows: list[dict[str, Any]] = []
    total_weights = 0
    total_groups = 0
    total_weight_bits = 0
    total_weight_bytes = 0
    for shape in normalized_shapes:
        weights = math.prod(shape)
        rows = math.prod(shape[:-1])
        groups = 0 if bits == 16 else rows * math.ceil(shape[-1] / group_size)
        weight_bits = weights * bits
        packed_bytes = math.ceil(weight_bits / 8)
        total_weights += weights
        total_groups += groups
        total_weight_bits += weight_bits
        total_weight_bytes += packed_bytes
        tensor_rows.append(
            {
                "shape": list(shape),
                "weights": weights,
                "input_features": shape[-1],
                "groups": groups,
                "quantized_weight_bits": weight_bits,
                "packed_weight_bytes": packed_bytes,
                "packing_padding_bits": packed_bytes * 8 - weight_bits,
            }
        )

    scale_bits = total_groups * SCALE_STORAGE_BITS
    scale_bytes = total_groups * (SCALE_STORAGE_BITS // 8)
    metadata_bits = 0
    metadata_bytes = 0
    projected_bytes = total_weight_bytes + scale_bytes + metadata_bytes
    bf16_bytes = total_weights * 2
    return {
        "raw_nominal_bit_width": bits,
        "group_size": group_size if bits < 16 else None,
        "weight_count": total_weights,
        "number_of_groups": total_groups,
        "quantized_weight_payload_bits": total_weight_bits,
        "quantized_weight_packed_bytes": total_weight_bytes,
        "weight_packing_padding_bits": total_weight_bytes * 8 - total_weight_bits,
        "scale_storage_dtype": "float16" if bits < 16 else None,
        "scale_storage_bits_per_group": SCALE_STORAGE_BITS if bits < 16 else 0,
        "scale_storage_bits": scale_bits,
        "scale_storage_bytes": scale_bytes,
        "zero_point_storage_bits": 0,
        "zero_point_storage_bytes": 0,
        "other_required_metadata_bits": metadata_bits,
        "other_required_metadata_bytes": metadata_bytes,
        "metadata_rationale": (
            "symmetric quantization needs no zero point; bit width, group size, "
            "tensor shape, and grouping axis are fixed globally by the pilot format"
            if bits < 16
            else "BF16 stores weights directly and needs no quantization metadata"
        ),
        "projected_bytes": projected_bytes,
        "bf16_projected_bytes": bf16_bytes,
        "effective_bits_per_weight": projected_bytes * 8.0 / total_weights,
        "compression_ratio_vs_bf16": bf16_bytes / projected_bytes,
        "tensor_accounting": tensor_rows,
        "is_projected_not_measured_runtime_memory": True,
    }


@dataclass(frozen=True)
class ExpertTensorReference:
    name: str
    parameter: nn.Parameter
    expert_axis: int | None
    expert_id: int

    def view(self) -> torch.Tensor:
        if self.expert_axis is None:
            return self.parameter
        return self.parameter.select(self.expert_axis, self.expert_id)


class ExpertWeightLayout:
    """Structural access to expert FFN matrices without parameter-name assumptions."""

    def __init__(self, spec: MoeLayerSpec) -> None:
        self.spec = spec
        self.num_experts = int(spec.num_experts)
        self._references = self._discover_references()
        if set(self._references) != set(range(self.num_experts)):
            raise RuntimeError("Could not structurally resolve every expert in the layer")
        expected_shapes = self.tensor_shapes(0)
        if not expected_shapes:
            raise RuntimeError("No expert FFN weight matrices were discovered")
        for expert_id in range(1, self.num_experts):
            if self.tensor_shapes(expert_id) != expected_shapes:
                raise RuntimeError("Experts in one layer do not share the same weight layout")

    def references(self, expert_id: int) -> tuple[ExpertTensorReference, ...]:
        self._validate_expert_id(expert_id)
        return self._references[expert_id]

    def tensor_shapes(self, expert_id: int) -> list[tuple[int, ...]]:
        return [tuple(reference.view().shape) for reference in self.references(expert_id)]

    def fingerprint(self, expert_id: int) -> str:
        digest = hashlib.sha256()
        digest.update(b"olmoe-expert-weight-fingerprint-v1\0")
        for reference in self.references(expert_id):
            tensor = reference.view().detach()
            digest.update(reference.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            _update_digest_with_tensor(digest, tensor)
        return digest.hexdigest()

    def all_fingerprints(self) -> dict[int, str]:
        return {
            expert_id: self.fingerprint(expert_id)
            for expert_id in range(self.num_experts)
        }

    def metadata(self) -> dict[str, Any]:
        references = self.references(0)
        return {
            "layer": self.spec.model_layer_index,
            "layer_ordinal": self.spec.ordinal,
            "expert_container_class": self.spec.experts.__class__.__name__,
            "contribution_backend": self.spec.contribution_backend,
            "num_experts": self.num_experts,
            "tensor_count_per_expert": len(references),
            "tensors": [
                {
                    "structural_name": reference.name,
                    "expert_axis": reference.expert_axis,
                    "expert_slice_shape": list(reference.view().shape),
                    "dtype": str(reference.view().dtype).replace("torch.", ""),
                }
                for reference in references
            ],
            "grouping_axis_within_expert_slice": -1,
            "grouping_axis_semantics": "input_feature_dimension",
        }

    def _discover_references(self) -> dict[int, tuple[ExpertTensorReference, ...]]:
        tensorized = self._discover_tensorized()
        if tensorized is not None:
            return tensorized
        module_list = self._discover_module_list()
        if module_list is not None:
            return module_list
        raise RuntimeError(
            f"Unsupported expert storage in {self.spec.experts.__class__.__name__}; "
            "expected tensorized [expert, output, input] parameters or a module list"
        )

    def _discover_tensorized(
        self,
    ) -> dict[int, tuple[ExpertTensorReference, ...]] | None:
        candidates: list[tuple[str, nn.Parameter, int]] = []
        for name, parameter in self.spec.experts.named_parameters(recurse=True):
            matching_axes = [
                axis
                for axis, size in enumerate(parameter.shape)
                if size == self.num_experts
            ]
            if len(matching_axes) != 1:
                continue
            expert_axis = matching_axes[0]
            slice_shape = parameter.shape[:expert_axis] + parameter.shape[expert_axis + 1 :]
            # Weight-only QDQ deliberately excludes vector biases and scalar state.
            if len(slice_shape) < 2:
                continue
            candidates.append((name, parameter, expert_axis))
        if not candidates:
            return None
        output: dict[int, tuple[ExpertTensorReference, ...]] = {}
        for expert_id in range(self.num_experts):
            output[expert_id] = tuple(
                ExpertTensorReference(name, parameter, axis, expert_id)
                for name, parameter, axis in sorted(candidates, key=lambda item: item[0])
            )
        return output

    def _discover_module_list(
        self,
    ) -> dict[int, tuple[ExpertTensorReference, ...]] | None:
        container: nn.ModuleList | nn.ModuleDict | None = None
        prefix = ""
        if isinstance(self.spec.experts, (nn.ModuleList, nn.ModuleDict)):
            container = self.spec.experts
        else:
            for child_name, child in self.spec.experts.named_children():
                if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    if container is not None:
                        raise RuntimeError("Expert container has multiple module-list children")
                    container = child
                    prefix = f"{child_name}."
        if container is None or len(container) != self.num_experts:
            return None
        modules = (
            list(container.values())
            if isinstance(container, nn.ModuleDict)
            else list(container)
        )
        output: dict[int, tuple[ExpertTensorReference, ...]] = {}
        for expert_id, module in enumerate(modules):
            references = [
                ExpertTensorReference(
                    f"{prefix}{expert_id}.{name}", parameter, None, expert_id
                )
                for name, parameter in module.named_parameters(recurse=True)
                if parameter.ndim >= 2
            ]
            if not references:
                raise RuntimeError(f"Expert {expert_id} contains no matrix-shaped weights")
            output[expert_id] = tuple(sorted(references, key=lambda item: item.name))
        return output

    def _validate_expert_id(self, expert_id: int) -> None:
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f"Expert {expert_id} is outside layer {self.spec.model_layer_index}'s "
                f"0..{self.num_experts - 1} range"
            )


class ReversibleExpertQuantization:
    """Apply one expert's QDQ weights and prove isolation plus exact restoration."""

    def __init__(
        self,
        layout: ExpertWeightLayout,
        expert_id: int,
        bits: int,
        group_size: int = 128,
        verify_unrelated_experts: bool = True,
    ) -> None:
        _validate_bits(bits)
        if group_size < 1:
            raise ValueError("group_size must be positive")
        layout.references(expert_id)
        self.layout = layout
        self.expert_id = expert_id
        self.bits = bits
        self.group_size = group_size
        self.verify_unrelated_experts = verify_unrelated_experts
        self._original_tensors: list[torch.Tensor] = []
        self._fingerprints_before: dict[int, str] = {}
        self._entered = False
        self._restoration_verified = False
        self._hooks_before: int | None = None
        self._hooks_after: int | None = None
        self.original_fingerprint: str | None = None
        self.quantized_fingerprint: str | None = None
        self.distortion: float | None = None
        self.tensor_distortions: list[dict[str, Any]] = []
        self.memory_accounting: dict[str, Any] | None = None

    def __enter__(self) -> "ReversibleExpertQuantization":
        if self._entered:
            raise RuntimeError("A reversible quantization context cannot be entered twice")
        self._entered = True
        self._hooks_before = module_hook_count(self.layout.spec.block)
        references = self.layout.references(self.expert_id)
        self._original_tensors = [reference.view().detach().clone() for reference in references]
        if self.verify_unrelated_experts:
            self._fingerprints_before = self.layout.all_fingerprints()
        else:
            self._fingerprints_before = {
                self.expert_id: self.layout.fingerprint(self.expert_id)
            }
        self.original_fingerprint = self._fingerprints_before[self.expert_id]
        self.memory_accounting = projected_expert_storage(
            self.layout.tensor_shapes(self.expert_id), self.bits, self.group_size
        )

        squared_error = 0.0
        squared_norm = 0.0
        try:
            with torch.no_grad():
                for reference, original in zip(references, self._original_tensors, strict=True):
                    result = symmetric_groupwise_qdq(
                        original, bits=self.bits, group_size=self.group_size
                    )
                    reference.view().copy_(result.dequantized)
                    squared_error += result.squared_error
                    squared_norm += result.squared_norm
                    self.tensor_distortions.append(
                        {
                            "structural_name": reference.name,
                            "shape": list(original.shape),
                            "number_of_groups": result.number_of_groups,
                            "squared_error": result.squared_error,
                            "squared_norm": result.squared_norm,
                            "relative_squared_error": result.relative_squared_error,
                            "scale_dtype": "float16" if self.bits < 16 else None,
                        }
                    )
            self.distortion = 0.0 if squared_norm == 0.0 else squared_error / squared_norm
            self.quantized_fingerprint = self.layout.fingerprint(self.expert_id)
            if self.verify_unrelated_experts:
                after = self.layout.all_fingerprints()
                changed = [
                    expert_id
                    for expert_id, fingerprint in after.items()
                    if expert_id != self.expert_id
                    and fingerprint != self._fingerprints_before[expert_id]
                ]
                if changed:
                    raise RuntimeError(
                        "QDQ modified unrelated experts: "
                        + ", ".join(str(value) for value in changed)
                    )
            if module_hook_count(self.layout.spec.block) != self._hooks_before:
                raise RuntimeError("Expert QDQ unexpectedly changed registered model hooks")
            return self
        except BaseException:
            self._restore_without_validation()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        references = self.layout.references(self.expert_id)
        with torch.no_grad():
            for reference, original in zip(references, self._original_tensors, strict=True):
                reference.view().copy_(original)
        restored = self.layout.all_fingerprints() if self.verify_unrelated_experts else {
            self.expert_id: self.layout.fingerprint(self.expert_id)
        }
        mismatches = [
            expert_id
            for expert_id, fingerprint in restored.items()
            if fingerprint != self._fingerprints_before[expert_id]
        ]
        if mismatches:
            raise RuntimeError(
                "Exact expert-weight restoration failed for experts: "
                + ", ".join(str(value) for value in mismatches)
            )
        for reference, original in zip(references, self._original_tensors, strict=True):
            if not torch.equal(reference.view(), original):
                raise RuntimeError("Restored expert tensor is not bitwise equal to its snapshot")
        self._hooks_after = module_hook_count(self.layout.spec.block)
        if self._hooks_after != self._hooks_before:
            raise RuntimeError("Expert QDQ leaked or removed a registered model hook")
        self._restoration_verified = True
        self._original_tensors.clear()

    @property
    def restoration_verified(self) -> bool:
        return self._restoration_verified

    def diagnostics(self) -> dict[str, Any]:
        if self.original_fingerprint is None or self.quantized_fingerprint is None:
            raise RuntimeError("Quantization diagnostics are unavailable before context entry")
        return {
            "method": "symmetric_groupwise_weight_only_qdq",
            "layer": self.layout.spec.model_layer_index,
            "layer_ordinal": self.layout.spec.ordinal,
            "expert_id": self.expert_id,
            "bits": self.bits,
            "group_size": self.group_size,
            "original_expert_fingerprint": self.original_fingerprint,
            "quantized_expert_fingerprint": self.quantized_fingerprint,
            "quantization_distortion": self.distortion,
            "tensor_distortions": self.tensor_distortions,
            "unrelated_experts_verified_unchanged": self.verify_unrelated_experts,
            "exact_restoration_verified": self._restoration_verified,
            "hooks_before": self._hooks_before,
            "hooks_after": self._hooks_after,
            "memory_accounting": self.memory_accounting,
        }

    def _restore_without_validation(self) -> None:
        if not self._original_tensors:
            return
        with torch.no_grad():
            for reference, original in zip(
                self.layout.references(self.expert_id),
                self._original_tensors,
                strict=True,
            ):
                reference.view().copy_(original)


@dataclass(frozen=True)
class QuantizationCheckpoint:
    statistics: LossStatistics
    metadata: dict[str, Any]
    resumed: bool


def load_or_compute_loss_checkpoint(
    path: Path,
    expected_metadata: Mapping[str, Any],
    expected_token_counts: np.ndarray,
    compute: Callable[[], tuple[LossStatistics, Mapping[str, Any]]],
    resume: bool = True,
) -> QuantizationCheckpoint:
    """Atomically checkpoint one expert/domain/bit-width loss evaluation."""

    metadata_path = path.with_suffix(".metadata.json")
    # A process can be killed between the two atomic writes. Such a half-checkpoint
    # is not resumable, but it is safe to recompute and atomically replace in place.
    if resume and path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        _validate_checkpoint_metadata(metadata, expected_metadata, path)
        statistics = LossStatistics.load(path)
        _validate_checkpoint_tokens(statistics, expected_token_counts, path)
        return QuantizationCheckpoint(statistics, metadata, True)

    statistics, diagnostics = compute()
    _validate_checkpoint_tokens(statistics, expected_token_counts, path)
    statistics.save(path)
    metadata = {
        **dict(expected_metadata),
        "diagnostics": dict(diagnostics),
    }
    atomic_write_json(metadata_path, metadata)
    return QuantizationCheckpoint(statistics, metadata, False)


def module_hook_count(module: nn.Module) -> int:
    total = 0
    for child in module.modules():
        total += len(getattr(child, "_forward_hooks", {}))
        total += len(getattr(child, "_forward_pre_hooks", {}))
        total += len(getattr(child, "_backward_hooks", {}))
    return total


def _validate_checkpoint_metadata(
    observed: Mapping[str, Any], expected: Mapping[str, Any], path: Path
) -> None:
    mismatched = [
        key for key, value in expected.items() if observed.get(key) != value
    ]
    if mismatched:
        raise RuntimeError(
            f"Quantization checkpoint fingerprint mismatch for {path}: "
            + ", ".join(mismatched)
        )


def _validate_checkpoint_tokens(
    statistics: LossStatistics, expected_token_counts: np.ndarray, path: Path
) -> None:
    expected = np.asarray(expected_token_counts, dtype=np.uint32)
    if not np.array_equal(statistics.token_counts, expected):
        raise RuntimeError(f"Quantization checkpoint token geometry mismatch for {path}")
    if len(statistics.loss_sums) != len(expected):
        raise RuntimeError(f"Quantization checkpoint example count mismatch for {path}")


def _validate_bits(bits: int) -> None:
    if bits != 16 and not 2 <= bits < 16:
        raise ValueError("bits must be 16 (identity) or an integer from 2 through 15")


def _stable_square_sum(values: torch.Tensor) -> float:
    if values.device.type == "mps":
        return float(values.detach().cpu().double().square().sum().item())
    return float(values.double().square().sum().item())


def _update_digest_with_tensor(digest: Any, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().contiguous()
    raw = contiguous.view(torch.uint8).cpu().numpy()
    digest.update(memoryview(raw))
