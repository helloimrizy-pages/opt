"""Stage 2B mixed-precision evaluation with verified model-state integrity.

The quantizer is exactly the Stage-1 ``symmetric_groupwise_qdq``; this module
only orchestrates applying it to every expert at its assigned precision (base
3/4-bit or protected 8-bit), evaluating teacher-forced NLL on the held-out
splits, and restoring the verified clean BF16 state between allocations.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .balanced import canonical_sha256
from .controlled import PreparedDomainExamples
from .io_utils import atomic_write_json, read_json
from .masking import LossStatistics, evaluate_next_token_loss, _model_hook_count
from .modeling import ModelBundle, MoeLayerSpec
from .quantization import (
    ExpertWeightLayout,
    load_or_compute_loss_checkpoint,
    module_hook_count,
    symmetric_groupwise_qdq,
)
from .specialist_preservation import STAGE2B_DOMAINS

CUBLAS_WORKSPACE_REQUIRED = (":4096:8", ":16:8")
STAGE1_QDQ_FUNCTION = symmetric_groupwise_qdq


def configure_strict_determinism(
    device_type: str, warn_only: bool = False
) -> dict[str, Any]:
    """Configure the strict deterministic execution preregistered for Stage 2B.

    For CUDA this must run before the first cuBLAS call; the entry scripts set
    ``CUBLAS_WORKSPACE_CONFIG`` before importing torch and this function fails
    if CUDA was initialized without it.
    """

    settings: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device_type": device_type,
        "requested_warn_only": bool(warn_only),
    }
    if device_type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in CUBLAS_WORKSPACE_REQUIRED:
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG must be one of "
                f"{CUBLAS_WORKSPACE_REQUIRED} before any CUDA work; found "
                f"{workspace!r}. Export it before launching Python."
            )
        settings["cublas_workspace_config"] = workspace
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        settings["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        settings["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        settings["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        settings["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    settings["use_deterministic_algorithms"] = True
    settings["deterministic_warn_only"] = bool(warn_only)
    settings["attn_implementation_requested"] = "eager"
    return settings


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    contiguous = tensor.detach().contiguous()
    raw = contiguous.view(torch.uint8).cpu().numpy()
    digest.update(memoryview(raw))
    return digest.hexdigest()


@dataclass
class _LayerState:
    spec: MoeLayerSpec
    layout: ExpertWeightLayout
    parameter_names: list[str]
    parameters: list[torch.nn.Parameter]
    snapshots: list[torch.Tensor]
    clean_hashes: list[str]


@dataclass
class MixedPrecisionExpertManager:
    """Apply and exactly revert a full per-expert bit assignment.

    A CPU snapshot of every expert parameter is taken from the verified clean
    model at construction. Application quantizes each expert in place with the
    Stage-1 QDQ; restoration copies the snapshot back and proves bitwise
    equality. Non-expert parameters are fingerprinted and re-verified after
    every restoration.
    """

    bundle: ModelBundle
    layer_specs: Sequence[MoeLayerSpec]
    group_size: int = 128
    _layers: list[_LayerState] = field(default_factory=list, init=False)
    _non_expert: list[tuple[str, torch.nn.Parameter, str]] = field(
        default_factory=list, init=False
    )
    _state: str = field(default="clean", init=False)
    _hooks_reference: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        expert_parameter_ids: set[int] = set()
        for spec in self.layer_specs:
            layout = ExpertWeightLayout(spec)
            names: list[str] = []
            parameters: list[torch.nn.Parameter] = []
            seen: set[int] = set()
            for expert_id in range(layout.num_experts):
                for reference in layout.references(expert_id):
                    if id(reference.parameter) in seen:
                        continue
                    seen.add(id(reference.parameter))
                    names.append(reference.name)
                    parameters.append(reference.parameter)
            snapshots = [
                parameter.detach().to("cpu", copy=True) for parameter in parameters
            ]
            hashes = [_tensor_sha256(snapshot) for snapshot in snapshots]
            self._layers.append(
                _LayerState(spec, layout, names, parameters, snapshots, hashes)
            )
            expert_parameter_ids.update(seen)
        for name, parameter in self.bundle.model.named_parameters():
            if id(parameter) not in expert_parameter_ids:
                self._non_expert.append((name, parameter, _tensor_sha256(parameter)))
        if not self._non_expert:
            raise RuntimeError("No non-expert parameters were found to fingerprint")
        self._hooks_reference = _model_hook_count(self.bundle.model)

    @property
    def state(self) -> str:
        return self._state

    def clean_fingerprints(self) -> dict[str, str]:
        output: dict[str, str] = {}
        for layer in self._layers:
            for name, digest in zip(layer.parameter_names, layer.clean_hashes, strict=True):
                output[f"layer{layer.spec.model_layer_index}.{name}"] = digest
        return output

    def expert_state_sha256(self) -> str:
        return canonical_sha256(self.clean_fingerprints())

    def verify_clean(self) -> None:
        for layer in self._layers:
            for name, parameter, snapshot in zip(
                layer.parameter_names, layer.parameters, layer.snapshots, strict=True
            ):
                if not torch.equal(parameter.detach().cpu(), snapshot):
                    raise RuntimeError(
                        f"Layer {layer.spec.model_layer_index} parameter {name} does "
                        "not match the clean BF16 snapshot"
                    )
        self.verify_non_expert_integrity()
        self._state = "clean"

    def verify_non_expert_integrity(self) -> None:
        for name, parameter, digest in self._non_expert:
            if _tensor_sha256(parameter) != digest:
                raise RuntimeError(f"Non-expert parameter {name} was modified")
        if _model_hook_count(self.bundle.model) != self._hooks_reference:
            raise RuntimeError("Model hook count changed during mixed-precision work")

    def apply_allocation(
        self,
        bits_matrix: np.ndarray,
        verification_seed: int,
        verification_samples_per_layer: int = 2,
    ) -> dict[str, Any]:
        bits = np.asarray(bits_matrix)
        if bits.shape != (len(self._layers), self._layers[0].layout.num_experts):
            raise ValueError("Bit-assignment matrix does not match the model layout")
        allowed = {3, 4, 8, 16}
        if not set(np.unique(bits).tolist()) <= allowed:
            raise ValueError(f"Bit assignments must be within {sorted(allowed)}")
        if self._state != "clean":
            raise RuntimeError("Apply requires a verified clean model state")
        started = time.monotonic()
        hooks_before = module_hook_count(self.bundle.model)
        per_layer_quantized: list[int] = []
        distortions = np.zeros(bits.shape, dtype=np.float64)
        with torch.no_grad():
            for layer_index, layer in enumerate(self._layers):
                quantized = 0
                for expert_id in range(layer.layout.num_experts):
                    expert_bits = int(bits[layer_index, expert_id])
                    if expert_bits == 16:
                        continue
                    squared_error = 0.0
                    squared_norm = 0.0
                    for reference in layer.layout.references(expert_id):
                        view = reference.view()
                        result = STAGE1_QDQ_FUNCTION(
                            view, bits=expert_bits, group_size=self.group_size
                        )
                        view.copy_(result.dequantized)
                        squared_error += result.squared_error
                        squared_norm += result.squared_norm
                    distortions[layer_index, expert_id] = (
                        squared_error / squared_norm if squared_norm > 0 else 0.0
                    )
                    quantized += 1
                per_layer_quantized.append(quantized)
        sample_report = self._verify_assignment_sample(
            bits, verification_seed, verification_samples_per_layer
        )
        self.verify_non_expert_integrity()
        if module_hook_count(self.bundle.model) != hooks_before:
            raise RuntimeError("Applying an allocation changed registered hooks")
        self._state = "quantized"
        quantized_mask = bits != 16
        return {
            "applied": True,
            "elapsed_seconds": time.monotonic() - started,
            "quantized_experts": int(quantized_mask.sum()),
            "quantized_experts_per_layer": per_layer_quantized,
            "bits_histogram": {
                str(int(width)): int((bits == width).sum())
                for width in np.unique(bits)
            },
            "mean_relative_distortion_by_bits": {
                str(int(width)): float(distortions[bits == width].mean())
                for width in np.unique(bits)
                if width != 16
            },
            "assignment_verification": sample_report,
        }

    def _verify_assignment_sample(
        self, bits: np.ndarray, seed: int, samples_per_layer: int
    ) -> dict[str, Any]:
        """Prove sampled experts hold exactly the QDQ of their assigned bits."""

        rng = np.random.default_rng(seed)
        checked: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(self._layers):
            num_experts = layer.layout.num_experts
            count = min(samples_per_layer, num_experts)
            snapshot_by_parameter = {
                id(parameter): snapshot
                for parameter, snapshot in zip(
                    layer.parameters, layer.snapshots, strict=True
                )
            }
            for expert_id in rng.choice(num_experts, size=count, replace=False):
                expert_id = int(expert_id)
                assigned = int(bits[layer_index, expert_id])
                for reference in layer.layout.references(expert_id):
                    snapshot = snapshot_by_parameter[id(reference.parameter)]
                    if reference.expert_axis is None:
                        clean_slice = snapshot
                    else:
                        clean_slice = snapshot.select(reference.expert_axis, expert_id)
                    device_clean = clean_slice.to(reference.view().device)
                    if assigned == 16:
                        expected = device_clean
                    else:
                        expected = STAGE1_QDQ_FUNCTION(
                            device_clean, bits=assigned, group_size=self.group_size
                        ).dequantized
                    if not torch.equal(reference.view(), expected):
                        raise RuntimeError(
                            f"Layer {layer.spec.model_layer_index} expert {expert_id} "
                            f"does not hold the exact {assigned}-bit QDQ weights"
                        )
                    if assigned in (3, 4):
                        other = STAGE1_QDQ_FUNCTION(
                            device_clean, bits=8, group_size=self.group_size
                        ).dequantized
                        if torch.equal(reference.view(), other):
                            raise RuntimeError(
                                f"Layer {layer.spec.model_layer_index} expert "
                                f"{expert_id} is indistinguishable from 8-bit QDQ; "
                                "assignment verification cannot separate precisions"
                            )
                checked.append(
                    {
                        "layer": layer.spec.model_layer_index,
                        "expert": expert_id,
                        "bits": assigned,
                    }
                )
        return {"verified_exact_qdq": True, "sampled_experts": checked, "seed": seed}

    def restore_clean(self) -> dict[str, Any]:
        started = time.monotonic()
        with torch.no_grad():
            for layer in self._layers:
                for parameter, snapshot in zip(
                    layer.parameters, layer.snapshots, strict=True
                ):
                    parameter.copy_(snapshot.to(parameter.device))
        self.verify_clean()
        return {
            "restored": True,
            "restoration_verified_bitwise": True,
            "elapsed_seconds": time.monotonic() - started,
        }


def verify_layout_against_memory_shapes(
    manager: MixedPrecisionExpertManager,
    expected_shapes_by_layer: Sequence[Sequence[Sequence[int]]],
) -> None:
    """The live model layout must match the shapes used for memory accounting."""

    if len(manager._layers) != len(expected_shapes_by_layer):
        raise RuntimeError("Layer count mismatch between model and memory accounting")
    for layer, expected in zip(manager._layers, expected_shapes_by_layer, strict=True):
        observed = sorted(tuple(shape) for shape in layer.layout.tensor_shapes(0))
        wanted = sorted(tuple(int(v) for v in shape) for shape in expected)
        if observed != wanted:
            raise RuntimeError(
                f"Layer {layer.spec.model_layer_index} expert shapes {observed} do not "
                f"match the accounted shapes {wanted}"
            )


def run_repeated_baseline_check(
    bundle: ModelBundle,
    examples_by_domain: Mapping[str, PreparedDomainExamples],
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate the clean model twice per domain and require bitwise equality."""

    report: dict[str, Any] = {"passed": True, "domains": {}}
    for domain, examples in examples_by_domain.items():
        first, _ = evaluate_next_token_loss(bundle, examples, batch_size=batch_size)
        second, _ = evaluate_next_token_loss(bundle, examples, batch_size=batch_size)
        identical = bool(
            np.array_equal(first.loss_sums, second.loss_sums)
            and np.array_equal(first.token_counts, second.token_counts)
        )
        report["domains"][domain] = {
            "bitwise_identical": identical,
            "max_abs_loss_difference": float(
                np.max(np.abs(first.loss_sums - second.loss_sums))
            ),
        }
        if not identical:
            report["passed"] = False
    if not report["passed"]:
        raise RuntimeError(
            "Repeated clean-baseline evaluation is not bitwise reproducible; "
            "strict determinism is not achieved. Stopping as preregistered: "
            + str(report)
        )
    return report


def evaluation_run_fingerprint(
    bundle: ModelBundle,
    registry: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    phase: str,
    batch_size: int,
    determinism: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "stage": "stage2b_protection_evaluation",
            "phase": phase,
            "model": bundle.checkpoint,
            "resolved_model_revision": bundle.resolved_revision,
            "dtype": str(bundle.runtime.dtype).replace("torch.", ""),
            "batch_size": batch_size,
            "registry_sha256": registry["registry_sha256"],
            "split_input_hashes": dict(split_hashes),
            "deterministic_settings": {
                key: value
                for key, value in determinism.items()
                if key != "torch_version"
            },
        }
    )


def allocation_slug(record: Mapping[str, Any]) -> str:
    if record["method_kind"] == "uniform_reference":
        return str(record["method"])
    return (
        f"{record['method']}_{record['regime']}_budget"
        f"{int(round(record['budget_fraction'] * 100))}"
    )


def allocation_is_complete(
    losses_dir: Path,
    record: Mapping[str, Any],
    run_fingerprint: str,
    domains: Sequence[str] = STAGE2B_DOMAINS,
) -> bool:
    slug = allocation_slug(record)
    for domain in domains:
        path = losses_dir / slug / f"{domain}.npz"
        metadata_path = path.with_suffix(".metadata.json")
        if not (path.is_file() and metadata_path.is_file()):
            return False
        metadata = read_json(metadata_path)
        if metadata.get("run_fingerprint") != run_fingerprint:
            return False
        if metadata.get("allocation_sha256") != record["allocation_sha256"]:
            return False
    return True


def evaluate_allocation_records(
    bundle: ModelBundle,
    manager: MixedPrecisionExpertManager,
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, PreparedDomainExamples],
    split_hashes: Mapping[str, str],
    losses_dir: Path,
    run_fingerprint: str,
    batch_size: int,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate every allocation with checkpointed resume and exact restoration."""

    progress_path = losses_dir / "progress.json"
    diagnostics: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        slug = allocation_slug(record)
        expected_base = {
            "run_fingerprint": run_fingerprint,
            "allocation_sha256": record["allocation_sha256"],
            "method": record["method"],
            "regime": record["regime"],
            "budget_fraction": record["budget_fraction"],
            "resolved_model_revision": bundle.resolved_revision,
            "group_size": record["group_size"],
        }
        if resume and allocation_is_complete(losses_dir, record, run_fingerprint):
            print(f"[{index + 1}/{len(records)}] resume complete: {slug}", flush=True)
            diagnostics.append({"allocation": slug, "resumed": True})
            continue
        print(f"[{index + 1}/{len(records)}] evaluating: {slug}", flush=True)
        manager.verify_clean()
        bits = np.asarray(record["expert_bits"], dtype=np.int64)
        is_bf16 = bool(np.all(bits == 16))
        application: dict[str, Any] = {"applied": False}
        if not is_bf16:
            verification_seed = int(record["allocation_sha256"][:8], 16)
            try:
                application = manager.apply_allocation(bits, verification_seed)
            except BaseException:
                manager.restore_clean()
                raise
        try:
            for domain in STAGE2B_DOMAINS:
                examples = splits[domain]
                checkpoint_path = losses_dir / slug / f"{domain}.npz"
                expected_metadata = {
                    **expected_base,
                    "domain": domain,
                    "split_input_ids_sha256": split_hashes[domain],
                }
                checkpoint = load_or_compute_loss_checkpoint(
                    checkpoint_path,
                    expected_metadata,
                    examples.measurement_mask.sum(axis=1).astype(np.uint32),
                    lambda examples=examples: evaluate_next_token_loss(
                        bundle, examples, batch_size=batch_size
                    ),
                    resume=resume,
                )
                del checkpoint
        finally:
            restoration = {"restored": False}
            if not is_bf16:
                restoration = manager.restore_clean()
            else:
                manager.verify_clean()
        diagnostics.append(
            {
                "allocation": slug,
                "resumed": False,
                "application": application,
                "restoration": restoration,
            }
        )
        atomic_write_json(
            progress_path,
            {
                "run_fingerprint": run_fingerprint,
                "completed_through_index": index,
                "total_records": len(records),
                "last_allocation": slug,
            },
        )
    return diagnostics


def load_allocation_losses(
    losses_dir: Path,
    record: Mapping[str, Any],
    run_fingerprint: str,
) -> dict[str, LossStatistics]:
    """Load one allocation's per-domain loss checkpoints with validation."""

    slug = allocation_slug(record)
    output: dict[str, LossStatistics] = {}
    for domain in STAGE2B_DOMAINS:
        path = losses_dir / slug / f"{domain}.npz"
        metadata = read_json(path.with_suffix(".metadata.json"))
        if metadata.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"Loss checkpoint {path} has a different run fingerprint")
        if metadata.get("allocation_sha256") != record["allocation_sha256"]:
            raise RuntimeError(f"Loss checkpoint {path} has a different allocation hash")
        output[domain] = LossStatistics.load(path)
    return output
