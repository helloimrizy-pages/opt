from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .balanced import array_sha256, file_sha256
from .controlled import PreparedDomainExamples
from .hooks import (
    extract_routing,
    find_hidden_tensor,
    run_expert,
    tensor_output,
)
from .io_utils import atomic_save_npz, atomic_write_json, read_json
from .modeling import ModelBundle, MoeLayerSpec


REPLAY_CAPTURE_SCHEMA_VERSION = 1
DEFAULT_REPLAY_ATOL = 3e-2
DEFAULT_REPLAY_RTOL = 3e-2


def _derived_seed(seed: int, label: str) -> int:
    payload = f"olmoe-stage2a-replay-v1\0{seed}\0{label}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)


def _encode_exact_tensor(tensor: torch.Tensor) -> tuple[np.ndarray, str, str]:
    """Return an exact CPU representation and its storage/logical dtype labels."""

    value = tensor.detach().contiguous().cpu()
    logical = str(value.dtype).replace("torch.", "")
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().copy(), "uint16_bfloat16_bits", logical
    if value.dtype == torch.float16:
        return value.numpy().copy(), "float16", logical
    if value.dtype == torch.float32:
        return value.numpy().copy(), "float32", logical
    raise TypeError(f"Unsupported replay tensor dtype {value.dtype}")


def _decode_exact_tensor(array: np.ndarray, storage_dtype: str) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    if storage_dtype == "uint16_bfloat16_bits":
        if contiguous.dtype != np.uint16:
            raise ValueError("BF16 replay storage must use uint16 payloads")
        return torch.from_numpy(contiguous.copy()).view(torch.bfloat16)
    if storage_dtype == "float16":
        if contiguous.dtype != np.float16:
            raise ValueError("FP16 replay storage has the wrong NumPy dtype")
        return torch.from_numpy(contiguous.copy())
    if storage_dtype == "float32":
        if contiguous.dtype != np.float32:
            raise ValueError("FP32 replay storage has the wrong NumPy dtype")
        return torch.from_numpy(contiguous.copy())
    raise ValueError(f"Unsupported replay storage dtype {storage_dtype!r}")


@dataclass
class ReplayCapture:
    domain: str
    model_layer_index: int
    hidden_states: torch.Tensor
    selected_expert_ids: torch.Tensor
    selected_gate_weights: torch.Tensor
    example_indices: np.ndarray
    token_positions: np.ndarray
    layer_energy_by_example: np.ndarray
    sample_row_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    sample_topk_positions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int16)
    )
    sample_live_contributions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    sample_live_moe_outputs: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return int(self.hidden_states.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.hidden_states.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.selected_expert_ids.shape[1])

    @property
    def layer_energy(self) -> float:
        return float(self.layer_energy_by_example.sum(dtype=np.float64))

    def route_rows(self, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.where(self.selected_expert_ids == int(expert_id))

    def validate(
        self,
        *,
        expected_tokens: int | None = None,
        expected_top_k: int | None = None,
        expected_examples: int | None = None,
        num_experts: int | None = None,
    ) -> None:
        if self.hidden_states.ndim != 2 or not self.hidden_states.is_floating_point():
            raise ValueError("Replay hidden states must have shape [token, hidden]")
        if self.selected_expert_ids.ndim != 2:
            raise ValueError("Replay expert IDs must have shape [token, top_k]")
        if tuple(self.selected_gate_weights.shape) != tuple(self.selected_expert_ids.shape):
            raise ValueError("Replay gate weights and expert IDs have different shapes")
        if self.selected_expert_ids.shape[0] != self.num_tokens:
            raise ValueError("Replay routes and hidden states have different token counts")
        if self.example_indices.shape != (self.num_tokens,):
            raise ValueError("Replay example indices have the wrong shape")
        if self.token_positions.shape != (self.num_tokens,):
            raise ValueError("Replay token positions have the wrong shape")
        if expected_tokens is not None and self.num_tokens != expected_tokens:
            raise ValueError(
                f"Replay contains {self.num_tokens} tokens, expected {expected_tokens}"
            )
        if expected_top_k is not None and self.top_k != expected_top_k:
            raise ValueError(f"Replay contains top-{self.top_k}, expected top-{expected_top_k}")
        if expected_examples is not None:
            if self.layer_energy_by_example.shape != (expected_examples,):
                raise ValueError("Layer-energy summaries have the wrong example count")
            if self.num_tokens and (
                int(self.example_indices.min()) < 0
                or int(self.example_indices.max()) >= expected_examples
            ):
                raise ValueError("Replay example index is out of range")
        if num_experts is not None and self.selected_expert_ids.numel():
            if int(self.selected_expert_ids.min()) < 0 or int(
                self.selected_expert_ids.max()
            ) >= int(num_experts):
                raise ValueError("Replay expert ID is out of range")
        if not bool(torch.isfinite(self.hidden_states.float()).all()):
            raise ValueError("Replay hidden states contain non-finite values")
        if not bool(torch.isfinite(self.selected_gate_weights).all()) or bool(
            (self.selected_gate_weights < 0).any()
        ):
            raise ValueError("Replay gate weights contain invalid values")
        if not np.all(np.isfinite(self.layer_energy_by_example)) or np.any(
            self.layer_energy_by_example < 0
        ):
            raise ValueError("Layer-energy summaries contain invalid values")
        if self.layer_energy <= 0:
            raise ValueError("A replay capture has non-positive MoE-output energy")
        sample_count = len(self.sample_row_indices)
        if self.sample_topk_positions.shape != (sample_count,):
            raise ValueError("Replay sample top-k positions are misaligned")
        if self.sample_live_contributions.shape != (sample_count, self.hidden_size):
            raise ValueError("Replay sample contributions have the wrong shape")
        if self.sample_live_moe_outputs.shape != (sample_count, self.hidden_size):
            raise ValueError("Replay sample MoE outputs have the wrong shape")
        if sample_count:
            if np.any(self.sample_row_indices < 0) or np.any(
                self.sample_row_indices >= self.num_tokens
            ):
                raise ValueError("Replay sample row is out of range")
            if np.any(self.sample_topk_positions < 0) or np.any(
                self.sample_topk_positions >= self.top_k
            ):
                raise ValueError("Replay sample top-k position is out of range")
            if not np.all(np.isfinite(self.sample_live_contributions)) or not np.all(
                np.isfinite(self.sample_live_moe_outputs)
            ):
                raise ValueError("Replay validation samples contain non-finite values")

    def save(self, path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
        self.validate()
        hidden_payload, storage_dtype, logical_dtype = _encode_exact_tensor(
            self.hidden_states
        )
        arrays = {
            "hidden_states": hidden_payload,
            "selected_expert_ids": self.selected_expert_ids.cpu().numpy().astype(
                np.int16, copy=False
            ),
            "selected_gate_weights": self.selected_gate_weights.cpu().numpy().astype(
                np.float32, copy=False
            ),
            "example_indices": self.example_indices.astype(np.int32, copy=False),
            "token_positions": self.token_positions.astype(np.int16, copy=False),
            "layer_energy_by_example": self.layer_energy_by_example.astype(
                np.float64, copy=False
            ),
            "sample_row_indices": self.sample_row_indices.astype(np.int64, copy=False),
            "sample_topk_positions": self.sample_topk_positions.astype(
                np.int16, copy=False
            ),
            "sample_live_contributions": self.sample_live_contributions.astype(
                np.float32, copy=False
            ),
            "sample_live_moe_outputs": self.sample_live_moe_outputs.astype(
                np.float32, copy=False
            ),
        }
        atomic_save_npz(path, **arrays)
        payload = {
            **dict(metadata),
            "schema_version": REPLAY_CAPTURE_SCHEMA_VERSION,
            "domain": self.domain,
            "model_layer_index": self.model_layer_index,
            "hidden_logical_dtype": logical_dtype,
            "hidden_storage_dtype": storage_dtype,
            "hidden_shape": list(self.hidden_states.shape),
            "route_shape": list(self.selected_expert_ids.shape),
            "num_measured_tokens": self.num_tokens,
            "hidden_size": self.hidden_size,
            "top_k": self.top_k,
            "layer_energy": self.layer_energy,
            "sample_count": len(self.sample_row_indices),
            "array_sha256": {
                key: array_sha256(np.asarray(value)) for key, value in arrays.items()
            },
            "npz_sha256": file_sha256(path),
        }
        atomic_write_json(path.with_suffix(".metadata.json"), payload)
        self.metadata = payload
        return payload

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> "ReplayCapture":
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Replay checkpoint is incomplete: {path}")
        metadata = read_json(metadata_path)
        if metadata.get("schema_version") != REPLAY_CAPTURE_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported replay schema in {path}")
        if expected_metadata:
            mismatches = [
                key
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            ]
            if mismatches:
                raise RuntimeError(
                    f"Replay checkpoint fingerprint mismatch for {path}: "
                    + ", ".join(mismatches)
                )
        if metadata.get("npz_sha256") != file_sha256(path):
            raise RuntimeError(f"Replay checkpoint hash mismatch for {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {
                "hidden_states",
                "selected_expert_ids",
                "selected_gate_weights",
                "example_indices",
                "token_positions",
                "layer_energy_by_example",
                "sample_row_indices",
                "sample_topk_positions",
                "sample_live_contributions",
                "sample_live_moe_outputs",
            }
            if set(data.files) != required:
                raise RuntimeError(f"Replay checkpoint arrays are incomplete in {path}")
            arrays = {key: data[key] for key in data.files}
        recorded_hashes = metadata.get("array_sha256", {})
        for key, value in arrays.items():
            if recorded_hashes.get(key) != array_sha256(value):
                raise RuntimeError(f"Replay array hash mismatch for {path}:{key}")
        result = cls(
            domain=str(metadata["domain"]),
            model_layer_index=int(metadata["model_layer_index"]),
            hidden_states=_decode_exact_tensor(
                arrays["hidden_states"], str(metadata["hidden_storage_dtype"])
            ),
            selected_expert_ids=torch.from_numpy(
                arrays["selected_expert_ids"].astype(np.int64)
            ),
            selected_gate_weights=torch.from_numpy(
                arrays["selected_gate_weights"].astype(np.float32)
            ),
            example_indices=arrays["example_indices"].astype(np.int64),
            token_positions=arrays["token_positions"].astype(np.int64),
            layer_energy_by_example=arrays["layer_energy_by_example"].astype(
                np.float64
            ),
            sample_row_indices=arrays["sample_row_indices"].astype(np.int64),
            sample_topk_positions=arrays["sample_topk_positions"].astype(np.int16),
            sample_live_contributions=arrays["sample_live_contributions"].astype(
                np.float32
            ),
            sample_live_moe_outputs=arrays["sample_live_moe_outputs"].astype(
                np.float32
            ),
            metadata=metadata,
        )
        result.validate()
        if list(result.hidden_states.shape) != metadata.get("hidden_shape"):
            raise RuntimeError(f"Replay hidden shape metadata mismatch in {path}")
        return result


@dataclass
class _CaptureBatch:
    example_indices: np.ndarray
    measurement_mask: torch.Tensor
    sequence_length: int
    routing: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    completed: set[int] = field(default_factory=set)


class ReplayCaptureSession:
    """Capture only routed MoE replay state from baseline forwards."""

    def __init__(
        self,
        layer_specs: Sequence[MoeLayerSpec],
        num_examples: int,
        sample_locations: Mapping[int, Sequence[tuple[int, int, int]]] | None = None,
    ) -> None:
        if not layer_specs:
            raise ValueError("At least one MoE layer is required for replay capture")
        self.layer_specs = list(layer_specs)
        self.num_examples = int(num_examples)
        if self.num_examples < 1:
            raise ValueError("Replay capture requires at least one example")
        self.sample_locations = {
            int(layer): set(tuple(int(value) for value in item) for item in locations)
            for layer, locations in (sample_locations or {}).items()
        }
        self._handles: list[Any] = []
        self._batch: _CaptureBatch | None = None
        self._entered = False
        self._hidden: dict[int, list[torch.Tensor]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._expert_ids: dict[int, list[torch.Tensor]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._gate_weights: dict[int, list[torch.Tensor]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._example_indices: dict[int, list[np.ndarray]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._token_positions: dict[int, list[np.ndarray]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._energy: dict[int, np.ndarray] = {
            spec.ordinal: np.zeros(self.num_examples, dtype=np.float64)
            for spec in self.layer_specs
        }
        self._captured_token_count: dict[int, int] = {
            spec.ordinal: 0 for spec in self.layer_specs
        }
        self._sample_rows: dict[int, list[int]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._sample_topk: dict[int, list[int]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._sample_contributions: dict[int, list[np.ndarray]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._sample_outputs: dict[int, list[np.ndarray]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }

    def __enter__(self) -> "ReplayCaptureSession":
        if self._entered:
            raise RuntimeError("Replay capture cannot be entered twice")
        self._entered = True
        for spec in self.layer_specs:
            self._handles.append(
                spec.router.register_forward_hook(
                    self._make_router_hook(spec), with_kwargs=True
                )
            )
            self._handles.append(
                spec.experts.register_forward_pre_hook(
                    self._make_experts_pre_hook(spec), with_kwargs=True
                )
            )
            self._handles.append(
                spec.experts.register_forward_hook(
                    self._make_experts_post_hook(spec), with_kwargs=True
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def registered_hook_count(self) -> int:
        return len(self._handles)

    @contextmanager
    def batch(
        self,
        example_indices: Sequence[int],
        measurement_mask: torch.Tensor,
    ) -> Iterator[None]:
        if not self._entered or not self._handles:
            raise RuntimeError("Enter ReplayCaptureSession before starting a batch")
        if self._batch is not None:
            raise RuntimeError("Nested replay-capture batches are unsupported")
        if measurement_mask.ndim != 2:
            raise ValueError("measurement_mask must have shape [batch, sequence]")
        indices = np.asarray(example_indices, dtype=np.int64)
        if len(indices) != measurement_mask.shape[0]:
            raise ValueError("One example index is required for each batch row")
        if np.any(indices < 0) or np.any(indices >= self.num_examples):
            raise IndexError("Replay example index is out of range")
        self._batch = _CaptureBatch(
            example_indices=indices,
            measurement_mask=measurement_mask.detach(),
            sequence_length=int(measurement_mask.shape[1]),
        )
        try:
            yield
            expected = {spec.ordinal for spec in self.layer_specs}
            if self._batch.completed != expected:
                missing = sorted(expected - self._batch.completed)
                raise RuntimeError(f"Replay capture missed MoE layer ordinals {missing}")
        finally:
            self._batch = None

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._batch = None
        self._entered = False

    def finalize(self, domain: str) -> dict[int, ReplayCapture]:
        if self._batch is not None:
            raise RuntimeError("Cannot finalize replay capture inside a batch")
        output: dict[int, ReplayCapture] = {}
        for spec in self.layer_specs:
            ordinal = spec.ordinal
            if not self._hidden[ordinal]:
                raise RuntimeError(f"No replay tokens captured for {spec.block_name}")
            hidden = torch.cat(self._hidden[ordinal], dim=0)
            expert_ids = torch.cat(self._expert_ids[ordinal], dim=0)
            gates = torch.cat(self._gate_weights[ordinal], dim=0)
            examples = np.concatenate(self._example_indices[ordinal])
            positions = np.concatenate(self._token_positions[ordinal])
            sample_count = len(self._sample_rows[ordinal])
            result = ReplayCapture(
                domain=domain,
                model_layer_index=spec.model_layer_index,
                hidden_states=hidden,
                selected_expert_ids=expert_ids,
                selected_gate_weights=gates,
                example_indices=examples,
                token_positions=positions,
                layer_energy_by_example=self._energy[ordinal],
                sample_row_indices=np.asarray(
                    self._sample_rows[ordinal], dtype=np.int64
                ),
                sample_topk_positions=np.asarray(
                    self._sample_topk[ordinal], dtype=np.int16
                ),
                sample_live_contributions=(
                    np.stack(self._sample_contributions[ordinal]).astype(np.float32)
                    if sample_count
                    else np.empty((0, hidden.shape[1]), dtype=np.float32)
                ),
                sample_live_moe_outputs=(
                    np.stack(self._sample_outputs[ordinal]).astype(np.float32)
                    if sample_count
                    else np.empty((0, hidden.shape[1]), dtype=np.float32)
                ),
            )
            result.validate(
                expected_top_k=spec.top_k,
                expected_examples=self.num_examples,
                num_experts=spec.num_experts,
            )
            output[spec.model_layer_index] = result
        return output

    def _make_router_hook(self, spec: MoeLayerSpec):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del args, kwargs
            batch = self._require_batch()
            if spec.ordinal in batch.routing:
                raise RuntimeError(f"Router for {spec.block_name} ran twice")
            indices, weights, _, _ = extract_routing(output, module, spec)
            valid_mask, local_examples, positions = self._token_mapping(
                indices.shape[0], indices.device
            )
            batch.routing[spec.ordinal] = (
                indices[valid_mask].detach(),
                weights[valid_mask].detach(),
                torch.stack((local_examples[valid_mask], positions[valid_mask]), dim=1),
            )

        return hook

    def _make_experts_pre_hook(self, spec: MoeLayerSpec):
        def hook(
            module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            del module
            batch = self._require_batch()
            captured = find_hidden_tensor(args, kwargs, spec.top_k)
            hidden = captured.reshape(-1, captured.shape[-1])
            valid_mask, _, _ = self._token_mapping(hidden.shape[0], hidden.device)
            batch.hidden[spec.ordinal] = hidden[valid_mask]

        return hook

    def _make_experts_post_hook(self, spec: MoeLayerSpec):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            batch = self._require_batch()
            if spec.ordinal in batch.completed:
                raise RuntimeError(f"Experts for {spec.block_name} ran twice")
            if spec.ordinal not in batch.routing or spec.ordinal not in batch.hidden:
                raise RuntimeError(f"Incomplete replay hook state for {spec.block_name}")
            indices, weights, mapping = batch.routing[spec.ordinal]
            hidden = batch.hidden.pop(spec.ordinal)
            moe_output = tensor_output(output).reshape(-1, hidden.shape[-1])
            valid_mask, _, _ = self._token_mapping(moe_output.shape[0], moe_output.device)
            moe_output = moe_output[valid_mask]
            if hidden.shape[0] != indices.shape[0] or moe_output.shape != hidden.shape:
                raise RuntimeError(
                    f"Replay hidden/route/output mismatch in {spec.block_name}: "
                    f"hidden={tuple(hidden.shape)}, routes={tuple(indices.shape)}, "
                    f"output={tuple(moe_output.shape)}"
                )
            local_examples = mapping[:, 0].long()
            positions = mapping[:, 1].long()
            global_examples = torch.as_tensor(
                batch.example_indices, dtype=torch.long, device=local_examples.device
            )[local_examples]

            ordinal = spec.ordinal
            offset = self._captured_token_count[ordinal]
            self._hidden[ordinal].append(hidden.detach().cpu())
            self._expert_ids[ordinal].append(indices.detach().long().cpu())
            self._gate_weights[ordinal].append(weights.detach().float().cpu())
            self._example_indices[ordinal].append(
                global_examples.detach().cpu().numpy().astype(np.int64)
            )
            self._token_positions[ordinal].append(
                positions.detach().cpu().numpy().astype(np.int64)
            )
            token_energy = moe_output.detach().float().square().sum(dim=-1).double()
            np.add.at(
                self._energy[ordinal],
                global_examples.detach().cpu().numpy().astype(np.int64),
                token_energy.cpu().numpy(),
            )

            wanted = self.sample_locations.get(spec.model_layer_index, set())
            if wanted:
                for row in range(hidden.shape[0]):
                    example_id = int(global_examples[row].item())
                    token_position = int(positions[row].item())
                    candidates = [
                        item
                        for item in wanted
                        if item[0] == example_id and item[1] == token_position
                    ]
                    for _, _, topk_position in candidates:
                        expert_id = int(indices[row, topk_position].item())
                        # Use the model container's own routed forward path for the
                        # capture-time reference. Offline validation separately uses
                        # ``run_expert``; agreement is therefore not a self-comparison
                        # of the same replay implementation.
                        isolated = tensor_output(
                            spec.experts.forward(
                                hidden[row : row + 1],
                                indices[row : row + 1, topk_position : topk_position + 1],
                                weights[row : row + 1, topk_position : topk_position + 1],
                            )
                        )
                        if isolated.shape != hidden[row : row + 1].shape:
                            raise RuntimeError(
                                f"Isolated live contribution for {spec.block_name} has "
                                f"shape {tuple(isolated.shape)}, expected "
                                f"{tuple(hidden[row : row + 1].shape)}"
                            )
                        contribution = isolated.float()[0]
                        self._sample_rows[ordinal].append(offset + row)
                        self._sample_topk[ordinal].append(topk_position)
                        self._sample_contributions[ordinal].append(
                            contribution.detach().cpu().numpy()
                        )
                        self._sample_outputs[ordinal].append(
                            moe_output[row].detach().float().cpu().numpy()
                        )
            self._captured_token_count[ordinal] += hidden.shape[0]
            batch.completed.add(ordinal)

        return hook

    def _token_mapping(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._require_batch()
        mask = batch.measurement_mask.to(device=device, dtype=torch.bool).reshape(-1)
        local = torch.arange(
            len(batch.example_indices), device=device
        ).repeat_interleave(batch.sequence_length)
        positions = torch.arange(batch.sequence_length, device=device).repeat(
            len(batch.example_indices)
        )
        if rows == mask.numel():
            return mask, local, positions
        if rows == int(mask.sum().item()):
            selected = torch.ones(rows, dtype=torch.bool, device=device)
            return selected, local[mask], positions[mask]
        raise RuntimeError(
            f"MoE produced {rows} rows for {mask.numel()} sequence positions and "
            f"{int(mask.sum().item())} measured positions"
        )

    def _require_batch(self) -> _CaptureBatch:
        if self._batch is None:
            raise RuntimeError("A replay hook fired outside an active batch")
        return self._batch


def deterministic_sample_locations(
    examples: PreparedDomainExamples,
    layer_indices: Sequence[int],
    *,
    samples_per_layer: int,
    top_k: int,
    seed: int,
) -> dict[int, list[tuple[int, int, int]]]:
    if samples_per_layer < 0:
        raise ValueError("samples_per_layer cannot be negative")
    candidates = np.argwhere(examples.measurement_mask.astype(bool))
    output: dict[int, list[tuple[int, int, int]]] = {}
    for layer in layer_indices:
        if samples_per_layer == 0:
            output[int(layer)] = []
            continue
        rng = np.random.default_rng(
            _derived_seed(seed, f"{examples.domain}-layer-{int(layer)}")
        )
        count = min(samples_per_layer, len(candidates))
        chosen = rng.choice(len(candidates), size=count, replace=False)
        topk_positions = rng.integers(0, top_k, size=count)
        output[int(layer)] = [
            (
                int(candidates[index, 0]),
                int(candidates[index, 1]),
                int(topk_position),
            )
            for index, topk_position in zip(chosen, topk_positions, strict=True)
        ]
    return output


def capture_replay_dataset(
    bundle: ModelBundle,
    layer_specs: Sequence[MoeLayerSpec],
    prepared: Mapping[str, PreparedDomainExamples],
    output_dir: Path,
    *,
    capture_fingerprint: str,
    controlled_input_file_sha256: Mapping[str, str],
    source_statistics: Mapping[str, Any] | None = None,
    batch_size: int = 1,
    seed: int = 42,
    validation_layer_indices: Sequence[int] = (),
    validation_samples_per_layer: int = 3,
    resume: bool = True,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    by_model_layer = {spec.model_layer_index: spec for spec in layer_specs}
    if set(by_model_layer) != {spec.model_layer_index for spec in layer_specs}:
        raise RuntimeError("Runtime MoE model-layer indices are not unique")
    manifest_rows: list[dict[str, Any]] = []
    for domain, examples in prepared.items():
        examples.validate()
        missing_specs: list[MoeLayerSpec] = []
        expected_common = {
            "capture_fingerprint": capture_fingerprint,
            "controlled_input_file_sha256": controlled_input_file_sha256[domain],
            "domain": domain,
        }
        for spec in layer_specs:
            path = output_dir / domain / f"layer_{spec.model_layer_index:02d}.npz"
            try:
                if not resume:
                    raise FileNotFoundError
                capture = ReplayCapture.load(path, expected_metadata=expected_common)
                capture.validate(
                    expected_tokens=int(examples.measurement_mask.sum()),
                    expected_top_k=spec.top_k,
                    expected_examples=examples.num_examples,
                    num_experts=spec.num_experts,
                )
                if source_statistics is not None:
                    observed = np.bincount(
                        capture.selected_expert_ids.numpy().reshape(-1),
                        minlength=spec.num_experts,
                    )
                    expected = source_statistics[domain].routing_counts[
                        :, spec.ordinal, :
                    ].sum(axis=0, dtype=np.uint64)
                    if not np.array_equal(observed.astype(np.uint64), expected):
                        raise RuntimeError(
                            f"Replay route counts differ from frozen statistics for "
                            f"{domain}/L{spec.model_layer_index}"
                        )
                manifest_rows.append(
                    {
                        "domain": domain,
                        "layer": spec.model_layer_index,
                        "path": str(path),
                        "npz_sha256": capture.metadata["npz_sha256"],
                        "resumed": True,
                    }
                )
                continue
            except FileNotFoundError:
                missing_specs.append(spec)
        if not missing_specs:
            continue

        sample_locations = deterministic_sample_locations(
            examples,
            [
                spec.model_layer_index
                for spec in missing_specs
                if spec.model_layer_index in set(validation_layer_indices)
            ],
            samples_per_layer=validation_samples_per_layer,
            top_k=missing_specs[0].top_k,
            seed=seed,
        )
        session = ReplayCaptureSession(
            missing_specs,
            examples.num_examples,
            sample_locations=sample_locations,
        )
        with session:
            for start in range(0, examples.num_examples, batch_size):
                stop = min(start + batch_size, examples.num_examples)
                input_ids = torch.as_tensor(
                    examples.input_ids[start:stop],
                    dtype=torch.long,
                    device=bundle.runtime.device,
                )
                attention_mask = torch.as_tensor(
                    examples.attention_mask[start:stop],
                    dtype=torch.long,
                    device=bundle.runtime.device,
                )
                measurement_mask = torch.as_tensor(
                    examples.measurement_mask[start:stop],
                    dtype=torch.bool,
                    device=bundle.runtime.device,
                )
                with session.batch(range(start, stop), measurement_mask):
                    with torch.inference_mode():
                        result = bundle.backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        del result
                if stop == examples.num_examples or stop % 10 == 0:
                    print(
                        f"[capture:{domain}] {stop}/{examples.num_examples} examples",
                        flush=True,
                    )
        captures = session.finalize(domain)
        if session.registered_hook_count != 0:
            raise RuntimeError("Replay capture leaked model hooks")
        for spec in missing_specs:
            capture = captures[spec.model_layer_index]
            expected_tokens = int(examples.measurement_mask.sum())
            capture.validate(
                expected_tokens=expected_tokens,
                expected_top_k=spec.top_k,
                expected_examples=examples.num_examples,
                num_experts=spec.num_experts,
            )
            observed_counts = np.bincount(
                capture.selected_expert_ids.numpy().reshape(-1),
                minlength=spec.num_experts,
            ).astype(np.uint64)
            if source_statistics is not None:
                expected_counts = source_statistics[domain].routing_counts[
                    :, spec.ordinal, :
                ].sum(axis=0, dtype=np.uint64)
                if not np.array_equal(observed_counts, expected_counts):
                    raise RuntimeError(
                        f"Fresh replay route counts differ from frozen statistics for "
                        f"{domain}/L{spec.model_layer_index}"
                    )
            path = output_dir / domain / f"layer_{spec.model_layer_index:02d}.npz"
            metadata = capture.save(
                path,
                {
                    **expected_common,
                    "layer_ordinal": spec.ordinal,
                    "num_examples": examples.num_examples,
                    "expected_measured_tokens": expected_tokens,
                    "num_experts": spec.num_experts,
                    "route_count_total": int(observed_counts.sum()),
                    "routing_count_validation": (
                        "exact_match_to_frozen_controlled_statistics"
                        if source_statistics is not None
                        else "not_requested"
                    ),
                    "example_ids_sha256": hashlib.sha256(
                        "\n".join(
                            str(value)
                            for value in examples.metadata.get(
                                "selected_example_ids", []
                            )
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            )
            manifest_rows.append(
                {
                    "domain": domain,
                    "layer": spec.model_layer_index,
                    "path": str(path),
                    "npz_sha256": metadata["npz_sha256"],
                    "resumed": False,
                }
            )
    manifest_rows.sort(key=lambda row: (row["domain"], row["layer"]))
    return {
        "schema_version": REPLAY_CAPTURE_SCHEMA_VERSION,
        "capture_fingerprint": capture_fingerprint,
        "domains": list(prepared),
        "layers": sorted(by_model_layer),
        "capture_files": manifest_rows,
        "all_capture_files_valid": len(manifest_rows)
        == len(prepared) * len(layer_specs),
        "hidden_dtype_preserved_exactly": True,
        "selected_gate_storage_dtype": "float32",
        "unnecessary_attention_states_stored": False,
        "full_transformer_activations_stored": False,
    }


def validate_replay_captures(
    layer_specs: Sequence[MoeLayerSpec],
    capture_dir: Path,
    domains: Sequence[str],
    *,
    capture_fingerprint: str,
    validation_layer_indices: Sequence[int],
    atol: float = DEFAULT_REPLAY_ATOL,
    rtol: float = DEFAULT_REPLAY_RTOL,
) -> dict[str, Any]:
    if atol < 0 or rtol < 0:
        raise ValueError("Replay tolerances must be nonnegative")
    specs = {spec.model_layer_index: spec for spec in layer_specs}
    rows: list[dict[str, Any]] = []
    unique_experts: set[tuple[int, int]] = set()
    maximum_contribution_error = 0.0
    maximum_aggregate_error = 0.0
    for domain in domains:
        for layer in validation_layer_indices:
            spec = specs[int(layer)]
            capture = ReplayCapture.load(
                capture_dir / domain / f"layer_{int(layer):02d}.npz",
                expected_metadata={"capture_fingerprint": capture_fingerprint},
            )
            for sample_index, (row_index, topk_position) in enumerate(
                zip(
                    capture.sample_row_indices,
                    capture.sample_topk_positions,
                    strict=True,
                )
            ):
                hidden = capture.hidden_states[int(row_index) : int(row_index) + 1].to(
                    next(spec.experts.parameters()).device
                )
                expert_id = int(
                    capture.selected_expert_ids[int(row_index), int(topk_position)].item()
                )
                gate = float(
                    capture.selected_gate_weights[
                        int(row_index), int(topk_position)
                    ].item()
                )
                with torch.no_grad():
                    contribution = (
                        run_expert(spec, expert_id, hidden).float()[0] * gate
                    ).cpu()
                    reconstructed = torch.zeros(
                        capture.hidden_size, dtype=torch.float32, device=hidden.device
                    )
                    for position in range(capture.top_k):
                        routed_expert = int(
                            capture.selected_expert_ids[
                                int(row_index), position
                            ].item()
                        )
                        routed_gate = capture.selected_gate_weights[
                            int(row_index), position
                        ].to(hidden.device)
                        reconstructed += (
                            run_expert(spec, routed_expert, hidden).float()[0]
                            * routed_gate.float()
                        )
                    reconstructed = reconstructed.cpu()
                live_contribution = torch.from_numpy(
                    capture.sample_live_contributions[sample_index]
                )
                live_output = torch.from_numpy(
                    capture.sample_live_moe_outputs[sample_index]
                )
                contribution_error = float(
                    (contribution - live_contribution).abs().max().item()
                )
                aggregate_error = float(
                    (reconstructed - live_output).abs().max().item()
                )
                maximum_contribution_error = max(
                    maximum_contribution_error, contribution_error
                )
                maximum_aggregate_error = max(maximum_aggregate_error, aggregate_error)
                contribution_passed = bool(
                    torch.allclose(
                        contribution, live_contribution, atol=atol, rtol=rtol
                    )
                )
                aggregate_passed = bool(
                    torch.allclose(reconstructed, live_output, atol=atol, rtol=rtol)
                )
                unique_experts.add((int(layer), expert_id))
                rows.append(
                    {
                        "domain": domain,
                        "layer": int(layer),
                        "example_index": int(capture.example_indices[int(row_index)]),
                        "token_position": int(capture.token_positions[int(row_index)]),
                        "topk_position": int(topk_position),
                        "expert_id": expert_id,
                        "gate_weight": gate,
                        "contribution_max_abs_error": contribution_error,
                        "aggregate_moe_output_max_abs_error": aggregate_error,
                        "contribution_passed": contribution_passed,
                        "aggregate_passed": aggregate_passed,
                    }
                )
    if not rows:
        raise RuntimeError("Replay validation selected no samples")
    passed = all(
        row["contribution_passed"] and row["aggregate_passed"] for row in rows
    )
    return {
        "passed": passed,
        "stopping_gate": True,
        "schema_version": REPLAY_CAPTURE_SCHEMA_VERSION,
        "capture_fingerprint": capture_fingerprint,
        "atol": atol,
        "rtol": rtol,
        "sample_count": len(rows),
        "validated_layers": sorted({row["layer"] for row in rows}),
        "validated_expert_count": len(unique_experts),
        "maximum_contribution_abs_error": maximum_contribution_error,
        "maximum_aggregate_moe_output_abs_error": maximum_aggregate_error,
        "validates_expert_ffn_math": True,
        "validates_tensorized_expert_indexing": True,
        "validates_activation_function": True,
        "validates_gate_handling": True,
        "validates_matrix_orientations": True,
        "validates_topk_coefficients": True,
        "validates_output_dimensions": True,
        "samples": rows,
    }
