from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .activation_quantization_cost import (
    replay_expert_outputs,
    selected_routes,
)
from .balanced import array_sha256, file_sha256
from .controlled import PreparedDomainExamples
from .expert_replay import (
    ReplayCapture,
    _decode_exact_tensor,
    _encode_exact_tensor,
)
from .hooks import tensor_output
from .io_utils import atomic_save_npz, atomic_write_json, read_json
from .modeling import ModelBundle, MoeLayerSpec
from .quantization import ExpertWeightLayout, ReversibleExpertQuantization, module_hook_count


GRADIENT_CAPTURE_SCHEMA_VERSION = 1
GRADIENT_COST_SCHEMA_VERSION = 1


@dataclass
class GradientCapture:
    domain: str
    model_layer_index: int
    gradients: torch.Tensor
    example_indices: np.ndarray
    token_positions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return int(self.gradients.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.gradients.shape[1])

    def validate(
        self,
        *,
        expected_tokens: int | None = None,
        expected_hidden_size: int | None = None,
    ) -> None:
        if self.gradients.ndim != 2 or not self.gradients.is_floating_point():
            raise ValueError("MoE-output gradients must have shape [token, hidden]")
        if self.example_indices.shape != (self.num_tokens,):
            raise ValueError("Gradient example indices have the wrong shape")
        if self.token_positions.shape != (self.num_tokens,):
            raise ValueError("Gradient token positions have the wrong shape")
        if expected_tokens is not None and self.num_tokens != expected_tokens:
            raise ValueError("Gradient token count differs from replay capture")
        if expected_hidden_size is not None and self.hidden_size != expected_hidden_size:
            raise ValueError("Gradient hidden size differs from replay capture")
        if not bool(torch.isfinite(self.gradients.float()).all()):
            raise ValueError("MoE-output gradients contain non-finite values")

    def save(self, path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
        self.validate()
        payload_array, storage_dtype, logical_dtype = _encode_exact_tensor(self.gradients)
        arrays = {
            "grad_y": payload_array,
            "example_indices": self.example_indices.astype(np.int32, copy=False),
            "token_positions": self.token_positions.astype(np.int16, copy=False),
        }
        atomic_save_npz(path, **arrays)
        payload = {
            **dict(metadata),
            "schema_version": GRADIENT_CAPTURE_SCHEMA_VERSION,
            "domain": self.domain,
            "model_layer_index": self.model_layer_index,
            "gradient_logical_dtype": logical_dtype,
            "gradient_storage_dtype": storage_dtype,
            "gradient_shape": list(self.gradients.shape),
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
    ) -> "GradientCapture":
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Gradient checkpoint is incomplete: {path}")
        metadata = read_json(metadata_path)
        if metadata.get("schema_version") != GRADIENT_CAPTURE_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported gradient schema in {path}")
        if expected_metadata:
            mismatches = [
                key
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            ]
            if mismatches:
                raise RuntimeError(
                    f"Gradient checkpoint fingerprint mismatch for {path}: "
                    + ", ".join(mismatches)
                )
        if metadata.get("npz_sha256") != file_sha256(path):
            raise RuntimeError(f"Gradient checkpoint hash mismatch for {path}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"grad_y", "example_indices", "token_positions"}:
                raise RuntimeError(f"Gradient checkpoint arrays are incomplete in {path}")
            arrays = {key: data[key] for key in data.files}
        for key, value in arrays.items():
            if metadata.get("array_sha256", {}).get(key) != array_sha256(value):
                raise RuntimeError(f"Gradient array hash mismatch for {path}:{key}")
        result = cls(
            domain=str(metadata["domain"]),
            model_layer_index=int(metadata["model_layer_index"]),
            gradients=_decode_exact_tensor(
                arrays["grad_y"], str(metadata["gradient_storage_dtype"])
            ),
            example_indices=arrays["example_indices"].astype(np.int64),
            token_positions=arrays["token_positions"].astype(np.int64),
            metadata=metadata,
        )
        result.validate()
        return result


@dataclass
class _GradientBatch:
    example_indices: np.ndarray
    measurement_mask: torch.Tensor
    sequence_length: int
    outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    collected: set[int] = field(default_factory=set)


class MoeOutputGradientSession:
    """Retain d(mean measured NLL)/d(y_moe) with one backward per example."""

    def __init__(self, layer_specs: Sequence[MoeLayerSpec]) -> None:
        if not layer_specs:
            raise ValueError("At least one MoE layer is required")
        self.layer_specs = list(layer_specs)
        self._handles: list[Any] = []
        self._batch: _GradientBatch | None = None
        self._entered = False
        self._gradients: dict[int, list[torch.Tensor]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._examples: dict[int, list[np.ndarray]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }
        self._positions: dict[int, list[np.ndarray]] = {
            spec.ordinal: [] for spec in self.layer_specs
        }

    def __enter__(self) -> "MoeOutputGradientSession":
        if self._entered:
            raise RuntimeError("Gradient capture cannot be entered twice")
        self._entered = True
        for spec in self.layer_specs:
            self._handles.append(
                spec.experts.register_forward_hook(
                    self._make_output_hook(spec), with_kwargs=True
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def batch(
        self, example_indices: Sequence[int], measurement_mask: torch.Tensor
    ) -> Iterator[None]:
        if not self._entered or not self._handles:
            raise RuntimeError("Enter gradient capture before starting a batch")
        if self._batch is not None:
            raise RuntimeError("Nested gradient batches are unsupported")
        indices = np.asarray(example_indices, dtype=np.int64)
        if measurement_mask.ndim != 2 or len(indices) != measurement_mask.shape[0]:
            raise ValueError("Gradient measurement geometry is invalid")
        self._batch = _GradientBatch(
            example_indices=indices,
            measurement_mask=measurement_mask.detach(),
            sequence_length=int(measurement_mask.shape[1]),
        )
        try:
            yield
            expected = {spec.ordinal for spec in self.layer_specs}
            if self._batch.collected != expected:
                missing = sorted(expected - self._batch.collected)
                raise RuntimeError(f"Gradient capture missed layer ordinals {missing}")
        finally:
            self._batch = None

    def collect_after_backward(self) -> None:
        batch = self._require_batch()
        for spec in self.layer_specs:
            output = batch.outputs.get(spec.ordinal)
            if output is None:
                raise RuntimeError(f"No retained MoE output for {spec.block_name}")
            gradient = output.grad
            if gradient is None:
                raise RuntimeError(f"No MoE-output gradient for {spec.block_name}")
            flat = gradient.reshape(-1, gradient.shape[-1])
            mask, local, positions = self._token_mapping(flat.shape[0], flat.device)
            selected = flat[mask].detach()
            global_examples = torch.as_tensor(
                batch.example_indices, dtype=torch.long, device=flat.device
            )[local[mask]]
            self._gradients[spec.ordinal].append(selected.cpu())
            self._examples[spec.ordinal].append(
                global_examples.cpu().numpy().astype(np.int64)
            )
            self._positions[spec.ordinal].append(
                positions[mask].cpu().numpy().astype(np.int64)
            )
            batch.collected.add(spec.ordinal)

    def finalize(self, domain: str) -> dict[int, GradientCapture]:
        output: dict[int, GradientCapture] = {}
        for spec in self.layer_specs:
            if not self._gradients[spec.ordinal]:
                raise RuntimeError(f"No gradients captured for {spec.block_name}")
            result = GradientCapture(
                domain=domain,
                model_layer_index=spec.model_layer_index,
                gradients=torch.cat(self._gradients[spec.ordinal], dim=0),
                example_indices=np.concatenate(self._examples[spec.ordinal]),
                token_positions=np.concatenate(self._positions[spec.ordinal]),
            )
            result.validate()
            output[spec.model_layer_index] = result
        return output

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._batch = None
        self._entered = False

    @property
    def registered_hook_count(self) -> int:
        return len(self._handles)

    def _make_output_hook(self, spec: MoeLayerSpec):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            batch = self._require_batch()
            value = tensor_output(output)
            if not value.requires_grad:
                raise RuntimeError(
                    f"MoE output for {spec.block_name} is not differentiable"
                )
            value.retain_grad()
            batch.outputs[spec.ordinal] = value

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
        raise RuntimeError("MoE-output gradient rows do not match measurement geometry")

    def _require_batch(self) -> _GradientBatch:
        if self._batch is None:
            raise RuntimeError("A gradient hook fired outside an active batch")
        return self._batch


def capture_gradient_dataset(
    bundle: ModelBundle,
    layer_specs: Sequence[MoeLayerSpec],
    prepared: Mapping[str, PreparedDomainExamples],
    replay_capture_dir: Path,
    output_dir: Path,
    *,
    capture_fingerprint: str,
    gradient_fingerprint: str,
    batch_size: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    if batch_size != 1:
        raise ValueError(
            "GQS is defined from one mean-NLL gradient per example; batch_size must be 1"
        )
    parameter_state = [
        (parameter, bool(parameter.requires_grad), int(parameter._version))
        for parameter in bundle.model.parameters()
    ]
    hooks_before = module_hook_count(bundle.model)
    for parameter, _, _ in parameter_state:
        parameter.requires_grad_(False)
    rows: list[dict[str, Any]] = []
    loss_by_domain: dict[str, list[float]] = {}
    try:
        for domain, examples in prepared.items():
            expected_common = {
                "gradient_fingerprint": gradient_fingerprint,
                "capture_fingerprint": capture_fingerprint,
                "domain": domain,
            }
            missing: list[MoeLayerSpec] = []
            for spec in layer_specs:
                path = output_dir / domain / f"layer_{spec.model_layer_index:02d}.npz"
                try:
                    if not resume:
                        raise FileNotFoundError
                    capture = GradientCapture.load(
                        path, expected_metadata=expected_common
                    )
                    replay = ReplayCapture.load(
                        replay_capture_dir
                        / domain
                        / f"layer_{spec.model_layer_index:02d}.npz",
                        expected_metadata={"capture_fingerprint": capture_fingerprint},
                    )
                    capture.validate(
                        expected_tokens=replay.num_tokens,
                        expected_hidden_size=replay.hidden_size,
                    )
                    if not np.array_equal(capture.example_indices, replay.example_indices):
                        raise RuntimeError("Gradient/replay example indices differ")
                    if not np.array_equal(capture.token_positions, replay.token_positions):
                        raise RuntimeError("Gradient/replay token positions differ")
                    rows.append(
                        {
                            "domain": domain,
                            "layer": spec.model_layer_index,
                            "path": str(path),
                            "npz_sha256": capture.metadata["npz_sha256"],
                            "resumed": True,
                        }
                    )
                except FileNotFoundError:
                    missing.append(spec)
            if not missing:
                continue
            session = MoeOutputGradientSession(missing)
            losses: list[float] = []
            with session:
                for example_index in range(examples.num_examples):
                    input_ids = torch.as_tensor(
                        examples.input_ids[example_index : example_index + 1],
                        dtype=torch.long,
                        device=bundle.runtime.device,
                    )
                    attention_mask = torch.as_tensor(
                        examples.attention_mask[example_index : example_index + 1],
                        dtype=torch.long,
                        device=bundle.runtime.device,
                    )
                    measurement_mask = torch.as_tensor(
                        examples.measurement_mask[example_index : example_index + 1],
                        dtype=torch.bool,
                        device=bundle.runtime.device,
                    )
                    embeddings = bundle.model.get_input_embeddings()(input_ids).detach()
                    embeddings.requires_grad_(True)
                    with session.batch([example_index], measurement_mask):
                        outputs = bundle.backbone(
                            inputs_embeds=embeddings,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        hidden = _first_tensor(outputs)
                        head = bundle.model.get_output_embeddings()
                        if head is None:
                            raise RuntimeError("Causal LM has no output embedding / LM head")
                        logits = head(hidden)
                        source_mask = measurement_mask[:, :-1]
                        selected_logits = logits[:, :-1, :][source_mask]
                        labels = input_ids[:, 1:][source_mask]
                        if selected_logits.shape[0] != int(
                            measurement_mask.sum().item()
                        ):
                            raise RuntimeError("Gradient loss positions are misaligned")
                        loss = F.cross_entropy(
                            selected_logits.float(), labels, reduction="mean"
                        )
                        loss.backward()
                        session.collect_after_backward()
                    losses.append(float(loss.detach().item()))
                    del loss, logits, hidden, outputs, embeddings
                    if (example_index + 1) % 10 == 0 or (
                        example_index + 1 == examples.num_examples
                    ):
                        print(
                            f"[gradient:{domain}] {example_index + 1}/"
                            f"{examples.num_examples} examples",
                            flush=True,
                        )
            captures = session.finalize(domain)
            if session.registered_hook_count != 0:
                raise RuntimeError("Gradient capture leaked hooks")
            loss_by_domain[domain] = losses
            for spec in missing:
                replay = ReplayCapture.load(
                    replay_capture_dir
                    / domain
                    / f"layer_{spec.model_layer_index:02d}.npz",
                    expected_metadata={"capture_fingerprint": capture_fingerprint},
                )
                capture = captures[spec.model_layer_index]
                capture.validate(
                    expected_tokens=replay.num_tokens,
                    expected_hidden_size=replay.hidden_size,
                )
                if not np.array_equal(capture.example_indices, replay.example_indices):
                    raise RuntimeError("Fresh gradient/replay example indices differ")
                if not np.array_equal(capture.token_positions, replay.token_positions):
                    raise RuntimeError("Fresh gradient/replay token positions differ")
                path = output_dir / domain / f"layer_{spec.model_layer_index:02d}.npz"
                metadata = capture.save(
                    path,
                    {
                        **expected_common,
                        "layer_ordinal": spec.ordinal,
                        "loss_definition": (
                            "per-example mean next-token NLL over exactly 64 measured "
                            "source positions"
                        ),
                        "model_parameters_frozen": True,
                    },
                )
                rows.append(
                    {
                        "domain": domain,
                        "layer": spec.model_layer_index,
                        "path": str(path),
                        "npz_sha256": metadata["npz_sha256"],
                        "resumed": False,
                    }
                )
    finally:
        version_mismatches: list[int] = []
        for index, (parameter, required_grad, version) in enumerate(parameter_state):
            parameter.requires_grad_(required_grad)
            if int(parameter._version) != version:
                version_mismatches.append(index)
        if version_mismatches:
            raise RuntimeError(
                "Model parameters changed during frozen gradient capture at indices: "
                + ", ".join(str(value) for value in version_mismatches[:20])
            )
    hooks_after = module_hook_count(bundle.model)
    if hooks_before != hooks_after:
        raise RuntimeError("Gradient fallback leaked model hooks")
    rows.sort(key=lambda row: (row["domain"], row["layer"]))
    return {
        "schema_version": GRADIENT_CAPTURE_SCHEMA_VERSION,
        "gradient_fingerprint": gradient_fingerprint,
        "capture_fingerprint": capture_fingerprint,
        "loss_definition": (
            "per-example mean next-token NLL over the frozen 64 measured positions"
        ),
        "one_backward_per_example": True,
        "batch_size": batch_size,
        "model_parameters_frozen": True,
        "model_parameter_versions_unchanged": True,
        "hooks_before": hooks_before,
        "hooks_after": hooks_after,
        "gradient_files": rows,
        "baseline_nll_by_domain": loss_by_domain,
    }


def calculate_gqs(
    baseline_output: torch.Tensor | np.ndarray,
    quantized_output: torch.Tensor | np.ndarray,
    gate_weights: torch.Tensor | np.ndarray,
    grad_y: torch.Tensor | np.ndarray,
    example_indices: np.ndarray,
    *,
    num_examples: int,
) -> tuple[float, float, np.ndarray]:
    baseline = torch.as_tensor(baseline_output).detach().double()
    quantized = torch.as_tensor(quantized_output).detach().double()
    gates = torch.as_tensor(gate_weights).detach().double().reshape(-1)
    gradients = torch.as_tensor(grad_y).detach().double()
    examples = np.asarray(example_indices, dtype=np.int64)
    if baseline.ndim != 2 or quantized.shape != baseline.shape:
        raise ValueError("GQS expert outputs must have matching [route, hidden] shapes")
    if gradients.shape != baseline.shape:
        raise ValueError("GQS gradients and expert perturbations have different shapes")
    if gates.shape != (baseline.shape[0],) or examples.shape != (baseline.shape[0],):
        raise ValueError("GQS route metadata is misaligned")
    if num_examples < 1 or (len(examples) and (
        int(examples.min()) < 0 or int(examples.max()) >= num_examples
    )):
        raise ValueError("GQS example indices are invalid")
    signed = np.zeros(num_examples, dtype=np.float64)
    if baseline.shape[0]:
        delta = (quantized - baseline) * gates[:, None]
        route_scores = (gradients * delta).sum(dim=-1).cpu().numpy()
        np.add.at(signed, examples, route_scores)
    gqs = float(np.mean(np.abs(signed)))
    gqs2 = float(np.mean(np.square(signed)))
    if not np.isfinite(gqs) or not np.isfinite(gqs2) or gqs < 0 or gqs2 < 0:
        raise RuntimeError("GQS produced an invalid score")
    return gqs, gqs2, signed


def evaluate_gradient_surrogate_for_panel(
    layer_specs: Sequence[MoeLayerSpec],
    layouts: Mapping[int, ExpertWeightLayout],
    replay_capture_dir: Path,
    gradient_capture_dir: Path,
    domains: Sequence[str],
    panel: Sequence[Mapping[str, Any]],
    bit_widths: Sequence[int],
    *,
    capture_fingerprint: str,
    gradient_fingerprint: str,
    num_examples: int,
    group_size: int = 128,
    chunk_size: int = 512,
    expected_qdq_fingerprints: Mapping[tuple[int, int, int], Mapping[str, str]]
    | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    bits = [int(value) for value in bit_widths]
    shape = (len(bits), len(panel), len(domains))
    raw = {
        "gqs": np.zeros(shape, dtype=np.float64),
        "gqs2": np.zeros(shape, dtype=np.float64),
        "route_counts": np.zeros(shape, dtype=np.int64),
        "unobserved": np.zeros(shape, dtype=np.bool_),
        "signed_first_order_by_example": np.zeros(
            (*shape, num_examples), dtype=np.float64
        ),
    }
    specs = {spec.model_layer_index: spec for spec in layer_specs}
    replay_cache: dict[tuple[str, int], ReplayCapture] = {}
    gradient_cache: dict[tuple[str, int], GradientCapture] = {}
    qdq_rows: list[dict[str, Any]] = []

    def captures(domain: str, layer: int) -> tuple[ReplayCapture, GradientCapture]:
        key = (domain, layer)
        if key not in replay_cache:
            replay_cache[key] = ReplayCapture.load(
                replay_capture_dir / domain / f"layer_{layer:02d}.npz",
                expected_metadata={"capture_fingerprint": capture_fingerprint},
            )
            gradient_cache[key] = GradientCapture.load(
                gradient_capture_dir / domain / f"layer_{layer:02d}.npz",
                expected_metadata={
                    "capture_fingerprint": capture_fingerprint,
                    "gradient_fingerprint": gradient_fingerprint,
                },
            )
            if not np.array_equal(
                replay_cache[key].example_indices, gradient_cache[key].example_indices
            ) or not np.array_equal(
                replay_cache[key].token_positions, gradient_cache[key].token_positions
            ):
                raise RuntimeError("Gradient capture is not aligned with replay capture")
        return replay_cache[key], gradient_cache[key]

    for panel_index, intervention in enumerate(panel):
        layer = int(intervention["layer"])
        expert_id = int(intervention["expert_id"])
        spec = specs[layer]
        routed: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]] = {}
        baseline: dict[str, torch.Tensor] = {}
        for domain in domains:
            replay, gradient = captures(domain, layer)
            hidden, gates, examples = selected_routes(replay, expert_id)
            rows, _ = replay.route_rows(expert_id)
            routed[domain] = (hidden, gates, examples, gradient.gradients[rows])
            baseline[domain] = replay_expert_outputs(
                spec, expert_id, hidden, chunk_size=chunk_size
            )
        for bit_index, bit_width in enumerate(bits):
            if bit_width == 16:
                for domain_index, domain in enumerate(domains):
                    count = int(routed[domain][0].shape[0])
                    raw["route_counts"][bit_index, panel_index, domain_index] = count
                    raw["unobserved"][bit_index, panel_index, domain_index] = count == 0
                continue
            context = ReversibleExpertQuantization(
                layouts[layer],
                expert_id,
                bit_width,
                group_size,
                verify_unrelated_experts=False,
            )
            with context:
                expected = (
                    expected_qdq_fingerprints.get((layer, expert_id, bit_width))
                    if expected_qdq_fingerprints is not None
                    else None
                )
                if expected is not None and (
                    context.original_fingerprint != expected.get("original")
                    or context.quantized_fingerprint != expected.get("quantized")
                ):
                    raise RuntimeError(
                        f"GQS replay QDQ differs from Stage 1 for L{layer}/E{expert_id}"
                    )
                for domain_index, domain in enumerate(domains):
                    hidden, gates, examples, gradients = routed[domain]
                    quantized = replay_expert_outputs(
                        spec, expert_id, hidden, chunk_size=chunk_size
                    )
                    gqs, gqs2, signed = calculate_gqs(
                        baseline[domain],
                        quantized,
                        gates,
                        gradients,
                        examples,
                        num_examples=num_examples,
                    )
                    raw["gqs"][bit_index, panel_index, domain_index] = gqs
                    raw["gqs2"][bit_index, panel_index, domain_index] = gqs2
                    raw["route_counts"][bit_index, panel_index, domain_index] = len(
                        examples
                    )
                    raw["unobserved"][bit_index, panel_index, domain_index] = (
                        len(examples) == 0
                    )
                    raw["signed_first_order_by_example"][
                        bit_index, panel_index, domain_index
                    ] = signed
            diagnostics = context.diagnostics()
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
                    "exact_restoration_verified": diagnostics[
                        "exact_restoration_verified"
                    ],
                }
            )
    return raw, qdq_rows


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    value = getattr(output, "last_hidden_state", None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError("Could not extract backbone hidden state")
