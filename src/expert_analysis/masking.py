from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .controlled import PreparedDomainExamples
from .io_utils import atomic_save_npz, atomic_write_json, read_json, write_csv
from .metrics import DomainStatistics
from .modeling import ModelBundle, MoeLayerSpec
from .statistics import confidence_interval, descending_ranks, safe_spearman


@dataclass(frozen=True)
class MaskTarget:
    model_layer_index: int
    expert_id: int
    expected_high_domain: str | None = None
    expected_low_domain: str | None = None

    @property
    def label(self) -> str:
        return f"L{self.model_layer_index}/E{self.expert_id}"

    @property
    def slug(self) -> str:
        return f"layer_{self.model_layer_index}_expert_{self.expert_id}"

    @classmethod
    def parse(cls, value: str) -> "MaskTarget":
        pieces = value.replace("/", ":").split(":")
        if len(pieces) not in (2, 4):
            raise ValueError(
                f"Invalid mask target {value!r}; expected LAYER:EXPERT or "
                "LAYER:EXPERT:HIGH_DOMAIN:LOW_DOMAIN"
            )
        try:
            layer, expert = (int(item) for item in pieces[:2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid mask target {value!r}; layer and expert must be integers"
            ) from exc
        if layer < 0 or expert < 0:
            raise ValueError("Mask layer and expert IDs must be nonnegative")
        high_domain = pieces[2] if len(pieces) == 4 else None
        low_domain = pieces[3] if len(pieces) == 4 else None
        if (high_domain is None) != (low_domain is None) or high_domain == low_domain:
            raise ValueError("Mask contrast domains must be distinct and supplied together")
        return cls(layer, expert, high_domain, low_domain)


@dataclass
class LossStatistics:
    loss_sums: np.ndarray
    token_counts: np.ndarray
    route_counts: np.ndarray | None = None
    zeroed_gate_mass: np.ndarray | None = None

    def validate(self) -> None:
        if self.loss_sums.ndim != 1 or self.token_counts.shape != self.loss_sums.shape:
            raise ValueError("Loss arrays must contain one value per example")
        if not np.all(np.isfinite(self.loss_sums)) or np.any(self.loss_sums < 0):
            raise ValueError("loss_sums contains invalid values")
        if np.any(self.token_counts <= 0):
            raise ValueError("Every loss example must contain a prediction target")
        for name, values in (
            ("route_counts", self.route_counts),
            ("zeroed_gate_mass", self.zeroed_gate_mass),
        ):
            if values is not None:
                if values.shape != self.loss_sums.shape:
                    raise ValueError(f"{name} shape does not match loss_sums")
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"{name} contains invalid values")

    @property
    def per_token_nll(self) -> np.ndarray:
        return self.loss_sums.astype(np.float64) / self.token_counts.astype(np.float64)

    def save(self, path: Path) -> None:
        self.validate()
        arrays: dict[str, Any] = {
            "loss_sums": self.loss_sums.astype(np.float64, copy=False),
            "token_counts": self.token_counts.astype(np.uint32, copy=False),
        }
        if self.route_counts is not None:
            arrays["route_counts"] = self.route_counts.astype(np.uint32, copy=False)
        if self.zeroed_gate_mass is not None:
            arrays["zeroed_gate_mass"] = self.zeroed_gate_mass.astype(
                np.float64, copy=False
            )
        atomic_save_npz(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "LossStatistics":
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                loss_sums=data["loss_sums"],
                token_counts=data["token_counts"],
                route_counts=data["route_counts"] if "route_counts" in data else None,
                zeroed_gate_mass=(
                    data["zeroed_gate_mass"]
                    if "zeroed_gate_mass" in data
                    else None
                ),
            )
        result.validate()
        return result


class ExpertGateMask:
    """Reversibly zero one expert's selected gate weights at measured positions."""

    def __init__(self, spec: MoeLayerSpec, expert_id: int) -> None:
        if not 0 <= expert_id < spec.num_experts:
            raise ValueError(
                f"Expert {expert_id} is outside layer {spec.model_layer_index}'s "
                f"0..{spec.num_experts - 1} range"
            )
        self.spec = spec
        self.expert_id = expert_id
        self._handle: Any | None = None
        self._measurement_mask: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._batch_route_counts: np.ndarray | None = None
        self._batch_gate_mass: np.ndarray | None = None
        self._batch_calls = 0
        self.total_calls = 0
        self.total_zeroed_routes = 0
        self.total_zeroed_gate_mass = 0.0

    def __enter__(self) -> "ExpertGateMask":
        if self._handle is not None:
            raise RuntimeError("ExpertGateMask cannot be entered twice")
        self._handle = self.spec.router.register_forward_hook(
            self._router_hook, with_kwargs=True
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def registered_hook_count(self) -> int:
        return int(self._handle is not None)

    @property
    def batch_route_counts(self) -> np.ndarray:
        if self._batch_route_counts is None:
            raise RuntimeError("No completed masking batch is available")
        return self._batch_route_counts

    @property
    def batch_gate_mass(self) -> np.ndarray:
        if self._batch_gate_mass is None:
            raise RuntimeError("No completed masking batch is available")
        return self._batch_gate_mass

    @contextmanager
    def batch(
        self, measurement_mask: torch.Tensor, attention_mask: torch.Tensor
    ) -> Iterator[None]:
        if self._handle is None:
            raise RuntimeError("Enter ExpertGateMask before starting a batch")
        if self._measurement_mask is not None:
            raise RuntimeError("Nested masking batches are not supported")
        if measurement_mask.shape != attention_mask.shape or measurement_mask.ndim != 2:
            raise ValueError("Masking masks must share shape [batch, sequence]")
        self._measurement_mask = measurement_mask.detach().bool()
        self._attention_mask = attention_mask.detach().bool()
        self._batch_route_counts = np.zeros(measurement_mask.shape[0], dtype=np.uint32)
        self._batch_gate_mass = np.zeros(measurement_mask.shape[0], dtype=np.float64)
        self._batch_calls = 0
        try:
            yield
            if self._batch_calls != 1:
                raise RuntimeError(
                    f"Expected router {self.spec.router_name} to run once, observed "
                    f"{self._batch_calls} calls"
                )
        finally:
            self._measurement_mask = None
            self._attention_mask = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._measurement_mask = None
        self._attention_mask = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "layer": self.spec.model_layer_index,
            "layer_ordinal": self.spec.ordinal,
            "layer_name": self.spec.block_name,
            "expert_id": self.expert_id,
            "intervention": "zero_selected_gate_weight_without_rerouting",
            "router_calls": self.total_calls,
            "zeroed_routes": self.total_zeroed_routes,
            "zeroed_gate_mass": self.total_zeroed_gate_mass,
            "hooks_remaining": self.registered_hook_count,
        }

    def _router_hook(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        del module, args, kwargs
        if self._measurement_mask is None or self._attention_mask is None:
            raise RuntimeError("Expert mask hook fired outside an active masking batch")
        indices_source, weights_source = _exposed_selected_routes(output, self.spec)
        indices = indices_source.reshape(-1, self.spec.top_k).long()
        weights = weights_source.reshape(-1, self.spec.top_k)
        measured_rows, local_examples = self._router_row_mapping(indices.shape[0])
        selected = (indices == self.expert_id) & measured_rows[:, None]

        new_weights = weights.clone()
        new_weights[selected] = 0
        if bool((new_weights[selected] != 0).any()):
            raise RuntimeError("Expert mask failed to zero every selected route")
        if not torch.equal(new_weights[~selected], weights[~selected]):
            raise RuntimeError("Expert mask modified non-target routing weights")
        replacement = new_weights.reshape_as(weights_source)
        replaced, replacements = _replace_tensor(output, weights_source, replacement)
        if replacements != 1:
            raise RuntimeError(
                "Could not uniquely replace selected router weights; found "
                f"{replacements} references"
            )

        assert self._batch_route_counts is not None
        assert self._batch_gate_mass is not None
        per_row_counts = selected.sum(dim=1).to(dtype=torch.float32)
        per_row_mass = (weights.float() * selected).sum(dim=1)
        batch_size = int(self._measurement_mask.shape[0])
        counts = torch.zeros(batch_size, dtype=torch.float32, device=weights.device)
        mass = torch.zeros(batch_size, dtype=torch.float32, device=weights.device)
        counts.scatter_add_(0, local_examples, per_row_counts)
        mass.scatter_add_(0, local_examples, per_row_mass)
        count_values = counts.to("cpu", dtype=torch.int64).numpy().astype(np.uint32)
        mass_values = mass.to("cpu", dtype=torch.float64).numpy()
        self._batch_route_counts[:] = count_values
        self._batch_gate_mass[:] = mass_values
        self._batch_calls += 1
        self.total_calls += 1
        self.total_zeroed_routes += int(count_values.sum())
        self.total_zeroed_gate_mass += float(mass_values.sum())
        return replaced

    def _router_row_mapping(
        self, router_rows: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._measurement_mask is not None
        assert self._attention_mask is not None
        device = self._measurement_mask.device
        measured = self._measurement_mask.reshape(-1)
        attended = self._attention_mask.reshape(-1)
        batch_size, sequence_length = self._measurement_mask.shape
        local = torch.arange(batch_size, device=device).repeat_interleave(
            sequence_length
        )
        if router_rows == measured.numel():
            return measured, local
        attended_count = int(attended.sum().item())
        if router_rows == attended_count:
            return measured[attended], local[attended]
        raise RuntimeError(
            f"Router produced {router_rows} rows for {measured.numel()} total and "
            f"{attended_count} attended tokens"
        )


def resolve_mask_targets(
    layer_specs: Sequence[MoeLayerSpec], targets: Sequence[MaskTarget]
) -> list[tuple[MaskTarget, MoeLayerSpec]]:
    by_model_layer = {spec.model_layer_index: spec for spec in layer_specs}
    resolved: list[tuple[MaskTarget, MoeLayerSpec]] = []
    seen: dict[tuple[int, int], MaskTarget] = {}
    for target in targets:
        key = (target.model_layer_index, target.expert_id)
        previous = seen.get(key)
        if previous == target:
            continue
        if previous is not None:
            raise ValueError(
                f"Mask target {target.label} was supplied with conflicting "
                "pre-registered domain contrasts"
            )
        seen[key] = target
        spec = by_model_layer.get(target.model_layer_index)
        if spec is None:
            raise ValueError(
                f"Mask target {target.label} does not identify a discovered MoE layer"
            )
        if target.expert_id >= spec.num_experts:
            raise ValueError(
                f"Mask target {target.label} exceeds layer expert count {spec.num_experts}"
            )
        resolved.append((target, spec))
    if not resolved:
        raise ValueError("At least one expert mask target is required")
    return resolved


def evaluate_next_token_loss(
    bundle: ModelBundle,
    examples: PreparedDomainExamples,
    batch_size: int,
    mask_spec: MoeLayerSpec | None = None,
    expert_id: int | None = None,
) -> tuple[LossStatistics, dict[str, Any]]:
    examples.validate()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (mask_spec is None) != (expert_id is None):
        raise ValueError("mask_spec and expert_id must either both be set or both be omitted")
    before_hooks = _model_hook_count(bundle.model)
    loss_sums = np.zeros(examples.num_examples, dtype=np.float64)
    token_counts = np.zeros(examples.num_examples, dtype=np.uint32)
    route_counts = (
        np.zeros(examples.num_examples, dtype=np.uint32)
        if mask_spec is not None
        else None
    )
    gate_mass = (
        np.zeros(examples.num_examples, dtype=np.float64)
        if mask_spec is not None
        else None
    )
    masker = ExpertGateMask(mask_spec, int(expert_id)) if mask_spec is not None else None
    started = time.monotonic()
    context = masker if masker is not None else _null_context()
    with context:
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
                dtype=torch.long,
                device=bundle.runtime.device,
            )
            if masker is None:
                batch_loss, batch_tokens = _next_token_loss_batch(
                    bundle, input_ids, attention_mask, measurement_mask
                )
            else:
                with masker.batch(measurement_mask, attention_mask):
                    batch_loss, batch_tokens = _next_token_loss_batch(
                        bundle, input_ids, attention_mask, measurement_mask
                    )
                assert route_counts is not None and gate_mass is not None
                route_counts[start:stop] = masker.batch_route_counts
                gate_mass[start:stop] = masker.batch_gate_mass
            loss_sums[start:stop] = batch_loss
            token_counts[start:stop] = batch_tokens
            if stop == examples.num_examples or stop % max(batch_size, 10) == 0:
                label = (
                    f"mask L{mask_spec.model_layer_index}/E{expert_id}"
                    if mask_spec is not None
                    else "baseline"
                )
                print(
                    f"[{examples.domain}] {label}: {stop}/{examples.num_examples} "
                    f"examples ({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
    after_hooks = _model_hook_count(bundle.model)
    if after_hooks != before_hooks or (masker and masker.registered_hook_count != 0):
        raise RuntimeError(
            f"Masking hook leak detected: before={before_hooks}, after={after_hooks}"
        )
    result = LossStatistics(loss_sums, token_counts, route_counts, gate_mass)
    result.validate()
    expected = examples.measurement_mask.sum(axis=1).astype(np.uint32)
    if not np.array_equal(result.token_counts, expected):
        raise RuntimeError("Loss token counts do not match the controlled measurement mask")
    diagnostics = {
        "elapsed_seconds": time.monotonic() - started,
        "hooks_before": before_hooks,
        "hooks_after": after_hooks,
        "mask": masker.diagnostics() if masker is not None else None,
    }
    return result, diagnostics


def run_masking_validation(
    bundle: ModelBundle,
    layer_specs: list[MoeLayerSpec],
    prepared_domains: Mapping[str, PreparedDomainExamples],
    importance_statistics: Mapping[str, DomainStatistics],
    targets: Sequence[MaskTarget],
    output_dir: Path,
    collection_fingerprint: str,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
    resume: bool = True,
) -> dict[str, Any]:
    resolved = resolve_mask_targets(layer_specs, targets)
    available_domains = set(prepared_domains)
    for target, _ in resolved:
        expected = {target.expected_high_domain, target.expected_low_domain} - {None}
        if not expected.issubset(available_domains):
            raise ValueError(
                f"Mask target {target.label} expects domains {sorted(expected)}, but "
                f"the run contains {sorted(available_domains)}"
            )
    masking_basis = {
        "collection_fingerprint": collection_fingerprint,
        "targets": [
            {
                "layer": target.model_layer_index,
                "expert_id": target.expert_id,
                "expected_high_domain": target.expected_high_domain,
                "expected_low_domain": target.expected_low_domain,
            }
            for target, _ in resolved
        ],
        "method": "zero_selected_gate_weight_without_rerouting",
        "scope": "measured_next_token_source_positions_only",
    }
    masking_fingerprint = hashlib.sha256(
        json.dumps(masking_basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mask_dir = output_dir / "masking"
    atomic_write_json(
        mask_dir / "masking_config.json",
        {
            **masking_basis,
            "masking_fingerprint": masking_fingerprint,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
        },
    )

    baselines: dict[str, LossStatistics] = {}
    raw_masked: dict[tuple[int, int, str], LossStatistics] = {}
    diagnostics: list[dict[str, Any]] = []
    for domain, examples in prepared_domains.items():
        baseline, diagnostic = _load_or_evaluate_loss(
            bundle,
            examples,
            path=mask_dir / "baseline" / f"{domain}.npz",
            metadata_path=mask_dir / "baseline" / f"{domain}.metadata.json",
            fingerprint=masking_fingerprint,
            batch_size=batch_size,
            resume=resume,
        )
        baselines[domain] = baseline
        diagnostics.append({"domain": domain, "target": "baseline", **diagnostic})

    for target, spec in resolved:
        for domain, examples in prepared_domains.items():
            result, diagnostic = _load_or_evaluate_loss(
                bundle,
                examples,
                path=mask_dir / target.slug / f"{domain}.npz",
                metadata_path=mask_dir / target.slug / f"{domain}.metadata.json",
                fingerprint=masking_fingerprint,
                batch_size=batch_size,
                resume=resume,
                mask_spec=spec,
                expert_id=target.expert_id,
            )
            if result.route_counts is None:
                raise RuntimeError(f"Mask target {target.label} stored no route counts")
            expected_routes = importance_statistics[domain].routing_counts[
                :, spec.ordinal, target.expert_id
            ]
            if not np.array_equal(result.route_counts, expected_routes):
                raise RuntimeError(
                    f"Mask route counts for {target.label}/{domain} do not match the "
                    "previously collected routing assignments"
                )
            raw_masked[(target.model_layer_index, target.expert_id, domain)] = result
            diagnostics.append(
                {"domain": domain, "target": target.label, **diagnostic}
            )

    rows, contrasts = analyze_masking_results(
        importance_statistics,
        baselines,
        raw_masked,
        resolved,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    write_csv(
        output_dir / "expert_masking_loss.csv",
        rows,
        [
            "layer",
            "layer_ordinal",
            "expert_id",
            "domain",
            "examples",
            "evaluated_tokens",
            "baseline_nll",
            "masked_nll",
            "delta_nll",
            "delta_nll_ci_low",
            "delta_nll_ci_high",
            "relative_delta_percent",
            "positive_delta_example_fraction",
            "masked_routes",
            "fraction_tokens_routed",
            "routing_frequency",
            "functional_contribution",
            "normalized_contribution",
            "functional_rank",
            "bootstrap_replicates",
        ],
    )
    write_csv(
        output_dir / "expert_masking_domain_contrasts.csv",
        contrasts,
        [
            "layer",
            "layer_ordinal",
            "expert_id",
            "proxy_high_domain",
            "proxy_low_domain",
            "contrast_high_domain",
            "contrast_low_domain",
            "contrast_source",
            "proxy_high_matches_preregistered",
            "proxy_low_matches_preregistered",
            "high_domain_delta_nll",
            "high_domain_delta_nll_ci_low",
            "high_domain_delta_nll_ci_high",
            "low_domain_delta_nll",
            "low_domain_delta_nll_ci_low",
            "low_domain_delta_nll_ci_high",
            "high_minus_low_delta_nll",
            "contrast_ci_low",
            "contrast_ci_high",
            "proxy_loss_spearman",
            "direction_aligned",
            "high_domain_loss_harm_ci_excludes_zero",
            "positive_contrast_ci_excludes_zero",
            "causal_specialization_supported",
            "largest_loss_delta_domain",
            "smallest_loss_delta_domain",
            "bootstrap_replicates",
        ],
    )
    result = {
        "masking_fingerprint": masking_fingerprint,
        "method": masking_basis["method"],
        "scope": masking_basis["scope"],
        "targets": masking_basis["targets"],
        "bootstrap_replicates": bootstrap_replicates,
        "loss_rows": rows,
        "domain_contrasts": contrasts,
        "diagnostics": diagnostics,
    }
    atomic_write_json(output_dir / "masking_results.json", result)
    return result


def analyze_masking_results(
    importance_statistics: Mapping[str, DomainStatistics],
    baselines: Mapping[str, LossStatistics],
    masked: Mapping[tuple[int, int, str], LossStatistics],
    resolved_targets: Sequence[tuple[MaskTarget, MoeLayerSpec]],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    deltas: dict[tuple[int, int, str], np.ndarray] = {}
    aggregate = {
        domain: statistics.aggregate()
        for domain, statistics in importance_statistics.items()
    }
    rng = np.random.default_rng(seed)
    for target, spec in resolved_targets:
        for domain, baseline in baselines.items():
            intervention = masked[(target.model_layer_index, target.expert_id, domain)]
            if not np.array_equal(baseline.token_counts, intervention.token_counts):
                raise RuntimeError("Baseline and masked loss token counts differ")
            per_example_delta = intervention.per_token_nll - baseline.per_token_nll
            deltas[(target.model_layer_index, target.expert_id, domain)] = per_example_delta
            bootstrap = _bootstrap_mean(per_example_delta, bootstrap_replicates, rng)
            _, low, high = confidence_interval(bootstrap)
            baseline_nll = float(baseline.loss_sums.sum() / baseline.token_counts.sum())
            masked_nll = float(
                intervention.loss_sums.sum() / intervention.token_counts.sum()
            )
            normalized = aggregate[domain]["normalized_contribution"][spec.ordinal]
            ranks = descending_ranks(normalized)
            route_counts = intervention.route_counts
            assert route_counts is not None
            total_tokens = int(intervention.token_counts.sum())
            rows.append(
                {
                    "layer": target.model_layer_index,
                    "layer_ordinal": spec.ordinal,
                    "expert_id": target.expert_id,
                    "domain": domain,
                    "examples": len(per_example_delta),
                    "evaluated_tokens": total_tokens,
                    "baseline_nll": baseline_nll,
                    "masked_nll": masked_nll,
                    "delta_nll": masked_nll - baseline_nll,
                    "delta_nll_ci_low": low,
                    "delta_nll_ci_high": high,
                    "relative_delta_percent": (
                        100.0 * (masked_nll - baseline_nll) / baseline_nll
                        if baseline_nll > 0
                        else float("nan")
                    ),
                    "positive_delta_example_fraction": float(
                        np.mean(per_example_delta > 0)
                    ),
                    "masked_routes": int(route_counts.sum()),
                    "fraction_tokens_routed": float(route_counts.sum() / total_tokens),
                    "routing_frequency": float(
                        aggregate[domain]["routing_frequency"][
                            spec.ordinal, target.expert_id
                        ]
                    ),
                    "functional_contribution": float(
                        aggregate[domain]["functional_contribution"][
                            spec.ordinal, target.expert_id
                        ]
                    ),
                    "normalized_contribution": float(normalized[target.expert_id]),
                    "functional_rank": float(ranks[target.expert_id]),
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )

    contrasts: list[dict[str, Any]] = []
    for target, spec in resolved_targets:
        target_rows = [
            row
            for row in rows
            if row["layer"] == target.model_layer_index
            and row["expert_id"] == target.expert_id
        ]
        ordered_domains = [row["domain"] for row in target_rows]
        proxy = np.asarray(
            [row["normalized_contribution"] for row in target_rows], dtype=float
        )
        loss_delta = np.asarray([row["delta_nll"] for row in target_rows], dtype=float)
        proxy_high_index = int(np.argmax(proxy))
        proxy_low_index = int(np.argmin(proxy))
        proxy_high_domain = ordered_domains[proxy_high_index]
        proxy_low_domain = ordered_domains[proxy_low_index]
        if target.expected_high_domain is not None:
            high_domain = target.expected_high_domain
            low_domain = target.expected_low_domain
            if high_domain not in ordered_domains or low_domain not in ordered_domains:
                raise RuntimeError(
                    f"Pre-registered contrast for {target.label} refers to a domain "
                    "that was not collected"
                )
            contrast_source = "pre_registered_prompt_only_run"
        else:
            high_domain = proxy_high_domain
            low_domain = proxy_low_domain
            contrast_source = "controlled_proxy_extrema_exploratory"
        rows_by_domain = {row["domain"]: row for row in target_rows}
        high_row = rows_by_domain[high_domain]
        low_row = rows_by_domain[low_domain]
        high_values = deltas[
            (target.model_layer_index, target.expert_id, high_domain)
        ]
        low_values = deltas[
            (target.model_layer_index, target.expert_id, low_domain)
        ]
        contrast_bootstrap = _bootstrap_independent_contrast(
            high_values, low_values, bootstrap_replicates, rng
        )
        _, contrast_low, contrast_high = confidence_interval(contrast_bootstrap)
        contrast = float(high_values.mean() - low_values.mean())
        high_domain_harm = bool(high_row["delta_nll_ci_low"] > 0)
        positive_contrast = bool(contrast_low > 0)
        contrasts.append(
            {
                "layer": target.model_layer_index,
                "layer_ordinal": spec.ordinal,
                "expert_id": target.expert_id,
                "proxy_high_domain": proxy_high_domain,
                "proxy_low_domain": proxy_low_domain,
                "contrast_high_domain": high_domain,
                "contrast_low_domain": low_domain,
                "contrast_source": contrast_source,
                "proxy_high_matches_preregistered": bool(
                    proxy_high_domain == high_domain
                ),
                "proxy_low_matches_preregistered": bool(proxy_low_domain == low_domain),
                "high_domain_delta_nll": float(high_values.mean()),
                "high_domain_delta_nll_ci_low": high_row["delta_nll_ci_low"],
                "high_domain_delta_nll_ci_high": high_row["delta_nll_ci_high"],
                "low_domain_delta_nll": float(low_values.mean()),
                "low_domain_delta_nll_ci_low": low_row["delta_nll_ci_low"],
                "low_domain_delta_nll_ci_high": low_row["delta_nll_ci_high"],
                "high_minus_low_delta_nll": contrast,
                "contrast_ci_low": contrast_low,
                "contrast_ci_high": contrast_high,
                "proxy_loss_spearman": safe_spearman(proxy, loss_delta),
                "direction_aligned": bool(contrast > 0),
                "high_domain_loss_harm_ci_excludes_zero": high_domain_harm,
                "positive_contrast_ci_excludes_zero": positive_contrast,
                "causal_specialization_supported": bool(
                    high_domain_harm and positive_contrast
                ),
                "largest_loss_delta_domain": ordered_domains[int(np.argmax(loss_delta))],
                "smallest_loss_delta_domain": ordered_domains[int(np.argmin(loss_delta))],
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
    return rows, contrasts


def validate_masking_mechanism(
    bundle: ModelBundle,
    spec: MoeLayerSpec,
    examples: PreparedDomainExamples,
) -> dict[str, Any]:
    """Find a routed expert on one example and verify reversible gate zeroing."""
    input_ids = torch.as_tensor(
        examples.input_ids[:1], dtype=torch.long, device=bundle.runtime.device
    )
    attention = torch.as_tensor(
        examples.attention_mask[:1], dtype=torch.long, device=bundle.runtime.device
    )
    measurement = torch.as_tensor(
        examples.measurement_mask[:1], dtype=torch.long, device=bundle.runtime.device
    )
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs
        indices, weights = _exposed_selected_routes(output, spec)
        captured.append((indices.detach(), weights.detach()))

    before_hooks = _model_hook_count(bundle.model)
    handle = spec.router.register_forward_hook(capture, with_kwargs=True)
    try:
        baseline_loss, baseline_tokens = _next_token_loss_batch(
            bundle, input_ids, attention, measurement
        )
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Masking validation could not observe selected expert IDs")
    rows = captured[0][0].reshape(-1, spec.top_k)
    weights = captured[0][1].reshape(-1, spec.top_k).float()
    measured_rows = measurement.bool().reshape(-1)
    routed = rows[measured_rows]
    routed_weights = weights[measured_rows]
    if routed.numel() == 0:
        raise RuntimeError("Masking validation example contains no measured routes")
    expert_mass = torch.zeros(spec.num_experts, device=routed.device)
    expert_mass.scatter_add_(0, routed.reshape(-1).long(), routed_weights.reshape(-1))
    expert_id = int(expert_mass.argmax().item())
    masker = ExpertGateMask(spec, expert_id)
    with masker:
        with masker.batch(measurement, attention):
            masked_loss, masked_tokens = _next_token_loss_batch(
                bundle, input_ids, attention, measurement
            )
    after_hooks = _model_hook_count(bundle.model)
    if before_hooks != after_hooks or masker.registered_hook_count != 0:
        raise RuntimeError("Masking validation leaked model hooks")
    if not np.array_equal(baseline_tokens, masked_tokens):
        raise RuntimeError("Masking validation changed evaluated token counts")
    if int(masker.batch_route_counts.sum()) == 0:
        raise RuntimeError("Masking validation did not zero a selected route")
    if not np.all(np.isfinite(masked_loss)):
        raise RuntimeError("Masking validation produced non-finite loss")
    absolute_loss_change = float(np.abs(masked_loss - baseline_loss).sum())
    if absolute_loss_change == 0:
        raise RuntimeError(
            "Selected-route weights were zeroed, but model loss was unchanged; "
            "the router output may not control the active expert computation"
        )
    return {
        "passed": True,
        "layer": spec.model_layer_index,
        "expert_id": expert_id,
        "zeroed_routes": int(masker.batch_route_counts.sum()),
        "baseline_nll": float(baseline_loss.sum() / baseline_tokens.sum()),
        "masked_nll": float(masked_loss.sum() / masked_tokens.sum()),
        "absolute_loss_sum_change": absolute_loss_change,
        "loss_changed": True,
        "hooks_before": before_hooks,
        "hooks_after": after_hooks,
    }


def _load_or_evaluate_loss(
    bundle: ModelBundle,
    examples: PreparedDomainExamples,
    path: Path,
    metadata_path: Path,
    fingerprint: str,
    batch_size: int,
    resume: bool,
    mask_spec: MoeLayerSpec | None = None,
    expert_id: int | None = None,
) -> tuple[LossStatistics, dict[str, Any]]:
    if resume and path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("masking_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing masking artifact {path} has a different configuration"
            )
        result = LossStatistics.load(path)
        if len(result.loss_sums) != examples.num_examples:
            raise RuntimeError(f"Existing masking artifact {path} has the wrong size")
        print(f"[{examples.domain}] resume: {path.parent.name}", flush=True)
        return result, {"resumed": True, **metadata.get("diagnostics", {})}
    result, diagnostics = evaluate_next_token_loss(
        bundle,
        examples,
        batch_size=batch_size,
        mask_spec=mask_spec,
        expert_id=expert_id,
    )
    result.save(path)
    atomic_write_json(
        metadata_path,
        {
            "masking_fingerprint": fingerprint,
            "domain": examples.domain,
            "layer": mask_spec.model_layer_index if mask_spec is not None else None,
            "expert_id": expert_id,
            "num_examples": examples.num_examples,
            "diagnostics": diagnostics,
        },
    )
    return result, {"resumed": False, **diagnostics}


def _next_token_loss_batch(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    measurement_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        output = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                logits = output[0]
            else:
                raise RuntimeError("Causal LM returned no logits for masking evaluation")
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise RuntimeError("Causal LM logits have an unexpected shape")
        labels = input_ids[:, 1:]
        prediction_mask = measurement_mask[:, :-1].bool()
        prediction_mask &= attention_mask[:, :-1].bool()
        prediction_mask &= attention_mask[:, 1:].bool()
        token_loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        if not torch.isfinite(token_loss[prediction_mask]).all():
            raise RuntimeError("Model produced non-finite next-token losses")
        loss_sums = (token_loss * prediction_mask).sum(dim=1)
        token_counts = prediction_mask.sum(dim=1)
        result_loss = loss_sums.to("cpu", dtype=torch.float64).numpy()
        result_counts = token_counts.to("cpu", dtype=torch.int64).numpy().astype(np.uint32)
        del output, logits, labels, prediction_mask, token_loss, loss_sums, token_counts
    return result_loss, result_counts


def _exposed_selected_routes(
    output: Any, spec: MoeLayerSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = _flatten_tensors(output)
    integer = [
        value
        for value in tensors
        if value.dtype
        in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        and value.ndim >= 2
        and value.shape[-1] == spec.top_k
    ]
    for indices in integer:
        weights = next(
            (
                value
                for value in tensors
                if value.is_floating_point()
                and tuple(value.shape) == tuple(indices.shape)
                and value.shape[-1] == spec.top_k
            ),
            None,
        )
        if weights is not None:
            return indices, weights
    raise RuntimeError(
        f"Router {spec.router_name} does not expose selected indices and weights; "
        "reversible selected-route masking is unavailable for this implementation"
    )


def _replace_tensor(value: Any, target: torch.Tensor, replacement: torch.Tensor) -> tuple[Any, int]:
    if value is target:
        return replacement, 1
    if isinstance(value, tuple):
        replaced = [_replace_tensor(item, target, replacement) for item in value]
        items = [item for item, _ in replaced]
        count = sum(item_count for _, item_count in replaced)
        if hasattr(value, "_fields"):
            return type(value)(*items), count
        return tuple(items), count
    if isinstance(value, list):
        replaced = [_replace_tensor(item, target, replacement) for item in value]
        return [item for item, _ in replaced], sum(count for _, count in replaced)
    if isinstance(value, dict):
        replaced = {
            key: _replace_tensor(item, target, replacement)
            for key, item in value.items()
        }
        items = {key: item for key, (item, _) in replaced.items()}
        count = sum(item_count for _, item_count in replaced.values())
        try:
            return type(value)(items), count
        except TypeError:
            return items, count
    return value, 0


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        result: list[torch.Tensor] = []
        for item in value.values():
            result.extend(_flatten_tensors(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    return []


def _bootstrap_mean(
    values: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    if replicates < 1:
        return np.asarray([], dtype=np.float64)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[indices].mean(axis=1)


def _bootstrap_independent_contrast(
    high: np.ndarray,
    low: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if replicates < 1:
        return np.asarray([], dtype=np.float64)
    high_indices = rng.integers(0, len(high), size=(replicates, len(high)))
    low_indices = rng.integers(0, len(low), size=(replicates, len(low)))
    return high[high_indices].mean(axis=1) - low[low_indices].mean(axis=1)


def _model_hook_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
        total += len(getattr(module, "_backward_hooks", {}))
    return total


@contextmanager
def _null_context() -> Iterator[None]:
    yield
