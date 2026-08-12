from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .datasets import DomainExamples, validation_prompts
from .hooks import ExpertInstrumentation
from .io_utils import atomic_write_json
from .metrics import DomainStatistics
from .modeling import ModelBundle, MoeLayerSpec


@dataclass
class CollectionResult:
    statistics: DomainStatistics
    metadata: dict[str, Any]
    diagnostics: list[dict[str, Any]]


def collection_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_parameters_for_gradient_attribution(model: torch.nn.Module) -> None:
    """Avoid allocating multi-billion-parameter gradients; input embeddings seed autograd."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def run_smoke_validation(
    bundle: ModelBundle,
    layer_specs: list[MoeLayerSpec],
    max_length: int,
    compute_gradient_attribution: bool = False,
) -> dict[str, Any]:
    prompts = validation_prompts()
    stats = DomainStatistics.zeros(
        num_examples=len(prompts),
        num_layers=len(layer_specs),
        num_experts=layer_specs[0].num_experts,
        layer_names=[spec.block_name for spec in layer_specs],
        compute_gradient=compute_gradient_attribution,
    )
    before_hooks = _model_hook_count(bundle.model)
    instrumentation = ExpertInstrumentation(
        layer_specs,
        stats,
        compute_gradient_attribution=compute_gradient_attribution,
    )
    with instrumentation:
        encoded = tokenize_texts(bundle, prompts, max_length)
        with instrumentation.batch(list(range(len(prompts))), encoded["attention_mask"]):
            _forward_batch(bundle, encoded, instrumentation, compute_gradient_attribution)
    after_hooks = _model_hook_count(bundle.model)
    if instrumentation.registered_hook_count != 0 or after_hooks != before_hooks:
        raise RuntimeError(
            f"Hook leak detected: before={before_hooks}, after={after_hooks}, "
            f"registered={instrumentation.registered_hook_count}"
        )
    stats.validate()
    if np.any(stats.routing_counts.sum(axis=(0, 2)) == 0):
        raise RuntimeError("At least one MoE layer recorded no routing assignments")
    if np.any(stats.gate_sums.sum(axis=(0, 2)) <= 0):
        raise RuntimeError("At least one MoE layer recorded no positive gate mass")
    if np.any(stats.contribution_sums.sum(axis=(0, 2)) <= 0):
        raise RuntimeError("At least one MoE layer recorded no positive functional contributions")

    aggregate = stats.aggregate()
    first_spec = layer_specs[0]
    first_values = aggregate["normalized_contribution"][0]
    ranking = np.argsort(-first_values, kind="stable")
    diagnostic_ranking = [
        {
            "expert_id": int(expert_id),
            "normalized_contribution": float(first_values[expert_id]),
            "routing_frequency": float(
                aggregate["routing_frequency"][0, expert_id]
            ),
            "gate_mass": float(aggregate["gate_mass"][0, expert_id]),
        }
        for expert_id in ranking[: min(8, len(ranking))]
    ]
    return {
        "passed": True,
        "num_examples": len(prompts),
        "token_counts": stats.token_counts.tolist(),
        "layer_diagnostics": instrumentation.diagnostic_report(),
        "diagnostic_layer": first_spec.model_layer_index,
        "diagnostic_layer_name": first_spec.block_name,
        "top_experts_by_contribution": diagnostic_ranking,
        "hooks_before": before_hooks,
        "hooks_after": after_hooks,
    }


def collect_domain(
    bundle: ModelBundle,
    layer_specs: list[MoeLayerSpec],
    examples: DomainExamples,
    max_length: int,
    batch_size: int,
    compute_gradient_attribution: bool = False,
) -> CollectionResult:
    if not examples.texts:
        raise ValueError(f"No examples supplied for domain {examples.domain}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stats = DomainStatistics.zeros(
        num_examples=len(examples.texts),
        num_layers=len(layer_specs),
        num_experts=layer_specs[0].num_experts,
        layer_names=[spec.block_name for spec in layer_specs],
        compute_gradient=compute_gradient_attribution,
    )
    started = time.monotonic()
    instrumentation = ExpertInstrumentation(
        layer_specs,
        stats,
        compute_gradient_attribution=compute_gradient_attribution,
    )
    with instrumentation:
        for start in range(0, len(examples.texts), batch_size):
            stop = min(start + batch_size, len(examples.texts))
            encoded = tokenize_texts(bundle, examples.texts[start:stop], max_length)
            with instrumentation.batch(list(range(start, stop)), encoded["attention_mask"]):
                _forward_batch(
                    bundle,
                    encoded,
                    instrumentation,
                    compute_gradient_attribution,
                )
            if stop == len(examples.texts) or stop % max(batch_size, 10) == 0:
                elapsed = time.monotonic() - started
                print(
                    f"[{examples.domain}] {stop}/{len(examples.texts)} examples "
                    f"({stats.token_counts[:stop].sum()} tokens, {elapsed:.1f}s)",
                    flush=True,
                )
    stats.validate()
    metadata = dict(examples.metadata)
    metadata.update(
        {
            "domain": examples.domain,
            "num_examples": len(examples.texts),
            "total_tokens": int(stats.token_counts.sum()),
            "min_tokens_per_example": int(stats.token_counts.min()),
            "max_tokens_per_example": int(stats.token_counts.max()),
            "mean_tokens_per_example": float(stats.token_counts.mean()),
            "max_sequence_length": max_length,
            "batch_size": batch_size,
            "elapsed_seconds": time.monotonic() - started,
            "compute_gradient_attribution": compute_gradient_attribution,
        }
    )
    return CollectionResult(stats, metadata, instrumentation.diagnostic_report())


def save_domain_result(
    output_dir: Path,
    domain: str,
    result: CollectionResult,
    fingerprint: str,
) -> None:
    domain_dir = output_dir / "domains"
    result.statistics.save(domain_dir / f"{domain}.npz")
    metadata = dict(result.metadata)
    metadata["collection_fingerprint"] = fingerprint
    metadata["instrumentation_diagnostics"] = result.diagnostics
    atomic_write_json(domain_dir / f"{domain}.metadata.json", metadata)


def load_resumable_domain(
    output_dir: Path,
    domain: str,
    expected_fingerprint: str,
    layer_specs: list[MoeLayerSpec],
) -> CollectionResult | None:
    data_path = output_dir / "domains" / f"{domain}.npz"
    metadata_path = output_dir / "domains" / f"{domain}.metadata.json"
    if not data_path.exists() or not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("collection_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"Existing domain artifact {domain!r} was created with a different collection "
            "configuration. Use --overwrite or a different output directory."
        )
    stats = DomainStatistics.load(data_path)
    expected_names = [spec.block_name for spec in layer_specs]
    if stats.layer_names != expected_names or stats.num_experts != layer_specs[0].num_experts:
        raise RuntimeError(f"Existing domain artifact {domain!r} does not match this model")
    diagnostics = metadata.get("instrumentation_diagnostics", [])
    return CollectionResult(stats, metadata, diagnostics)


def tokenize_texts(
    bundle: ModelBundle, texts: Sequence[str], max_length: int
) -> dict[str, torch.Tensor]:
    encoded = bundle.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="pt",
    )
    if "input_ids" not in encoded:
        raise RuntimeError("Tokenizer returned no input_ids")
    input_ids = encoded["input_ids"].to(bundle.runtime.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(bundle.runtime.device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _forward_batch(
    bundle: ModelBundle,
    encoded: dict[str, torch.Tensor],
    instrumentation: ExpertInstrumentation,
    compute_gradient_attribution: bool,
) -> None:
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if compute_gradient_attribution:
        embeddings = bundle.model.get_input_embeddings()(input_ids).detach()
        embeddings.requires_grad_(True)
        outputs = bundle.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = _first_tensor(outputs)
        output_embeddings = bundle.model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Causal LM exposes no output embedding / LM head")
        logits = output_embeddings(hidden)
        if logits.shape[1] < 2:
            raise RuntimeError("At least two tokens are required for next-token attribution")
        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss.backward()
        instrumentation.finalize_gradients()
        del logits, hidden, outputs, embeddings, loss
    else:
        with torch.inference_mode():
            outputs = bundle.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            # Force realization before leaving inference mode on lazy backends.
            hidden = _first_tensor(outputs)
            if not torch.isfinite(hidden).all():
                raise RuntimeError("Model produced non-finite hidden states")
            del hidden, outputs


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    first = getattr(output, "last_hidden_state", None)
    if isinstance(first, torch.Tensor):
        return first
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError("Could not extract last hidden state from backbone output")


def _model_hook_count(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
        total += len(getattr(module, "_backward_hooks", {}))
    return total
