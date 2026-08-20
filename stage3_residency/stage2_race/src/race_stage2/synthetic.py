"""Controlled synthetic streams and traces for the Stage 2 algorithm tests.

These are algorithm tests, not scientific evaluation evidence.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from residency_headroom import TRACE_SCHEMA_VERSION
from residency_headroom.trace import RoutingTrace

from . import NOT_REUSED_WITHIN_HORIZON


DOMAINS = ("general", "coding", "math", "reasoning")


def build_trace(
    request_for: Callable[[int, int, int], Sequence[int]],
    *,
    prompts_per_domain: int = 3,
    generation_length: int = 60,
    num_layers: int = 2,
    num_experts: int = 24,
    top_k: int = 2,
) -> RoutingTrace:
    """A synthetic routing trace whose atomic requests come from ``request_for``.

    ``request_for(sequence_id, layer, within_sequence_position)`` must return
    ``top_k`` distinct expert identifiers. The position is the within-sequence
    same-layer index, so a periodic pattern stays periodic when the frozen workload
    builder concatenates sequences in any order.
    """

    columns: dict[str, list] = {
        name: []
        for name in (
            "sequence_id",
            "domain_id",
            "prompt_index",
            "generated_token_index",
            "layer_index",
            "requested_expert_ids",
            "router_weights",
            "token_id",
            "prompt_length",
            "generation_length",
        )
    }
    manifest = []
    sequence_id = 0
    for domain_id, domain in enumerate(DOMAINS):
        for prompt_index in range(prompts_per_domain):
            token_ids = []
            for generated_index in range(generation_length):
                token_id = 1000 + sequence_id * 100 + generated_index
                token_ids.append(token_id)
                for layer in range(num_layers):
                    request = np.asarray(
                        request_for(sequence_id, layer, generated_index), dtype=np.int16
                    )
                    if request.size != top_k or np.unique(request).size != top_k:
                        raise ValueError("Synthetic request must contain distinct experts")
                    if request.min() < 0 or request.max() >= num_experts:
                        raise ValueError("Synthetic request left the expert range")
                    columns["sequence_id"].append(sequence_id)
                    columns["domain_id"].append(domain_id)
                    columns["prompt_index"].append(prompt_index)
                    columns["generated_token_index"].append(generated_index)
                    columns["layer_index"].append(layer)
                    columns["requested_expert_ids"].append(request)
                    columns["router_weights"].append(
                        np.linspace(0.6, 0.4, top_k, dtype=np.float32)
                    )
                    columns["token_id"].append(token_id)
                    columns["prompt_length"].append(12 + prompt_index)
                    columns["generation_length"].append(generation_length)
            manifest.append(
                {
                    "sequence_id": sequence_id,
                    "domain": domain,
                    "domain_id": domain_id,
                    "prompt_index": prompt_index,
                    "prompt_id": f"{domain}-{prompt_index}",
                    "prompt_sha256": f"hash-{domain}-{prompt_index}",
                    "prompt_length": 12 + prompt_index,
                    "generation_length": generation_length,
                    "generated_token_ids": token_ids,
                    "stopped_on_eos": False,
                }
            )
            sequence_id += 1
    count = len(columns["sequence_id"])
    arrays = {
        "event_index": np.arange(count, dtype=np.int64),
        "sequence_id": np.asarray(columns["sequence_id"], dtype=np.int32),
        "domain_id": np.asarray(columns["domain_id"], dtype=np.int8),
        "prompt_index": np.asarray(columns["prompt_index"], dtype=np.int32),
        "generated_token_index": np.asarray(columns["generated_token_index"], dtype=np.int32),
        "layer_index": np.asarray(columns["layer_index"], dtype=np.int16),
        "requested_expert_ids": np.stack(columns["requested_expert_ids"]),
        "router_weights": np.stack(columns["router_weights"]),
        "token_id": np.asarray(columns["token_id"], dtype=np.int32),
        "prompt_length": np.asarray(columns["prompt_length"], dtype=np.int32),
        "generation_length": np.asarray(columns["generation_length"], dtype=np.int32),
    }
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "hash_basis": {"kind": "race-stage2-synthetic"},
        "domains": list(DOMAINS),
        "layer_indices": list(range(num_layers)),
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_bytes_by_layer": np.full((num_layers, num_experts), 1024).tolist(),
        "all_experts_equal_size": True,
        "sequences": manifest,
    }
    trace = RoutingTrace.from_mapping(arrays, metadata, validate=True)
    metadata["trace_hash"] = trace.logical_hash()
    return RoutingTrace.from_mapping(arrays, metadata, validate=True)


def ring_request(period: int, top_k: int = 2, spread: int = 0, seed: int = 7):
    """Cyclic reuse with a fixed period; a longer period means longer reuse distance."""

    rng = np.random.default_rng(seed)
    jitter = rng.integers(0, spread + 1, size=(period, 1)) if spread else None

    def request_for(sequence_id: int, layer: int, position: int) -> Sequence[int]:
        del sequence_id
        slot = position % period
        offset = int(jitter[slot, 0]) if jitter is not None else 0
        base = ((slot + offset) % period) * top_k + layer * period * top_k
        return [base + index for index in range(top_k)]

    return request_for


def horizon_examples(
    *,
    rounds: int,
    advisers: int,
    good: Sequence[int],
    candidates: int = 8,
    noise: float = 1.0,
    seed: int = 11,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Synthetic learning examples where only ``good`` advisers rank correctly.

    Returns normalized adviser score blocks and capped next-use distances.
    """

    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _round in range(int(rounds)):
        distances = rng.choice(
            np.arange(1, NOT_REUSED_WITHIN_HORIZON + 1), size=candidates, replace=False
        )
        perfect = np.argsort(np.argsort(-distances)).astype(np.float64) / max(candidates - 1, 1)
        block = rng.random((advisers, candidates))
        for index in good:
            block[index] = np.clip(
                perfect + noise * 0.0, 0.0, 1.0
            )
        blocks.append(block)
        targets.append(distances.astype(np.int64))
    return blocks, targets


def regime_switch_examples(
    *,
    rounds_per_regime: int,
    advisers: int,
    first_good: int,
    second_good: int,
    candidates: int = 8,
    seed: int = 13,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """First half is solved by ``first_good``, second half by ``second_good``."""

    left = horizon_examples(
        rounds=rounds_per_regime,
        advisers=advisers,
        good=[first_good],
        candidates=candidates,
        seed=seed,
    )
    right = horizon_examples(
        rounds=rounds_per_regime,
        advisers=advisers,
        good=[second_good],
        candidates=candidates,
        seed=seed + 1,
    )
    return left[0] + right[0], left[1] + right[1]


def uninformative_examples(
    *, rounds: int, advisers: int, candidates: int = 8, seed: int = 17
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """No adviser carries usable signal, so every adviser loss is near one half."""

    rng = np.random.default_rng(seed)
    blocks = []
    targets = []
    for _round in range(int(rounds)):
        distances = rng.choice(
            np.arange(1, NOT_REUSED_WITHIN_HORIZON + 1), size=candidates, replace=False
        )
        blocks.append(rng.random((advisers, candidates)))
        targets.append(distances.astype(np.int64))
    return blocks, targets
