from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from residency_headroom import TRACE_SCHEMA_VERSION
from residency_headroom.trace import RoutingTrace


DOMAINS = ("general", "coding", "math", "reasoning")


def make_trace(
    *,
    prompts_per_domain: int = 3,
    generation_length: int = 3,
    num_layers: int = 2,
    num_experts: int = 8,
    top_k: int = 2,
) -> RoutingTrace:
    events: dict[str, list] = {
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
    sequences = []
    sequence_id = 0
    for domain_id, domain in enumerate(DOMAINS):
        for prompt_index in range(prompts_per_domain):
            token_ids = []
            for generated_index in range(generation_length):
                token_id = 1000 + sequence_id * 10 + generated_index
                token_ids.append(token_id)
                for layer in range(num_layers):
                    first = (domain_id * 2 + prompt_index + generated_index + layer) % num_experts
                    request = np.asarray(
                        [(first + offset * 3) % num_experts for offset in range(top_k)],
                        dtype=np.int16,
                    )
                    if len(np.unique(request)) != top_k:
                        request = np.arange(top_k, dtype=np.int16)
                    events["sequence_id"].append(sequence_id)
                    events["domain_id"].append(domain_id)
                    events["prompt_index"].append(prompt_index)
                    events["generated_token_index"].append(generated_index)
                    events["layer_index"].append(layer)
                    events["requested_expert_ids"].append(request)
                    events["router_weights"].append(
                        np.linspace(0.6, 0.4, top_k, dtype=np.float32)
                    )
                    events["token_id"].append(token_id)
                    events["prompt_length"].append(12 + prompt_index)
                    events["generation_length"].append(generation_length)
            sequences.append(
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
    count = len(events["sequence_id"])
    arrays = {
        "event_index": np.arange(count, dtype=np.int64),
        "sequence_id": np.asarray(events["sequence_id"], dtype=np.int32),
        "domain_id": np.asarray(events["domain_id"], dtype=np.int8),
        "prompt_index": np.asarray(events["prompt_index"], dtype=np.int32),
        "generated_token_index": np.asarray(events["generated_token_index"], dtype=np.int32),
        "layer_index": np.asarray(events["layer_index"], dtype=np.int16),
        "requested_expert_ids": np.stack(events["requested_expert_ids"]),
        "router_weights": np.stack(events["router_weights"]),
        "token_id": np.asarray(events["token_id"], dtype=np.int32),
        "prompt_length": np.asarray(events["prompt_length"], dtype=np.int32),
        "generation_length": np.asarray(events["generation_length"], dtype=np.int32),
    }
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "hash_basis": {"kind": "synthetic-unit-test"},
        "domains": list(DOMAINS),
        "layer_indices": list(range(num_layers)),
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_bytes_by_layer": np.full((num_layers, num_experts), 1024).tolist(),
        "all_experts_equal_size": True,
        "sequences": sequences,
    }
    trace = RoutingTrace.from_mapping(arrays, metadata, validate=True)
    metadata["trace_hash"] = trace.logical_hash()
    return RoutingTrace.from_mapping(arrays, metadata, validate=True)


def workload_config(
    *, prompts_per_domain: int = 3, num_layers: int = 2, num_experts: int = 8, top_k: int = 2
) -> dict:
    return {
        "schema_version": "race_stage0_config_v1",
        "stage": "synthetic test",
        "run_kind": "pilot",
        "source_commit": "test",
        "model": "toy",
        "model_revision": "toy-revision",
        "seed": 42,
        "domains": list(DOMAINS),
        "num_prompts_per_domain": prompts_per_domain,
        "expected_architecture": {
            "num_moe_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        "calibration_fraction": 1 / prompts_per_domain,
        "cache_capacities": [2, 3, 4, 6, 8],
        "random_policy_seeds": [1001, 1002, 1003, 1004, 1005],
        "lfu_decay_alphas": [0.9, 0.95, 0.99],
        "abrupt_pairs": [
            ["general", "coding"],
            ["coding", "general"],
            ["general", "math"],
            ["math", "reasoning"],
        ],
        "repeated_domain_order": ["general", "coding", "math", "reasoning", "general"],
        "repeated_segment_prompts": 1,
        "mixed_workload_seed": 99,
        "lambda_values": [0.0, 0.25, 0.5, 1.0],
        "cost_models": ["unit_miss", "expert_bytes"],
        "primary_cost_model": "unit_miss",
        "primary_lambda": 0.0,
        "bootstrap_replicates": 20,
        "bootstrap_seed": 123,
        "bootstrap_unit": "source_decode_sequence",
        "workload_bootstrap_strata": ["segment_index", "domain"],
        "regime_bootstrap_cluster": "source_sequence_id_stratified_by_domain",
        "go_no_go": {
            "strong_point_headroom": 0.15,
            "strong_ci_lower": 0.05,
            "strong_required_budgets": 3,
            "total_budgets": 5,
            "weak_point_headroom": 0.05,
            "weak_required_budgets": 3,
            "weak_ci_lower": 0.0,
        },
        "decision_enabled": False,
    }
