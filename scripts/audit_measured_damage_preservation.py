#!/usr/bin/env python3
"""Independent Stage 3 audit.

This auditor deliberately imports no production analysis code from
``expert_analysis``. Every hash, damage delta, additive prediction, MILP
constraint, budget, split, additivity gate, metric, bootstrap interval,
development gate, and decision is recomputed from raw saved artifacts using
only the standard library and numpy. Audit failure blocks progression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DOMAINS = ("general", "math", "coding", "reasoning")
DOMAIN_SEED_OFFSETS = {"general": 11, "math": 23, "coding": 37, "reasoning": 53}
NUM_LAYERS = 16
NUM_EXPERTS = 64
CALIBRATION_PER_DOMAIN = 25
PROFILE_BITS = (3, 4, 8)
PROTECTED_BITS = 8
BASE_BITS = {"4to8": 4, "3to8": 3}
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RANDOM_SEEDS = (1001, 1002, 1003, 1004, 1005)
DEVELOPMENT_BUDGET = 0.20
DEVELOPMENT_SEED = 46
DEVELOPMENT_EXAMPLES = 50
SEQUENCE_LENGTH = 68
MEASURED_PER_EXAMPLE = 64
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260815
GATE_E_RELATIVE_TOLERANCE = 0.10
GATE_E_EPSILON = 1e-12
ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN = 0.80
ADDITIVITY_MIN_WORST_DELTA_SPEARMAN = 0.80
EXPECTED_STAGE2B_REGISTRY_SHA = (
    "b0221262f0e51700cc16fa5e6a681f63ab6507a9d768714f853f3dfc3f87aa34"
)
EXPECTED_STAGE2C_REGISTRY_SHA = (
    "b1b6b9a68c0840e60b1d080678ed7a8fb7f56a0595100a76223c4f3860b52caf"
)
STAGE2B_METHODS = (
    "robust_functional", "robust_routing", "average_specialization",
    "global_importance", "general_only", "math_only", "coding_only",
    "reasoning_only",
)
PRIMARY = "measured_damage_robust"


class Audit:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[dict[str, Any]] = []
        self.sections: dict[str, dict[str, int]] = {}
        self.max_numeric_difference = 0.0

    def check(self, section: str, name: str, condition: bool, detail: Any = None) -> None:
        entry = self.sections.setdefault(section, {"passed": 0, "failed": 0})
        if condition:
            self.passed += 1
            entry["passed"] += 1
        else:
            self.failed += 1
            entry["failed"] += 1
            self.failures.append({"section": section, "check": name, "detail": detail})

    def close(self, value: float, expected: float, tolerance: float = 1e-9) -> bool:
        if value is None or expected is None:
            return value == expected
        value = float(value)
        expected = float(expected)
        if not (math.isfinite(value) and math.isfinite(expected)):
            return value == expected or (math.isnan(value) and math.isnan(expected))
        difference = abs(value - expected)
        self.max_numeric_difference = max(self.max_numeric_difference, difference)
        return difference <= tolerance


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = rankdata(np.asarray(first, dtype=np.float64))
    second_ranks = rankdata(np.asarray(second, dtype=np.float64))
    fr = first_ranks - first_ranks.mean()
    sr = second_ranks - second_ranks.mean()
    denominator = math.sqrt(float((fr * fr).sum()) * float((sr * sr).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((fr * sr).sum() / denominator)


def expert_bytes(shapes: list[list[int]], bits: int, group_size: int = 128) -> int:
    total = 0
    for shape in shapes:
        weights = 1
        for value in shape:
            weights *= int(value)
        rows = weights // int(shape[-1])
        if bits == 16:
            total += weights * 2
            continue
        packed = math.ceil(weights * bits / 8)
        groups = rows * math.ceil(int(shape[-1]) / group_size)
        total += packed + groups * 2
    return total


def load_losses(losses_dir: Path, slug: str) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for domain in DOMAINS:
        with np.load(losses_dir / slug / f"{domain}.npz", allow_pickle=False) as data:
            output[domain] = np.asarray(data["loss_sums"], dtype=np.float64) / np.asarray(
                data["token_counts"], dtype=np.float64
            )
    return output


def allocation_slug(record: dict[str, Any]) -> str:
    if record["method_kind"] == "uniform_reference":
        return str(record["method"])
    return (
        f"{record['method']}_{record['regime']}_budget"
        f"{int(round(record['budget_fraction'] * 100))}"
    )


def replicate_indices(count: int, replicates: int, seed: int, domain: str) -> np.ndarray:
    rng = np.random.default_rng([seed, DOMAIN_SEED_OFFSETS[domain]])
    return rng.integers(0, count, size=(replicates, count))


def audit_prior_state(audit: Audit, results_root: Path) -> None:
    section = "prior_state"
    surrogate = read_json(
        results_root / "quantization_cost_surrogate" / "surrogate_decision.json"
    )
    audit.check(
        section, "stage2a_no_go", surrogate.get("decision") == "SURROGATE_NO_GO"
    )
    audit.check(
        section,
        "stage2a_matrix_unauthorized",
        surrogate.get("full_cost_matrix_authorized") is False,
    )
    stage2b = read_json(
        results_root / "robust_specialist_preservation" / "stage2b_decision.json"
    )
    audit.check(
        section, "stage2b_no_go", stage2b.get("decision") == "ROBUST_PRESERVATION_NO_GO"
    )
    audit.check(
        section,
        "stage2b_registry_sha",
        stage2b.get("registry_sha256") == EXPECTED_STAGE2B_REGISTRY_SHA,
    )
    stage2c = read_json(
        results_root / "fragility_robust_preservation" / "stage2c_decision.json"
    )
    audit.check(
        section, "stage2c_no_go", stage2c.get("decision") == "FRAGILITY_ROBUST_NO_GO"
    )
    audit.check(
        section,
        "stage2c_registry_sha",
        stage2c.get("registry_sha256") == EXPECTED_STAGE2C_REGISTRY_SHA,
    )
    audit.check(
        section,
        "stage2c_no_authorized_regimes",
        stage2c.get("development_decision", {}).get("authorized_regimes") == [],
    )


def audit_damage(audit: Audit, results_dir: Path) -> dict[str, Any] | None:
    section = "damage_matrix"
    damage_dir = results_dir / "damage"
    record_path = damage_dir / "damage_matrix.json"
    if not record_path.is_file():
        return None
    record = read_json(record_path)
    deterministic = {
        key: value
        for key, value in record.items()
        if key not in ("damage_sha256", "created_at_utc")
    }
    audit.check(
        section,
        "record_sha",
        canonical_sha256(deterministic) == record.get("damage_sha256"),
    )
    with np.load(damage_dir / "damage_matrix.npz", allow_pickle=False) as data:
        mean_nll = np.asarray(data["mean_nll"], dtype=np.float64)
        delta_nll = np.asarray(data["delta_nll"], dtype=np.float64)
        bf16_nll = np.asarray(data["bf16_nll"], dtype=np.float64)
    audit.check(
        section,
        "mean_array_sha",
        record["array_sha256"]["mean_nll"] == array_sha256(mean_nll),
    )
    audit.check(
        section,
        "delta_array_sha",
        record["array_sha256"]["delta_nll"] == array_sha256(delta_nll),
    )
    audit.check(
        section,
        "bf16_array_sha",
        record["array_sha256"]["bf16_nll"] == array_sha256(bf16_nll),
    )
    audit.check(
        section,
        "shape",
        mean_nll.shape == (NUM_LAYERS, NUM_EXPERTS, len(DOMAINS), len(PROFILE_BITS)),
    )
    # BF16 baseline recomputed from the saved reference losses.
    bf16 = load_losses(damage_dir / "losses", "bf16")
    for index, domain in enumerate(DOMAINS):
        audit.check(
            section,
            f"bf16_{domain}",
            audit.close(float(bf16[domain].mean()), float(bf16_nll[index])),
        )
        audit.check(
            section,
            f"bf16_record_{domain}",
            audit.close(record["bf16_nll"][domain], float(bf16_nll[index])),
        )
    for state in ("uniform8", "uniform4", "uniform3"):
        values = load_losses(damage_dir / "losses", state)
        for domain in DOMAINS:
            audit.check(
                section,
                f"{state}_{domain}",
                audit.close(
                    float(values[domain].mean()), record["uniform_nll"][state][domain]
                ),
            )
    # Reassemble mean/delta from raw chunks.
    rebuilt = np.zeros_like(mean_nll)
    for bit_index, bits in enumerate(PROFILE_BITS):
        for layer in range(NUM_LAYERS):
            path = damage_dir / "chunks" / f"bits{bits}" / f"layer_{layer:02d}.npz"
            relative = f"chunks/bits{bits}/layer_{layer:02d}.npz"
            audit.check(
                section,
                f"chunk_hash_{relative}",
                record["chunk_sha256"].get(relative) == file_sha256(path),
            )
            with np.load(path, allow_pickle=False) as data:
                loss_sums = np.asarray(data["loss_sums"], dtype=np.float64)
                token_counts = np.asarray(data["token_counts"], dtype=np.float64)
            audit.check(
                section,
                f"chunk_shape_{relative}",
                loss_sums.shape
                == (NUM_EXPERTS, len(DOMAINS), CALIBRATION_PER_DOMAIN),
            )
            rebuilt[layer, :, :, bit_index] = (loss_sums / token_counts).mean(axis=2)
    audit.check(
        section, "mean_rebuilt_exact", bool(np.array_equal(rebuilt, mean_nll))
    )
    audit.check(
        section,
        "delta_definition",
        bool(np.array_equal(delta_nll, mean_nll - bf16_nll[None, None, :, None])),
    )
    return {"record": record, "delta_nll": delta_nll, "bf16_nll": bf16_nll}


def predicted_delta(bits_matrix: np.ndarray, delta_nll: np.ndarray) -> np.ndarray:
    predicted = np.zeros(len(DOMAINS), dtype=np.float64)
    for bit_index, width in enumerate(PROFILE_BITS):
        mask = bits_matrix == width
        if mask.any():
            predicted += delta_nll[:, :, :, bit_index][mask].sum(axis=0)
    return predicted


def audit_registry(
    audit: Audit,
    results_dir: Path,
    stage2b_dir: Path,
    stage2c_dir: Path,
    damage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    section = "allocations"
    registry_path = results_dir / "allocations" / "allocation_registry.json"
    if not registry_path.is_file():
        return None
    registry = read_json(registry_path)
    expected = canonical_sha256(
        {
            key: value
            for key, value in registry.items()
            if key not in ("registry_sha256", "created_at_utc")
        }
    )
    audit.check(
        section, "registry_sha", expected == registry.get("registry_sha256")
    )
    audit.check(
        section,
        "stage2b_registry_sha",
        registry.get("stage2b_registry_sha256") == EXPECTED_STAGE2B_REGISTRY_SHA,
    )
    audit.check(
        section,
        "stage2c_registry_sha",
        registry.get("stage2c_registry_sha256") == EXPECTED_STAGE2C_REGISTRY_SHA,
    )
    if damage is not None:
        audit.check(
            section,
            "registry_damage_sha",
            registry.get("damage_sha256") == damage["record"]["damage_sha256"],
        )

    with np.load(
        stage2b_dir / "calibration" / "memory_matrix.npz", allow_pickle=False
    ) as data:
        shapes = [list(int(v) for v in shape) for shape in data["tensor_shapes"].tolist()]
    per_expert = {bits: expert_bytes(shapes, bits) for bits in (3, 4, 8, 16)}
    total_experts = NUM_LAYERS * NUM_EXPERTS

    dirs = {
        "stage3_new": results_dir / "allocations",
        "stage2b_frozen": stage2b_dir / "allocations",
        "stage2c_frozen": stage2c_dir / "allocations",
    }
    groups = (
        ("new_entries", "stage3_new"),
        ("reused_stage2b_entries", "stage2b_frozen"),
        ("reused_stage2c_entries", "stage2c_frozen"),
    )
    records_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}
    for group_key, source in groups:
        for entry in registry.get(group_key, []):
            path = dirs[source] / entry["file"]
            audit.check(
                section,
                f"file_hash_{entry['file']}",
                path.is_file() and file_sha256(path) == entry["file_sha256"],
            )
            record = read_json(path)
            deterministic = {
                key: value
                for key, value in record.items()
                if key not in ("allocation_sha256", "created_at_utc")
            }
            audit.check(
                section,
                f"record_sha_{entry['file']}",
                canonical_sha256(deterministic) == record.get("allocation_sha256")
                == entry["allocation_sha256"],
            )
            if record["method_kind"] != "uniform_reference":
                records_by_key[
                    (record["method"], record["regime"], record["budget_fraction"])
                ] = record

    for entry in registry.get("new_entries", []):
        record = read_json(dirs["stage3_new"] / entry["file"])
        regime = record["regime"]
        base = BASE_BITS[regime]
        bits = np.asarray(record["expert_bits"], dtype=np.int64)
        protected = bits == PROTECTED_BITS
        audit.check(
            section,
            f"bits_binary_{entry['file']}",
            bool(np.all((bits == base) | (bits == PROTECTED_BITS))),
        )
        audit.check(
            section,
            f"protected_count_{entry['file']}",
            int(protected.sum()) == record["protected_expert_count"],
        )
        increment = per_expert[8] - per_expert[base]
        total_increment = increment * total_experts
        budget = math.floor(record["budget_fraction"] * total_increment)
        audit.check(
            section,
            f"budget_bytes_{entry['file']}",
            budget == record["budget_bytes"],
        )
        used = int(protected.sum()) * increment
        audit.check(
            section,
            f"budget_respected_{entry['file']}",
            used == record["used_protection_bytes"] and used <= budget,
        )
        if damage is not None:
            predicted = predicted_delta(bits, damage["delta_nll"])
            for index, domain in enumerate(DOMAINS):
                audit.check(
                    section,
                    f"predicted_{entry['file']}_{domain}",
                    audit.close(
                        float(predicted[index]),
                        record["predicted_domain_delta_nll"][domain],
                    ),
                )
            audit.check(
                section,
                f"predicted_worst_{entry['file']}",
                audit.close(
                    float(predicted.max()), record["predicted_worst_delta_nll"]
                ),
            )
            # Optimality: no frozen comparator beats the MILP optimum.
            optimum = float(predicted.max())
            for (method, regime_key, fraction), comparator in records_by_key.items():
                if regime_key != regime or fraction != record["budget_fraction"]:
                    continue
                if method == PRIMARY:
                    continue
                comparator_max = float(
                    predicted_delta(
                        np.asarray(comparator["expert_bits"], dtype=np.int64),
                        damage["delta_nll"],
                    ).max()
                )
                audit.check(
                    section,
                    f"optimality_{entry['file']}_{method}",
                    comparator_max >= optimum - 1e-6,
                    {"comparator": method, "comparator_max": comparator_max,
                     "optimum": optimum},
                )
    return {"registry": registry, "records_by_key": records_by_key, "dirs": dirs}


def audit_preregistration(
    audit: Audit,
    results_dir: Path,
    registry_info: dict[str, Any] | None,
    damage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    section = "preregistration"
    path = results_dir / "stage3_preregistration.json"
    sha_path = results_dir / "stage3_preregistration_sha256.txt"
    if not path.is_file():
        return None
    audit.check(
        section,
        "sha_file_matches",
        sha_path.is_file() and sha_path.read_text().strip() == file_sha256(path),
    )
    prereg = read_json(path)
    if registry_info is not None:
        audit.check(
            section,
            "registry_sha_matches",
            prereg.get("allocation_registry_sha256")
            == registry_info["registry"]["registry_sha256"],
        )
    if damage is not None:
        audit.check(
            section,
            "damage_sha_matches",
            prereg.get("damage_sha256") == damage["record"]["damage_sha256"],
        )
    audit.check(
        section, "development_seed", prereg.get("development_seed") == DEVELOPMENT_SEED
    )
    audit.check(section, "final_seed", prereg.get("final_seed") == 44)
    audit.check(
        section,
        "development_budget",
        prereg.get("development_budget_fraction") == DEVELOPMENT_BUDGET,
    )
    return prereg


def audit_split(audit: Audit, results_dir: Path, prereg: dict[str, Any] | None) -> None:
    section = "seed46_split"
    manifest_path = results_dir / "splits" / "split_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = read_json(manifest_path)
    audit.check(
        section, "seed", manifest.get("development_seed") == DEVELOPMENT_SEED
    )
    for domain in DOMAINS:
        entry = manifest["domains"][domain]
        path = results_dir / "splits" / "development" / f"{domain}.npz"
        with np.load(path, allow_pickle=False) as data:
            input_ids = np.asarray(data["input_ids"])
            mask = np.asarray(data["measurement_mask"])
        audit.check(
            section,
            f"input_hash_{domain}",
            array_sha256(input_ids) == entry["input_ids_sha256"],
        )
        audit.check(
            section,
            f"mask_hash_{domain}",
            array_sha256(mask) == entry["measurement_mask_sha256"],
        )
        audit.check(
            section,
            f"geometry_{domain}",
            input_ids.shape == (DEVELOPMENT_EXAMPLES, SEQUENCE_LENGTH)
            and int(mask.sum()) == DEVELOPMENT_EXAMPLES * MEASURED_PER_EXAMPLE,
        )
        audit.check(
            section,
            f"disjointness_flag_{domain}",
            entry.get("disjointness_verified") is True
            and sum(entry["overlap_checks"].values()) == 0,
        )
        if prereg is not None:
            audit.check(
                section,
                f"prereg_hash_{domain}",
                prereg["development_split_input_hashes"][domain]
                == entry["input_ids_sha256"],
            )


def audit_additivity(
    audit: Audit,
    results_dir: Path,
    registry_info: dict[str, Any] | None,
    damage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    section = "additivity"
    report_path = results_dir / "additivity" / "additivity_report.json"
    if not report_path.is_file() or registry_info is None or damage is None:
        return None
    report = read_json(report_path)
    losses_dir = results_dir / "additivity" / "losses"
    bf16 = damage["bf16_nll"]
    rows_by_regime: dict[str, list[dict[str, np.ndarray]]] = {r: [] for r in BASE_BITS}
    audit.check(
        section,
        "probe_count",
        len(report["probe_rows"])
        == len(BASE_BITS) * (len(STAGE2B_METHODS) + len(RANDOM_SEEDS) + 2),
    )
    for row in report["probe_rows"]:
        key = (row["method"], row["regime"], row["budget_fraction"])
        record = registry_info["records_by_key"].get(key)
        audit.check(section, f"probe_known_{row['slug']}", record is not None)
        if record is None:
            continue
        audit.check(
            section,
            f"probe_budget_{row['slug']}",
            row["budget_fraction"] == DEVELOPMENT_BUDGET,
        )
        losses = load_losses(losses_dir, row["slug"])
        measured = np.asarray(
            [float(losses[domain].mean()) - bf16[index] for index, domain in enumerate(DOMAINS)]
        )
        predicted = predicted_delta(
            np.asarray(record["expert_bits"], dtype=np.int64), damage["delta_nll"]
        )
        for index, domain in enumerate(DOMAINS):
            audit.check(
                section,
                f"measured_{row['slug']}_{domain}",
                audit.close(float(measured[index]), row["measured"][index]),
            )
            audit.check(
                section,
                f"predicted_{row['slug']}_{domain}",
                audit.close(float(predicted[index]), row["predicted"][index]),
            )
        rows_by_regime[row["regime"]].append(
            {"predicted": predicted, "measured": measured}
        )
    recomputed_authorized: list[str] = []
    for regime, rows in rows_by_regime.items():
        predicted = np.stack([row["predicted"] for row in rows])
        measured = np.stack([row["measured"] for row in rows])
        gates = report["gates_by_regime"][regime]
        gate1_pass = True
        for index, domain in enumerate(DOMAINS):
            value = spearman(predicted[:, index], measured[:, index])
            audit.check(
                section,
                f"gate1_spearman_{regime}_{domain}",
                audit.close(
                    value, gates["gate_add_1"]["spearman_by_domain"][domain]
                ),
            )
            gate1_pass = gate1_pass and (
                math.isfinite(value) and value >= ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN
            )
        worst = spearman(predicted.max(axis=1), measured.max(axis=1))
        audit.check(
            section,
            f"gate2_spearman_{regime}",
            audit.close(worst, gates["gate_add_2"]["spearman"]),
        )
        gate2_pass = math.isfinite(worst) and worst >= ADDITIVITY_MIN_WORST_DELTA_SPEARMAN
        audit.check(
            section,
            f"gate1_pass_{regime}",
            gate1_pass == gates["gate_add_1"]["passed"],
        )
        audit.check(
            section,
            f"gate2_pass_{regime}",
            gate2_pass == gates["gate_add_2"]["passed"],
        )
        audit.check(
            section,
            f"all_passed_{regime}",
            (gate1_pass and gate2_pass) == gates["all_passed"],
        )
        if gate1_pass and gate2_pass:
            recomputed_authorized.append(regime)
    audit.check(
        section,
        "authorized_regimes",
        sorted(recomputed_authorized)
        == sorted(report["additivity_decision"]["authorized_regimes"]),
    )
    expected_decision = (
        "ADDITIVITY_GO" if recomputed_authorized else "MEASURED_DAMAGE_NO_GO"
    )
    audit.check(
        section,
        "decision",
        report["additivity_decision"]["decision"] == expected_decision,
    )
    return report


def method_metrics(
    nll: dict[str, np.ndarray],
    bf16: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative = np.zeros(len(DOMAINS))
    recovery = np.zeros(len(DOMAINS))
    replicate_relative = np.zeros((BOOTSTRAP_REPLICATES, len(DOMAINS)))
    for index, domain in enumerate(DOMAINS):
        allocation = nll[domain]
        baseline = bf16[domain]
        base_uniform = base[domain]
        relative[index] = (allocation.mean() - baseline.mean()) / baseline.mean()
        recovery[index] = base_uniform.mean() - allocation.mean()
        rep = indices[domain]
        replicate_relative[:, index] = (
            allocation[rep].mean(axis=1) - baseline[rep].mean(axis=1)
        ) / baseline[rep].mean(axis=1)
    return {
        "relative": relative,
        "recovery": recovery,
        "worst": float(relative.max()),
        "mean": float(relative.mean()),
        "replicate_worst": replicate_relative.max(axis=1),
        "replicate_mean": replicate_relative.mean(axis=1),
    }


def audit_development(
    audit: Audit,
    results_dir: Path,
    registry_info: dict[str, Any] | None,
    additivity: dict[str, Any] | None,
) -> bool:
    section = "development"
    phase_dir = results_dir / "development_seed46"
    results_path = phase_dir / "development_results.json"
    if not results_path.is_file() or registry_info is None:
        return False
    results = read_json(results_path)
    losses_dir = phase_dir / "losses"
    bf16 = load_losses(losses_dir, "bf16_reference")
    counts = {domain: len(values) for domain, values in bf16.items()}
    indices = {
        domain: replicate_indices(counts[domain], BOOTSTRAP_REPLICATES,
                                  BOOTSTRAP_SEED, domain)
        for domain in DOMAINS
    }
    base_by_regime = {
        "4to8": load_losses(losses_dir, "uniform_4bit_reference"),
        "3to8": load_losses(losses_dir, "uniform_3bit_reference"),
    }
    authorized = results.get("authorized_regimes", [])
    if additivity is not None:
        audit.check(
            section,
            "authorized_matches_additivity",
            sorted(authorized)
            == sorted(additivity["additivity_decision"]["authorized_regimes"]),
        )
    rows_by_key = {
        (row["method"], row["regime"], row["budget_fraction"]): row
        for row in results["method_rows"]
    }
    metrics_by_regime: dict[str, dict[str, dict[str, Any]]] = {}
    for (method, regime, fraction), row in rows_by_key.items():
        record = registry_info["records_by_key"].get((method, regime, fraction))
        audit.check(section, f"row_known_{method}_{regime}", record is not None)
        if record is None:
            continue
        slug = allocation_slug(record)
        nll = load_losses(losses_dir, slug)
        metrics = method_metrics(nll, bf16, base_by_regime[regime], indices)
        metrics_by_regime.setdefault(regime, {})[method] = metrics
        for index, domain in enumerate(DOMAINS):
            audit.check(
                section,
                f"relative_{slug}_{domain}",
                audit.close(
                    float(metrics["relative"][index]),
                    row[f"relative_delta_{domain}"],
                ),
            )
            audit.check(
                section,
                f"recovery_{slug}_{domain}",
                audit.close(
                    float(metrics["recovery"][index]), row[f"recovery_{domain}"]
                ),
            )
        audit.check(
            section,
            f"worst_{slug}",
            audit.close(metrics["worst"], row["worst_relative_delta"]),
        )
        audit.check(
            section,
            f"mean_{slug}",
            audit.close(metrics["mean"], row["mean_relative_delta"]),
        )
        audit.check(
            section,
            f"worst_ci_{slug}",
            audit.close(
                float(np.quantile(metrics["replicate_worst"], 0.025)),
                row["worst_relative_delta_ci_low"],
            )
            and audit.close(
                float(np.quantile(metrics["replicate_worst"], 0.975)),
                row["worst_relative_delta_ci_high"],
            ),
        )

    decision = read_json(results_dir / "stage3_decision.json")
    recomputed_passing: list[str] = []
    for regime in authorized:
        methods = metrics_by_regime.get(regime, {})
        primary = methods.get(PRIMARY)
        if primary is None:
            audit.check(section, f"primary_present_{regime}", False)
            continue
        randoms = [
            methods[f"random_seed{seed}"]["worst"] for seed in RANDOM_SEEDS
        ]
        gates = decision["development_gates"][regime]
        gate_a = (
            primary["worst"] < methods["robust_functional"]["worst"]
            and primary["worst"] < methods["fragility_robust"]["worst"]
        )
        gate_b = primary["worst"] < float(np.mean(randoms))
        gate_c = (
            primary["worst"] < methods["global_importance"]["worst"]
            and primary["worst"] < methods["average_specialization"]["worst"]
        )
        gate_d = int((primary["recovery"] > 0).sum()) >= 3
        comparator_mean = min(
            methods["global_importance"]["mean"],
            methods["average_specialization"]["mean"],
        )
        denominator = max(abs(comparator_mean), GATE_E_EPSILON)
        gate_e = (
            (primary["mean"] - comparator_mean) / denominator
            <= GATE_E_RELATIVE_TOLERANCE
        )
        for name, value in (
            ("gate_a", gate_a), ("gate_b", gate_b), ("gate_c", gate_c),
            ("gate_d", gate_d), ("gate_e", gate_e),
        ):
            audit.check(
                section,
                f"{name}_{regime}",
                bool(value) == gates[name]["passed"],
                {"recomputed": bool(value), "recorded": gates[name]["passed"]},
            )
        all_passed = gate_a and gate_b and gate_c and gate_d and gate_e
        audit.check(
            section, f"all_passed_{regime}", all_passed == gates["all_passed"]
        )
        if all_passed:
            recomputed_passing.append(regime)
    expected_decision = (
        "FINAL_CONFIRMATION_GO" if recomputed_passing else "MEASURED_DAMAGE_NO_GO"
    )
    if decision.get("phase") in ("development", "final"):
        audit.check(
            section,
            "decision_label",
            decision["development_decision"]["decision"] == expected_decision,
        )
        audit.check(
            section,
            "decision_regimes",
            sorted(decision["development_decision"]["authorized_regimes"])
            == sorted(recomputed_passing),
        )
    return True


def audit_seed44(audit: Audit, results_root: Path, results_dir: Path) -> None:
    section = "seed44_isolation"
    stage2b_dir = results_root / "robust_specialist_preservation"
    manifest = read_json(stage2b_dir / "splits" / "split_manifest.json")
    for domain in DOMAINS:
        entry = manifest["domains"][domain]["final"]
        path = stage2b_dir / "splits" / "final" / f"{domain}.npz"
        with np.load(path, allow_pickle=False) as data:
            input_ids = np.asarray(data["input_ids"])
        audit.check(
            section,
            f"final_hash_{domain}",
            array_sha256(input_ids) == entry["input_ids_sha256"],
        )
    decision_path = results_dir / "stage3_decision.json"
    final_authorized = False
    if decision_path.is_file():
        decision = read_json(decision_path)
        final_authorized = decision.get("decision") == "FINAL_CONFIRMATION_GO"
    final_losses = results_dir / "final_seed44" / "losses"
    outputs_exist = final_losses.exists() and any(final_losses.iterdir())
    audit.check(
        section,
        "no_unauthorized_final_outputs",
        (not outputs_exist) or final_authorized,
        {"outputs_exist": outputs_exist, "final_authorized": final_authorized},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/measured_damage_preservation"),
    )
    parser.add_argument(
        "--stage2b-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--stage2c-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.results_dir / "audits" / "independent_audit.json")

    audit = Audit()
    audit_prior_state(audit, args.results_root)
    damage = audit_damage(audit, args.results_dir)
    registry_info = audit_registry(
        audit, args.results_dir, args.stage2b_dir, args.stage2c_dir, damage
    )
    prereg = audit_preregistration(audit, args.results_dir, registry_info, damage)
    audit_split(audit, args.results_dir, prereg)
    additivity = audit_additivity(audit, args.results_dir, registry_info, damage)
    development_audited = audit_development(
        audit, args.results_dir, registry_info, additivity
    )
    audit_seed44(audit, args.results_root, args.results_dir)

    report = {
        "auditor": "audit_measured_damage_preservation",
        "production_analysis_functions_imported": False,
        "passed": audit.failed == 0,
        "checks_passed": audit.passed,
        "checks_failed": audit.failed,
        "max_numeric_difference": audit.max_numeric_difference,
        "sections": audit.sections,
        "damage_matrix_audited": damage is not None,
        "allocations_audited": registry_info is not None,
        "additivity_audited": additivity is not None,
        "development_results_audited": development_audited,
        "final_results_audited": (
            (args.results_dir / "final_seed44" / "final_results.json").is_file()
        ),
        "failures": audit.failures[:200],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Audit {'PASSED' if report['passed'] else 'FAILED'}: "
        f"{audit.passed} passed, {audit.failed} failed "
        f"(max numeric difference {audit.max_numeric_difference:.3e})"
    )
    for failure in audit.failures[:20]:
        print(f"  FAIL {failure['section']}/{failure['check']}: {failure['detail']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
