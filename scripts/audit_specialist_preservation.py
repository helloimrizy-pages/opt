#!/usr/bin/env python3
"""Independent Stage 2B audit.

This auditor deliberately imports no production analysis code from
``expert_analysis``. Every hash, score, constraint, metric, bootstrap interval,
gate, and decision is recomputed from raw saved artifacts using only the
standard library and numpy. Audit failure blocks scientific completion.
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
CALIBRATION_SEED = 20260815
CALIBRATION_PER_DOMAIN = 25
NUM_LAYERS = 16
NUM_EXPERTS = 64
PREFIX_IDS = (8982, 27, 187)
SEQUENCE_LENGTH = 68
MEASURED_PER_EXAMPLE = 64
PROTECTED_BITS = 8
BASE_BITS = {"4to8": 4, "3to8": 3}
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RANDOM_SEEDS = (1001, 1002, 1003, 1004, 1005)
SCALE_BITS_PER_GROUP = 16
GATE_C_RELATIVE_TOLERANCE = 0.10
GATE_C_ABSOLUTE_TOLERANCE = 1e-4
DEVELOPMENT_BUDGET = 0.20
FINAL_REQUIRED_POINT_BUDGETS = 3
QUALIFIED_REQUIRED_POINT_BUDGETS = 2
FINAL_REQUIRED_CI_WINS = 2
DETERMINISTIC_METHODS = (
    "robust_functional", "robust_routing", "average_specialization",
    "global_importance", "general_only", "math_only", "coding_only",
    "reasoning_only",
)
SINGLE_DOMAIN = {
    "general_only": "general", "math_only": "math",
    "coding_only": "coding", "reasoning_only": "reasoning",
}


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


def json_safe_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def expert_bytes(shapes: list[list[int]], bits: int, group_size: int) -> int:
    total = 0
    groups = 0
    for shape in shapes:
        weights = int(np.prod(shape))
        rows = int(np.prod(shape[:-1]))
        if bits == 16:
            total += weights * 2
            continue
        total += math.ceil(weights * bits / 8)
        groups += rows * math.ceil(shape[-1] / group_size)
    return total + groups * (SCALE_BITS_PER_GROUP // 8)


def layer_equalized(raw: np.ndarray) -> np.ndarray:
    totals = raw.sum(axis=1, keepdims=True)
    return raw / totals / raw.shape[0]


def specialization(imp: np.ndarray) -> np.ndarray:
    margins = np.empty_like(imp)
    for d in range(imp.shape[2]):
        others = np.delete(imp, d, axis=2)
        margins[:, :, d] = imp[:, :, d] - others.max(axis=2)
    positive = np.clip(margins, 0.0, None)
    return positive / positive.sum(axis=(0, 1))[None, None, :]


def audit_calibration(audit: Audit, root: Path, source_dir: Path) -> dict[str, Any]:
    section = "calibration_and_scores"
    metadata = read_json(root / "calibration" / "calibration_metadata.json")
    audit.check(section, "seed", metadata["calibration_seed"] == CALIBRATION_SEED)
    audit.check(
        section, "per_domain",
        metadata["calibration_examples_per_domain"] == CALIBRATION_PER_DOMAIN,
    )
    audit.check(
        section, "equal_total_budget",
        metadata["multi_domain_total_calibration_examples"] == 100
        and metadata["single_domain_total_calibration_examples"] == 100,
    )
    functional_raw = np.zeros((NUM_LAYERS, NUM_EXPERTS, 4))
    routing_raw = np.zeros_like(functional_raw)
    single_raw = np.zeros_like(functional_raw)
    for index, domain in enumerate(DOMAINS):
        rng = np.random.default_rng([CALIBRATION_SEED, DOMAIN_SEED_OFFSETS[domain]])
        expected_indices = np.sort(
            rng.choice(100, size=CALIBRATION_PER_DOMAIN, replace=False)
        ).tolist()
        recorded = metadata["domains_detail"][domain][
            "calibration_indices_into_frozen_set"
        ]
        audit.check(
            section, f"{domain}_indices_deterministic", recorded == expected_indices
        )
        with np.load(source_dir / "domains" / f"{domain}.npz", allow_pickle=False) as data:
            contribution = np.asarray(data["contribution_sums"], dtype=np.float64)
            routing = np.asarray(data["routing_counts"], dtype=np.float64)
        functional_raw[:, :, index] = contribution[recorded].sum(axis=0)
        routing_raw[:, :, index] = routing[recorded].sum(axis=0)
        single_raw[:, :, index] = contribution.sum(axis=0)
        with np.load(
            source_dir / "controlled_inputs" / f"{domain}.npz", allow_pickle=False
        ) as inputs:
            input_ids = np.asarray(inputs["input_ids"])
        row_hashes = [
            hashlib.sha256(
                np.ascontiguousarray(input_ids[i]).view(np.uint8)
            ).hexdigest()
            for i in recorded
        ]
        audit.check(
            section, f"{domain}_calibration_row_hashes",
            row_hashes == metadata["domains_detail"][domain][
                "calibration_input_row_sha256"
            ],
        )

    functional = np.stack(
        [layer_equalized(functional_raw[:, :, d]) for d in range(4)], axis=-1
    )
    routing = np.stack(
        [layer_equalized(routing_raw[:, :, d]) for d in range(4)], axis=-1
    )
    single = np.stack(
        [layer_equalized(single_raw[:, :, d]) for d in range(4)], axis=-1
    )
    with np.load(root / "calibration" / "functional_importance.npz") as data:
        audit.check(
            section, "functional_importance_matches",
            np.allclose(data["functional"], functional, atol=1e-12),
        )
        audit.check(
            section, "single_domain_matches",
            np.allclose(data["single_domain"], single, atol=1e-12),
        )
        audit.check(
            section, "global_importance_matches",
            np.allclose(data["global_importance"], functional.mean(axis=2), atol=1e-12),
        )
    functional_spec = specialization(functional)
    routing_spec = specialization(routing)
    with np.load(root / "calibration" / "functional_specialization.npz") as data:
        audit.check(
            section, "functional_specialization_matches",
            np.allclose(data["specialization"], functional_spec, atol=1e-12),
        )
        audit.check(
            section, "specialization_normalized",
            np.allclose(data["specialization"].sum(axis=(0, 1)), 1.0, atol=1e-9),
        )
    with np.load(root / "calibration" / "routing_specialization.npz") as data:
        audit.check(
            section, "routing_specialization_matches",
            np.allclose(data["specialization"], routing_spec, atol=1e-12),
        )
    for d in range(4):
        audit.check(
            section, f"importance_sums_to_one_domain{d}",
            abs(functional[:, :, d].sum() - 1.0) < 1e-9,
        )
    return {
        "functional": functional,
        "functional_spec": functional_spec,
        "routing_spec": routing_spec,
        "single": single,
    }


def audit_allocations(audit: Audit, root: Path, scores: dict[str, Any]) -> dict[str, Any]:
    section = "allocations_and_memory"
    allocations_dir = root / "allocations"
    registry = read_json(allocations_dir / "allocation_registry.json")
    audit.check(section, "registry_frozen", registry.get("frozen") is True)
    expected_registry_sha = canonical_sha256(
        {
            k: v
            for k, v in registry.items()
            if k not in ("registry_sha256", "created_at_utc")
        }
    )
    audit.check(
        section, "registry_sha", registry["registry_sha256"] == expected_registry_sha
    )
    with np.load(root / "calibration" / "memory_matrix.npz") as data:
        shapes = [list(map(int, shape)) for shape in data["tensor_shapes"].tolist()]
        group_size = int(data["group_size"][0])
        saved_bytes = {
            bits: np.asarray(data[f"bytes_bits{bits}"]) for bits in (3, 4, 8, 16)
        }
    recomputed = {bits: expert_bytes(shapes, bits, group_size) for bits in (3, 4, 8, 16)}
    for bits in (3, 4, 8, 16):
        audit.check(
            section, f"memory_bytes_bits{bits}",
            bool(np.all(saved_bytes[bits] == recomputed[bits])),
            {"recomputed": recomputed[bits]},
        )
    delta = {
        regime: saved_bytes[8] - saved_bytes[base] for regime, base in BASE_BITS.items()
    }
    budgets: dict[tuple[str, float], int] = {}
    for regime, base in BASE_BITS.items():
        total = int(delta[regime].sum())
        audit.check(
            section, f"{regime}_total_increment",
            registry["regimes"][regime]["total_increment_bytes"] == total,
        )
        for fraction in FRACTIONS:
            expected_budget = int(math.floor(fraction * total))
            recorded = registry["regimes"][regime]["budgets_bytes"][str(fraction)]
            audit.check(
                section, f"{regime}_budget_{fraction}", recorded == expected_budget
            )
            budgets[(regime, fraction)] = expected_budget

    records: dict[str, dict[str, Any]] = {}
    min_coverage: dict[tuple[str, float], dict[str, float]] = {}
    robust_objective: dict[tuple[str, float], float] = {}
    for entry in registry["entries"]:
        path = allocations_dir / entry["file"]
        audit.check(section, f"{entry['file']}_exists", path.is_file())
        audit.check(
            section, f"{entry['file']}_file_sha",
            file_sha256(path) == entry["file_sha256"],
        )
        record = read_json(path)
        records[entry["file"]] = record
        expected_sha = canonical_sha256(
            {
                k: v
                for k, v in record.items()
                if k not in ("allocation_sha256", "created_at_utc")
            }
        )
        audit.check(
            section, f"{entry['file']}_allocation_sha",
            record["allocation_sha256"] == expected_sha
            and record["allocation_sha256"] == entry["allocation_sha256"],
        )
        bits = np.asarray(record["expert_bits"])
        if record["method_kind"] == "uniform_reference":
            audit.check(
                section, f"{entry['file']}_uniform_bits",
                bool(np.all(bits == record["base_bits"])),
            )
            continue
        regime = record["regime"]
        base = BASE_BITS[regime]
        protected = (bits == PROTECTED_BITS).astype(np.int64)
        audit.check(
            section, f"{entry['file']}_bits_valid",
            bool(np.all(np.isin(bits, (base, PROTECTED_BITS)))),
        )
        listed = {(e["layer"], e["expert"]) for e in record["protected_experts"]}
        observed = {tuple(map(int, pair)) for pair in np.argwhere(protected == 1)}
        audit.check(section, f"{entry['file']}_protected_list", listed == observed)
        audit.check(
            section, f"{entry['file']}_protected_count",
            record["protected_expert_count"] == int(protected.sum()),
        )
        used = int((delta[regime] * protected).sum())
        audit.check(
            section, f"{entry['file']}_used_bytes",
            record["used_protection_bytes"] == used,
        )
        budget = budgets[(regime, record["budget_fraction"])]
        audit.check(section, f"{entry['file']}_feasible", used <= budget)
        audit.check(section, f"{entry['file']}_budget_bytes",
                    record["budget_bytes"] == budget)
        coverage = np.einsum(
            "led,le->d", scores["functional_spec"], protected.astype(np.float64)
        )
        for index, domain in enumerate(DOMAINS):
            audit.check(
                section, f"{entry['file']}_coverage_{domain}",
                audit.close(
                    record["functional_specialist_coverage"][domain],
                    float(coverage[index]),
                ),
            )
        key = (regime, record["budget_fraction"])
        min_coverage.setdefault(key, {})[record["method"]] = float(coverage.min())
        if record["method"] == "robust_functional":
            robust_objective[key] = float(record["solver_metadata"]["objective_z"])
            audit.check(
                section, f"{entry['file']}_z_equals_min_coverage",
                audit.close(
                    float(record["solver_metadata"]["objective_z"]),
                    float(coverage.min()),
                    1e-6,
                ),
            )
        total_bytes = 0
        for width in np.unique(bits):
            total_bytes += int(saved_bytes[int(width)][bits == width].sum())
        audit.check(
            section, f"{entry['file']}_total_projected_bytes",
            record["total_projected_expert_bytes"] == total_bytes,
        )
    for key, objective in robust_objective.items():
        for method, value in min_coverage[key].items():
            audit.check(
                "milp_optimality",
                f"{key}_{method}_not_above_optimum",
                value <= objective + 1e-6,
                {"method": method, "min_coverage": value, "optimum": objective},
            )
    return {"records": records, "registry": registry, "delta": delta, "budgets": budgets}


def audit_splits(audit: Audit, root: Path, source_dir: Path) -> None:
    section = "heldout_splits"
    manifest = read_json(root / "splits" / "split_manifest.json")
    audit.check(section, "development_seed", manifest["development_seed"] == 43)
    audit.check(section, "final_seed", manifest["final_seed"] == 44)
    audit.check(
        section, "geometry",
        manifest["measured_tokens_per_example"] == MEASURED_PER_EXAMPLE
        and manifest["model_sequence_length"] == SEQUENCE_LENGTH,
    )
    for domain in DOMAINS:
        with np.load(
            source_dir / "controlled_inputs" / f"{domain}.npz", allow_pickle=False
        ) as data:
            prior_rows = {
                bytes(np.ascontiguousarray(row[3:]).view(np.uint8))
                for row in np.asarray(data["input_ids"])
            }
        rows_by_split: dict[str, set[bytes]] = {}
        for split, expected_count in (("development", 50), ("final", 100)):
            entry = manifest["domains"][domain][split]
            with np.load(
                root / "splits" / split / f"{domain}.npz", allow_pickle=False
            ) as data:
                input_ids = np.asarray(data["input_ids"])
                mask = np.asarray(data["measurement_mask"])
            audit.check(
                section, f"{domain}_{split}_shape",
                input_ids.shape == (expected_count, SEQUENCE_LENGTH),
            )
            audit.check(
                section, f"{domain}_{split}_hash",
                array_sha256(input_ids) == entry["input_ids_sha256"],
            )
            audit.check(
                section, f"{domain}_{split}_prefix",
                bool(np.all(input_ids[:, :3] == np.asarray(PREFIX_IDS))),
            )
            audit.check(
                section, f"{domain}_{split}_measured",
                bool(np.all(mask.sum(axis=1) == MEASURED_PER_EXAMPLE)),
            )
            rows = {
                bytes(np.ascontiguousarray(row[3:]).view(np.uint8))
                for row in input_ids
            }
            audit.check(
                section, f"{domain}_{split}_unique_rows", len(rows) == expected_count
            )
            audit.check(
                section, f"{domain}_{split}_disjoint_from_prior",
                not (rows & prior_rows),
            )
            rows_by_split[split] = rows
        audit.check(
            section, f"{domain}_dev_final_disjoint",
            not (rows_by_split["development"] & rows_by_split["final"]),
        )


def load_losses(losses_dir: Path, slug: str) -> dict[str, np.ndarray]:
    output = {}
    for domain in DOMAINS:
        with np.load(losses_dir / slug / f"{domain}.npz", allow_pickle=False) as data:
            loss_sums = np.asarray(data["loss_sums"], dtype=np.float64)
            token_counts = np.asarray(data["token_counts"], dtype=np.float64)
        output[domain] = loss_sums / token_counts
    return output


def replicate_indices(count: int, replicates: int, seed: int, domain: str) -> np.ndarray:
    rng = np.random.default_rng([seed, DOMAIN_SEED_OFFSETS[domain]])
    return rng.integers(0, count, size=(replicates, count))


def method_metrics(
    alloc: dict[str, np.ndarray],
    bf16: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative, recovery, delta = {}, {}, {}
    replicates = next(iter(indices.values())).shape[0]
    rel_reps = np.zeros((replicates, 4))
    rec_reps = np.zeros((replicates, 4))
    for j, domain in enumerate(DOMAINS):
        a, b, u = alloc[domain], bf16[domain], base[domain]
        relative[domain] = (a.mean() - b.mean()) / b.mean()
        delta[domain] = a.mean() - b.mean()
        recovery[domain] = u.mean() - a.mean()
        idx = indices[domain]
        rel_reps[:, j] = (a[idx].mean(axis=1) - b[idx].mean(axis=1)) / b[idx].mean(axis=1)
        rec_reps[:, j] = u[idx].mean(axis=1) - a[idx].mean(axis=1)
    point = np.asarray([relative[d] for d in DOMAINS])
    return {
        "relative": relative,
        "delta": delta,
        "recovery": recovery,
        "mean_relative": float(point.mean()),
        "worst_relative": float(point.max()),
        "worst_domain": DOMAINS[int(point.argmax())],
        "rel_reps": rel_reps,
        "worst_reps": rel_reps.max(axis=1),
        "mean_reps": rel_reps.mean(axis=1),
        "rec_reps": rec_reps,
        "recovery_ci": {
            d: (
                float(np.quantile(rec_reps[:, j], 0.025)),
                float(np.quantile(rec_reps[:, j], 0.975)),
            )
            for j, d in enumerate(DOMAINS)
        },
        "worst_ci": (
            float(np.quantile(rel_reps.max(axis=1), 0.025)),
            float(np.quantile(rel_reps.max(axis=1), 0.975)),
        ),
    }


def audit_phase(
    audit: Audit,
    root: Path,
    phase: str,
    allocation_records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    section = f"{phase}_results"
    phase_dir = root / phase
    results_path = phase_dir / f"{phase}_results.json"
    if not results_path.is_file():
        return None
    results = read_json(results_path)
    run_config = read_json(phase_dir / "run_config.json")
    audit.check(
        section, "run_fingerprint_consistent",
        results["run_fingerprint"] == run_config["run_fingerprint"],
    )
    losses_dir = phase_dir / "losses"
    seed = results["bootstrap_seed"]
    replicates = results["bootstrap_replicates"]

    bf16 = load_losses(losses_dir, "bf16_reference")
    for domain in DOMAINS:
        audit.check(
            section, f"bf16_{domain}_finite",
            bool(np.all(np.isfinite(bf16[domain])) and np.all(bf16[domain] > 0)),
        )
    base_by_regime = {
        "4to8": load_losses(losses_dir, "uniform_4bit_reference"),
        "3to8": load_losses(losses_dir, "uniform_3bit_reference"),
    }
    counts = {domain: len(bf16[domain]) for domain in DOMAINS}
    indices = {
        domain: replicate_indices(counts[domain], replicates, seed, domain)
        for domain in DOMAINS
    }

    expected_budgets = [DEVELOPMENT_BUDGET] if phase == "development" else list(FRACTIONS)
    metrics: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for file_name, record in allocation_records.items():
        if record["method_kind"] == "uniform_reference":
            continue
        if record["budget_fraction"] not in expected_budgets:
            continue
        slug = file_name[: -len(".json")]
        alloc = load_losses(losses_dir, slug)
        for domain in DOMAINS:
            metadata = read_json(losses_dir / slug / f"{domain}.metadata.json")
            audit.check(
                section, f"{slug}_{domain}_checkpoint_fingerprints",
                metadata.get("run_fingerprint") == run_config["run_fingerprint"]
                and metadata.get("allocation_sha256") == record["allocation_sha256"],
            )
            audit.check(
                section, f"{slug}_{domain}_finite",
                bool(np.all(np.isfinite(alloc[domain]))),
            )
        key = (record["regime"], record["budget_fraction"])
        metrics.setdefault(key, {})[record["method"]] = method_metrics(
            alloc, bf16, base_by_regime[record["regime"]], indices
        )

    rows_by_key = {
        (row["method"], row["regime"], row["budget_fraction"]): row
        for row in results["method_rows"]
    }
    for (regime, budget), methods in metrics.items():
        for method, metric in methods.items():
            row = rows_by_key[(method, regime, budget)]
            checks = [
                ("worst_relative_delta", metric["worst_relative"]),
                ("mean_relative_delta", metric["mean_relative"]),
                ("worst_relative_delta_ci_low", metric["worst_ci"][0]),
                ("worst_relative_delta_ci_high", metric["worst_ci"][1]),
            ]
            for domain in DOMAINS:
                checks.append(
                    (f"relative_delta_{domain}", metric["relative"][domain])
                )
                checks.append((f"delta_nll_{domain}", metric["delta"][domain]))
                checks.append((f"recovery_{domain}", metric["recovery"][domain]))
            for field, expected in checks:
                audit.check(
                    section,
                    f"{method}_{regime}_{budget}_{field}",
                    audit.close(json_safe_float(row[field]), expected, 1e-9),
                    {"recorded": row[field], "recomputed": expected},
                )
            audit.check(
                section, f"{method}_{regime}_{budget}_worst_domain",
                row["worst_domain"] == metric["worst_domain"],
            )

    if phase == "development":
        decision = read_json(root / "stage2b_decision.json")
        gates_ok = {}
        for regime in BASE_BITS:
            methods = metrics[(regime, DEVELOPMENT_BUDGET)]
            robust = methods["robust_functional"]
            randoms = [
                m for name, m in methods.items() if name.startswith("random_seed")
            ]
            random_mean_worst = float(
                np.mean([m["worst_relative"] for m in randoms])
            )
            gate_a = robust["worst_relative"] < random_mean_worst
            gate_b = (
                robust["worst_relative"]
                < methods["global_importance"]["worst_relative"]
                and robust["worst_relative"]
                < methods["average_specialization"]["worst_relative"]
            )
            average_mean = methods["average_specialization"]["mean_relative"]
            tolerance = max(
                GATE_C_RELATIVE_TOLERANCE * abs(average_mean),
                GATE_C_ABSOLUTE_TOLERANCE,
            )
            gate_c = robust["mean_relative"] <= average_mean + tolerance
            gate_d = (
                sum(1 for d in DOMAINS if robust["recovery"][d] > 0) >= 3
            )
            gates_ok[regime] = gate_a and gate_b and gate_c and gate_d
            recorded_gates = decision["development_gates"][regime]
            for name, value in (
                ("gate_a", gate_a), ("gate_b", gate_b),
                ("gate_c", gate_c), ("gate_d", gate_d),
            ):
                audit.check(
                    section, f"{regime}_{name}_recomputed",
                    recorded_gates[name]["passed"] == value,
                    {"recorded": recorded_gates[name]["passed"], "recomputed": value},
                )
        expected_decision = (
            "FULL_EVALUATION_GO" if any(gates_ok.values())
            else "ROBUST_PRESERVATION_NO_GO"
        )
        audit.check(
            section, "development_decision_recomputed",
            decision["decision"] == expected_decision,
            {"recorded": decision["decision"], "recomputed": expected_decision},
        )
    else:
        assessments = {}
        for regime in BASE_BITS:
            all_three = 0
            improvements = []
            ci_wins = 0
            catastrophic = []
            for budget in FRACTIONS:
                methods = metrics[(regime, budget)]
                robust = methods["robust_functional"]
                randoms = [
                    m for name, m in methods.items()
                    if name.startswith("random_seed")
                ]
                random_mean_worst = float(
                    np.mean([m["worst_relative"] for m in randoms])
                )
                wins = (
                    robust["worst_relative"] < random_mean_worst
                    and robust["worst_relative"]
                    < methods["global_importance"]["worst_relative"]
                    and robust["worst_relative"]
                    < methods["average_specialization"]["worst_relative"]
                )
                all_three += int(wins)
                improvements.append(
                    methods["average_specialization"]["worst_relative"]
                    - robust["worst_relative"]
                )
                difference = (
                    robust["worst_reps"]
                    - methods["average_specialization"]["worst_reps"]
                )
                if float(np.quantile(difference, 0.975)) < 0:
                    ci_wins += 1
            for domain in DOMAINS:
                if all(
                    metrics[(regime, budget)]["robust_functional"]["recovery"][domain]
                    < 0
                    and metrics[(regime, budget)]["robust_functional"][
                        "recovery_ci"
                    ][domain][1]
                    < 0
                    for budget in FRACTIONS
                ):
                    catastrophic.append(domain)
            average_improvement = float(np.mean(improvements))
            assessments[regime] = {
                "strong": all_three >= FINAL_REQUIRED_POINT_BUDGETS
                and average_improvement > 0
                and ci_wins >= FINAL_REQUIRED_CI_WINS
                and not catastrophic,
                "qualified": all_three >= QUALIFIED_REQUIRED_POINT_BUDGETS
                and average_improvement > 0
                and not catastrophic,
            }
            recorded = results["final_regime_assessments"][regime]
            audit.check(
                section, f"{regime}_strong_recomputed",
                recorded["strong_success"] == assessments[regime]["strong"],
            )
            audit.check(
                section, f"{regime}_qualified_recomputed",
                recorded["qualified_success"] == assessments[regime]["qualified"],
            )
        if any(a["strong"] for a in assessments.values()):
            expected_label = "STRONG SUCCESS"
        elif any(a["qualified"] for a in assessments.values()):
            expected_label = "SUCCESS WITH QUALIFICATIONS"
        else:
            expected_label = "NEGATIVE RESULT"
        audit.check(
            section, "final_decision_recomputed",
            results["final_decision"]["decision"] == expected_label,
            {
                "recorded": results["final_decision"]["decision"],
                "recomputed": expected_label,
            },
        )
        for row in results["single_domain_transfer"]:
            metric = metrics[(row["regime"], row["budget_fraction"])][
                row["calibration_method"]
            ]
            audit.check(
                section,
                f"transfer_{row['regime']}_{row['calibration_method']}_"
                f"{row['evaluation_domain']}",
                audit.close(
                    row["relative_delta"],
                    metric["relative"][row["evaluation_domain"]],
                    1e-9,
                ),
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.results_dir
    audit = Audit()

    scores = audit_calibration(audit, root, args.source_dir)
    allocation_state = audit_allocations(audit, root, scores)
    audit_splits(audit, root, args.source_dir)
    development = audit_phase(
        audit, root, "development", allocation_state["records"]
    )
    final = audit_phase(audit, root, "final", allocation_state["records"])

    report = {
        "auditor": "standalone_stage2b_audit_v1",
        "production_analysis_functions_imported": False,
        "passed": audit.failed == 0,
        "checks_passed": audit.passed,
        "checks_failed": audit.failed,
        "max_numeric_difference": audit.max_numeric_difference,
        "sections": audit.sections,
        "development_results_audited": development is not None,
        "final_results_audited": final is not None,
        "failures": audit.failures[:200],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output or (root / "audits" / "independent_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Independent audit: {audit.passed} passed, {audit.failed} failed; "
        f"max numeric difference {audit.max_numeric_difference:.3e}"
    )
    print(f"Report: {output}")
    return 0 if audit.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
