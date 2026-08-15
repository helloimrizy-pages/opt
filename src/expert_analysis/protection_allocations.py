"""Stage 2B allocation generation, freezing, and score-space sanity analysis.

All allocations for both precision regimes, all four protection budgets, every
deterministic method, and all five random seeds are generated and hashed before
any held-out NLL is computed. After the registry is frozen the files must never
be regenerated or edited; the evaluator and auditor verify the recorded hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .balanced import canonical_sha256, file_sha256
from .io_utils import atomic_write_json, read_json, write_csv
from .protection_optimization import (
    BASE_BITS_BY_REGIME,
    PROTECTED_BITS,
    PROTECTION_FRACTIONS,
    RANDOM_ALLOCATION_SEEDS,
    ExpertMemoryMatrix,
    SolverResult,
    bits_matrix_for_allocation,
    random_allocation,
    solve_max_min_coverage,
    solve_weighted_selection,
    uniform_bits_matrix,
)
from .specialist_preservation import (
    STAGE2B_DOMAINS,
    SpecialistScores,
    specialist_coverage,
)

ALLOCATION_SCHEMA_VERSION = "stage2b_allocation_v1"

DETERMINISTIC_METHODS = (
    ("robust_functional", "Robust-Functional"),
    ("robust_routing", "Robust-Routing"),
    ("average_specialization", "Average-Specialization"),
    ("global_importance", "Global-Importance"),
    ("general_only", "General-Only"),
    ("math_only", "Math-Only"),
    ("coding_only", "Coding-Only"),
    ("reasoning_only", "Reasoning-Only"),
)
METHOD_LABELS = dict(DETERMINISTIC_METHODS)
SINGLE_DOMAIN_METHODS = {
    "general_only": "general",
    "math_only": "math",
    "coding_only": "coding",
    "reasoning_only": "reasoning",
}
UNIFORM_REFERENCES = (
    ("bf16_reference", 16, "BF16 baseline"),
    ("uniform_8bit_reference", 8, "Uniform 8-bit"),
    ("uniform_4bit_reference", 4, "Uniform 4-bit"),
    ("uniform_3bit_reference", 3, "Uniform 3-bit"),
)


def allocation_file_name(method: str, regime: str, fraction: float) -> str:
    return f"{method}_{regime}_budget{int(round(fraction * 100))}.json"


def _solve_deterministic(
    method: str,
    scores: SpecialistScores,
    delta_bytes: np.ndarray,
    budget_bytes: int,
) -> SolverResult:
    if method == "robust_functional":
        return solve_max_min_coverage(
            scores.functional_specialization, delta_bytes, budget_bytes
        )
    if method == "robust_routing":
        return solve_max_min_coverage(
            scores.routing_specialization, delta_bytes, budget_bytes
        )
    if method == "average_specialization":
        weights = scores.functional_specialization.mean(axis=2)
        return solve_weighted_selection(
            weights, delta_bytes, budget_bytes, "average_specialization"
        )
    if method == "global_importance":
        return solve_weighted_selection(
            scores.global_importance, delta_bytes, budget_bytes, "global_importance"
        )
    if method in SINGLE_DOMAIN_METHODS:
        domain = SINGLE_DOMAIN_METHODS[method]
        domain_index = STAGE2B_DOMAINS.index(domain)
        weights = scores.single_domain[:, :, domain_index]
        return solve_weighted_selection(
            weights, delta_bytes, budget_bytes, f"single_domain_{domain}"
        )
    raise ValueError(f"Unknown deterministic method {method!r}")


def build_allocation_record(
    method: str,
    method_label: str,
    method_kind: str,
    regime: str,
    fraction: float,
    solution: SolverResult,
    scores: SpecialistScores,
    memory: ExpertMemoryMatrix,
) -> dict[str, Any]:
    base_bits = BASE_BITS_BY_REGIME[regime]
    delta = memory.delta_protection_bytes(base_bits)
    budget = memory.protection_budget_bytes(base_bits, fraction)
    x = solution.protected
    bits = bits_matrix_for_allocation(x, base_bits)
    used = int((delta * x.astype(np.int64)).sum())
    if used > budget:
        raise RuntimeError(f"Allocation {method}/{regime}/{fraction} exceeds its budget")
    functional_coverage = specialist_coverage(scores.functional_specialization, x)
    routing_coverage = specialist_coverage(scores.routing_specialization, x)
    protected_experts = [
        {"layer": int(layer), "expert": int(expert)}
        for layer, expert in np.argwhere(x == 1).tolist()
    ]
    record: dict[str, Any] = {
        "schema": ALLOCATION_SCHEMA_VERSION,
        "stage": "stage2b_robust_specialist_preservation",
        "method": method,
        "method_label": method_label,
        "method_kind": method_kind,
        "matched_budget_competitor": True,
        "regime": regime,
        "base_bits": base_bits,
        "protected_bits": PROTECTED_BITS,
        "group_size": memory.group_size,
        "budget_fraction": fraction,
        "budget_bytes": budget,
        "total_increment_bytes": memory.total_increment_bytes(base_bits),
        "used_protection_bytes": used,
        "budget_utilization": used / budget if budget > 0 else 0.0,
        "expert_bits": bits.tolist(),
        "protected_experts": protected_experts,
        "protected_expert_count": int(x.sum()),
        "protected_experts_per_layer": x.sum(axis=1).astype(int).tolist(),
        "total_projected_expert_bytes": memory.allocation_bytes(bits),
        "bf16_reference_expert_bytes": int(memory.bytes_by_bits[16].sum()),
        "effective_bits_per_weight": memory.effective_bits_per_weight(bits),
        "functional_specialist_coverage": {
            domain: float(functional_coverage[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "functional_specialist_coverage_min": float(functional_coverage.min()),
        "functional_specialist_coverage_mean": float(functional_coverage.mean()),
        "routing_specialist_coverage": {
            domain: float(routing_coverage[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "routing_specialist_coverage_min": float(routing_coverage.min()),
        "calibration_fingerprint": scores.metadata["calibration_fingerprint"],
        "calibration_hashes": {
            domain: scores.metadata["domains_detail"][domain][
                "single_domain_input_ids_sha256"
            ]
            for domain in STAGE2B_DOMAINS
        },
        "score_hashes": scores.metadata["score_hashes"],
        "solver_metadata": solution.metadata,
    }
    record["allocation_sha256"] = canonical_sha256(record)
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def build_uniform_reference_record(
    name: str,
    bits: int,
    label: str,
    scores: SpecialistScores,
    memory: ExpertMemoryMatrix,
) -> dict[str, Any]:
    bits_matrix = uniform_bits_matrix(bits, memory.num_layers, memory.num_experts)
    record: dict[str, Any] = {
        "schema": ALLOCATION_SCHEMA_VERSION,
        "stage": "stage2b_robust_specialist_preservation",
        "method": name,
        "method_label": label,
        "method_kind": "uniform_reference",
        "matched_budget_competitor": False,
        "matched_budget_note": (
            "reference operating point; not presented as a matched-budget "
            "competitor of any partial protection method"
        ),
        "regime": None,
        "base_bits": bits,
        "protected_bits": None,
        "group_size": memory.group_size,
        "budget_fraction": None,
        "budget_bytes": None,
        "expert_bits": bits_matrix.tolist(),
        "protected_experts": [],
        "protected_expert_count": 0,
        "total_projected_expert_bytes": memory.allocation_bytes(bits_matrix),
        "bf16_reference_expert_bytes": int(memory.bytes_by_bits[16].sum()),
        "effective_bits_per_weight": memory.effective_bits_per_weight(bits_matrix),
        "calibration_fingerprint": scores.metadata["calibration_fingerprint"],
        "score_hashes": scores.metadata["score_hashes"],
        "solver_metadata": {"solver": "uniform_reference", "bits": bits},
    }
    record["allocation_sha256"] = canonical_sha256(record)
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def allocation_deterministic_content(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in ("allocation_sha256", "created_at_utc")
    }


def verify_allocation_record(record: Mapping[str, Any]) -> None:
    expected = canonical_sha256(allocation_deterministic_content(record))
    if record.get("allocation_sha256") != expected:
        raise RuntimeError(
            f"Allocation {record.get('method')}/{record.get('regime')}/"
            f"{record.get('budget_fraction')} failed its SHA-256 integrity check"
        )


def generate_all_allocations(
    scores: SpecialistScores,
    memory: ExpertMemoryMatrix,
    allocations_dir: Path,
) -> dict[str, Any]:
    """Solve and write every preregistered allocation, then freeze the registry."""

    allocations_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for regime, base_bits in BASE_BITS_BY_REGIME.items():
        delta = memory.delta_protection_bytes(base_bits)
        for fraction in PROTECTION_FRACTIONS:
            budget = memory.protection_budget_bytes(base_bits, fraction)
            robust_objective: float | None = None
            for method, label in DETERMINISTIC_METHODS:
                solution = _solve_deterministic(method, scores, delta, budget)
                record = build_allocation_record(
                    method, label, "deterministic_milp", regime, fraction,
                    solution, scores, memory,
                )
                if method == "robust_functional":
                    robust_objective = solution.objective_value
                elif robust_objective is not None:
                    other_min = record["functional_specialist_coverage_min"]
                    if other_min > robust_objective + 1e-6:
                        raise RuntimeError(
                            f"{method} at {regime}/{fraction} achieves min functional "
                            "coverage above the Robust-Functional MILP optimum; the "
                            "max-min solve cannot be optimal"
                        )
                file_name = allocation_file_name(method, regime, fraction)
                atomic_write_json(allocations_dir / file_name, record)
                entries.append(_registry_entry(allocations_dir / file_name, record))
            for seed in RANDOM_ALLOCATION_SEEDS:
                solution = random_allocation(delta, budget, seed)
                record = build_allocation_record(
                    f"random_seed{seed}", f"Random (seed {seed})", "random",
                    regime, fraction, solution, scores, memory,
                )
                file_name = allocation_file_name(f"random_seed{seed}", regime, fraction)
                atomic_write_json(allocations_dir / file_name, record)
                entries.append(_registry_entry(allocations_dir / file_name, record))
    for name, bits, label in UNIFORM_REFERENCES:
        record = build_uniform_reference_record(name, bits, label, scores, memory)
        path = allocations_dir / f"{name}.json"
        atomic_write_json(path, record)
        entries.append(_registry_entry(path, record))

    registry: dict[str, Any] = {
        "schema": "stage2b_allocation_registry_v1",
        "stage": "stage2b_robust_specialist_preservation",
        "frozen": True,
        "frozen_before_any_heldout_nll": True,
        "regimes": {
            regime: {
                "base_bits": base,
                "protected_bits": PROTECTED_BITS,
                "total_increment_bytes": memory.total_increment_bytes(base),
                "budgets_bytes": {
                    str(fraction): memory.protection_budget_bytes(base, fraction)
                    for fraction in PROTECTION_FRACTIONS
                },
            }
            for regime, base in BASE_BITS_BY_REGIME.items()
        },
        "protection_fractions": list(PROTECTION_FRACTIONS),
        "random_seeds": list(RANDOM_ALLOCATION_SEEDS),
        "calibration_fingerprint": scores.metadata["calibration_fingerprint"],
        "score_hashes": scores.metadata["score_hashes"],
        "entries": entries,
    }
    registry["registry_sha256"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    registry["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(allocations_dir / "allocation_registry.json", registry)
    return registry


def _registry_entry(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file": path.name,
        "method": record["method"],
        "method_label": record["method_label"],
        "method_kind": record["method_kind"],
        "regime": record["regime"],
        "budget_fraction": record["budget_fraction"],
        "allocation_sha256": record["allocation_sha256"],
        "file_sha256": file_sha256(path),
    }


def load_frozen_registry(allocations_dir: Path) -> dict[str, Any]:
    """Load the registry and fail loudly if any frozen artifact was altered."""

    registry_path = allocations_dir / "allocation_registry.json"
    registry = read_json(registry_path)
    if registry.get("frozen") is not True:
        raise RuntimeError("The allocation registry is not marked frozen")
    expected = canonical_sha256(
        {
            key: value
            for key, value in registry.items()
            if key not in ("registry_sha256", "created_at_utc")
        }
    )
    if registry.get("registry_sha256") != expected:
        raise RuntimeError("allocation_registry.json failed its integrity hash")
    for entry in registry["entries"]:
        path = allocations_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"Frozen allocation {entry['file']} is missing")
        record = read_json(path)
        verify_allocation_record(record)
        if record["allocation_sha256"] != entry["allocation_sha256"]:
            raise RuntimeError(
                f"Frozen allocation {entry['file']} does not match the registry hash"
            )
    return registry


def load_allocation(allocations_dir: Path, file_name: str) -> dict[str, Any]:
    record = read_json(allocations_dir / file_name)
    verify_allocation_record(record)
    return record


def write_allocation_summaries(
    allocations_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Score-space sanity tables generated before any model evaluation."""

    registry = load_frozen_registry(allocations_dir)
    coverage_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        record = load_allocation(allocations_dir, entry["file"])
        base_row = {
            "method": record["method"],
            "method_label": record["method_label"],
            "method_kind": record["method_kind"],
            "regime": record["regime"],
            "budget_fraction": record["budget_fraction"],
            "budget_bytes": record.get("budget_bytes"),
            "used_protection_bytes": record.get("used_protection_bytes"),
            "budget_utilization": record.get("budget_utilization"),
            "protected_expert_count": record["protected_expert_count"],
            "total_projected_expert_bytes": record["total_projected_expert_bytes"],
            "effective_bits_per_weight": record["effective_bits_per_weight"],
        }
        summary_rows.append(
            {
                **base_row,
                "protected_experts_per_layer": ";".join(
                    str(v) for v in record.get("protected_experts_per_layer", [])
                ),
                "allocation_sha256": record["allocation_sha256"],
                "file": entry["file"],
            }
        )
        if record["method_kind"] != "uniform_reference":
            coverage = record["functional_specialist_coverage"]
            coverage_rows.append(
                {
                    **base_row,
                    **{f"coverage_{domain}": coverage[domain] for domain in STAGE2B_DOMAINS},
                    "coverage_min": record["functional_specialist_coverage_min"],
                    "coverage_mean": record["functional_specialist_coverage_mean"],
                    "routing_coverage_min": record["routing_specialist_coverage_min"],
                }
            )
    coverage_path = output_dir / "coverage_summary.csv"
    summary_path = output_dir / "allocation_summary.csv"
    write_csv(
        coverage_path,
        coverage_rows,
        [
            "method", "method_label", "method_kind", "regime", "budget_fraction",
            "budget_bytes", "used_protection_bytes", "budget_utilization",
            "protected_expert_count", "total_projected_expert_bytes",
            "effective_bits_per_weight",
            *[f"coverage_{domain}" for domain in STAGE2B_DOMAINS],
            "coverage_min", "coverage_mean", "routing_coverage_min",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "method", "method_label", "method_kind", "regime", "budget_fraction",
            "budget_bytes", "used_protection_bytes", "budget_utilization",
            "protected_expert_count", "total_projected_expert_bytes",
            "effective_bits_per_weight", "protected_experts_per_layer",
            "allocation_sha256", "file",
        ],
    )
    return coverage_path, summary_path
