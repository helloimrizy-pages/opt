"""Stage 3 Measured-Damage-Robust optimization, frozen allocations, registry.

The PRIMARY Stage 3 method minimizes the largest additively predicted domain
delta NLL under the exact Stage 2B memory budgets:

    minimize z
    subject to  PredictedDelta_d(x) <= z            for every domain
                sum_{l,e} DeltaM[l,e] * x[l,e] <= Budget
                x[l,e] binary

where ``PredictedDelta_d(x) = BaseTotal_d - sum_{l,e} r[l,e,d] x[l,e]`` with
measured per-expert benefits ``r[l,e,d] = m[l,e,d,base] - m[l,e,d,8]``. Every
coefficient is a measured calibration loss difference; no score, surrogate,
fragility weight, fitted coefficient, or tuned term enters the objective, and
negative measured values are used exactly as measured. All eight regime/budget
allocations are generated and frozen before any probe or development NLL is
inspected. Comparator allocations are never regenerated; the frozen Stage 2B
and Stage 2C files are reused by hash.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from .balanced import canonical_sha256, file_sha256
from .fragility import Stage2BScoreArtifacts
from .io_utils import atomic_write_json, json_safe, read_json, write_csv
from .measured_damage import (
    STAGE3_PROFILE_BITS,
    STAGE3_REGIMES,
    STAGE3_STAGE,
    predicted_domain_delta_nll,
    profile_bits_index,
)
from .protection_allocations import (
    allocation_file_name,
    verify_allocation_record,
)
from .protection_optimization import (
    BASE_BITS_BY_REGIME,
    PROTECTED_BITS,
    PROTECTION_FRACTIONS,
    RANDOM_ALLOCATION_SEEDS,
    ExpertMemoryMatrix,
    SolverResult,
    bits_matrix_for_allocation,
)
from .specialist_preservation import STAGE2B_DOMAINS, specialist_coverage

STAGE3_ALLOCATION_SCHEMA = "stage3_allocation_v1"
STAGE3_REGISTRY_SCHEMA = "stage3_allocation_registry_v1"
MEASURED_DAMAGE_METHOD = "measured_damage_robust"
MEASURED_DAMAGE_LABEL = "Measured-Damage-Robust"
STAGE3_DEVELOPMENT_SEED = 46
STAGE3_FINAL_SEED = 44
# Frozen comparators reused by hash: every Stage 2B method plus the Stage 2C
# Fragility-Robust primary method.
REUSED_STAGE2B_METHODS = (
    "robust_functional",
    "robust_routing",
    "average_specialization",
    "global_importance",
    "general_only",
    "math_only",
    "coding_only",
    "reasoning_only",
)
REUSED_STAGE2C_METHODS = ("fragility_robust",)


def regime_damage_slices(
    delta_nll: np.ndarray, regime: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(base_delta, protected_delta, benefit)`` for one regime."""

    if regime not in STAGE3_REGIMES:
        raise ValueError(f"Regime {regime!r} is not preregistered")
    delta = np.asarray(delta_nll, dtype=np.float64)
    base_delta = delta[:, :, :, profile_bits_index(STAGE3_REGIMES[regime])]
    protected_delta = delta[:, :, :, profile_bits_index(PROTECTED_BITS)]
    return base_delta, protected_delta, base_delta - protected_delta


def solve_measured_damage_robust(
    delta_nll: np.ndarray,
    regime: str,
    delta_bytes: np.ndarray,
    budget_bytes: int,
    mip_rel_gap: float = 0.0,
) -> SolverResult:
    """Minimize the maximum additively predicted domain delta NLL."""

    base_delta, protected_delta, benefit = regime_damage_slices(delta_nll, regime)
    delta = np.asarray(delta_bytes, dtype=np.float64)
    if delta.ndim != 2 or np.any(delta <= 0):
        raise ValueError("Every protection increment must be positive")
    if budget_bytes < 0:
        raise ValueError("The protection budget cannot be negative")
    if base_delta.shape[:2] != delta.shape:
        raise ValueError("Damage deltas and memory increments do not align")
    if not np.all(np.isfinite(benefit)):
        raise ValueError("Measured benefits contain non-finite values")
    base_total = base_delta.sum(axis=(0, 1))
    n = delta.size
    c = np.zeros(n + 1, dtype=np.float64)
    c[-1] = 1.0
    # BaseTotal_d - r_d . x <= z  <=>  r_d . x + z >= BaseTotal_d
    benefit_rows = np.concatenate(
        [
            benefit.reshape(n, len(STAGE2B_DOMAINS)).T,
            np.ones((len(STAGE2B_DOMAINS), 1)),
        ],
        axis=1,
    )
    memory_row = np.concatenate([delta.reshape(1, n), np.zeros((1, 1))], axis=1)
    constraints = [
        LinearConstraint(benefit_rows, lb=base_total, ub=np.inf),
        LinearConstraint(memory_row, lb=-np.inf, ub=float(budget_bytes)),
    ]
    integrality = np.concatenate([np.ones(n), np.zeros(1)])
    bounds = Bounds(
        lb=np.concatenate([np.zeros(n), np.asarray([-np.inf])]),
        ub=np.concatenate([np.ones(n), np.asarray([np.inf])]),
    )
    started = time.monotonic()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"mip_rel_gap": mip_rel_gap},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Measured-Damage-Robust MILP failed: {result.message}")
    rounded = np.round(result.x[:n])
    if np.max(np.abs(result.x[:n] - rounded)) > 1e-6:
        raise RuntimeError("MILP returned a non-integral protection variable")
    x = rounded.astype(np.uint8).reshape(delta.shape)
    used = int((np.asarray(delta_bytes, dtype=np.int64) * x).sum())
    if used > budget_bytes:
        raise RuntimeError(
            f"MILP solution uses {used} bytes, exceeding the {budget_bytes}-byte budget"
        )
    solver_z = float(result.x[-1])
    bits = bits_matrix_for_allocation(x, BASE_BITS_BY_REGIME[regime])
    predicted = predicted_domain_delta_nll(bits, delta_nll)
    achieved_z = float(predicted.max())
    if abs(achieved_z - solver_z) > 1e-5:
        raise RuntimeError(
            f"Measured-Damage MILP z={solver_z} disagrees with the recomputed "
            f"maximum predicted delta {achieved_z} beyond solver tolerance"
        )
    metadata: dict[str, Any] = {
        "solver": "scipy.optimize.milp",
        "backend": "HiGHS",
        "scipy_version": scipy.__version__,
        "problem": "min_max_measured_additive_domain_delta_nll",
        "objective_sense": "minimize",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_success": bool(result.success),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
        "mip_dual_bound": float(getattr(result, "mip_dual_bound", float("nan"))),
        "mip_gap": float(getattr(result, "mip_gap", float("nan"))),
        "runtime_seconds": time.monotonic() - started,
        "budget_bytes": int(budget_bytes),
        "used_protection_bytes": used,
        "budget_utilization": used / budget_bytes if budget_bytes > 0 else 0.0,
        "memory_constraint_residual_bytes": int(budget_bytes - used),
        "protected_expert_count": int(x.sum()),
        "objective_max_predicted_delta_nll": achieved_z,
        "solver_reported_z": solver_z,
        "base_total_delta_nll_by_domain": {
            domain: float(base_total[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "predicted_delta_nll_by_domain": {
            domain: float(predicted[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "negative_benefit_cells": int((benefit < 0).sum()),
        "negative_base_damage_cells": int((base_delta < 0).sum()),
    }
    return SolverResult(protected=x, objective_value=achieved_z, metadata=metadata)


def build_measured_allocation_record(
    regime: str,
    fraction: float,
    solution: SolverResult,
    scores: Stage2BScoreArtifacts,
    memory: ExpertMemoryMatrix,
    damage_record: Mapping[str, Any],
    delta_nll: np.ndarray,
) -> dict[str, Any]:
    """One frozen Measured-Damage-Robust allocation with full provenance."""

    base_bits = BASE_BITS_BY_REGIME[regime]
    delta_bytes = memory.delta_protection_bytes(base_bits)
    budget = memory.protection_budget_bytes(base_bits, fraction)
    x = solution.protected
    bits = bits_matrix_for_allocation(x, base_bits)
    used = int((delta_bytes * x.astype(np.int64)).sum())
    if used > budget:
        raise RuntimeError(
            f"Measured-Damage allocation {regime}/{fraction} exceeds its budget"
        )
    predicted = predicted_domain_delta_nll(bits, delta_nll)
    bf16 = np.asarray(
        [damage_record["bf16_nll"][domain] for domain in STAGE2B_DOMAINS],
        dtype=np.float64,
    )
    functional_coverage = specialist_coverage(scores.functional_specialization, x)
    record: dict[str, Any] = {
        "schema": STAGE3_ALLOCATION_SCHEMA,
        "stage": STAGE3_STAGE,
        "method": MEASURED_DAMAGE_METHOD,
        "method_label": MEASURED_DAMAGE_LABEL,
        "method_kind": "deterministic_milp",
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
        "protected_experts": [
            {"layer": int(layer), "expert": int(expert)}
            for layer, expert in np.argwhere(x == 1).tolist()
        ],
        "protected_expert_count": int(x.sum()),
        "protected_experts_per_layer": x.sum(axis=1).astype(int).tolist(),
        "total_projected_expert_bytes": memory.allocation_bytes(bits),
        "bf16_reference_expert_bytes": int(memory.bytes_by_bits[16].sum()),
        "effective_bits_per_weight": memory.effective_bits_per_weight(bits),
        "predicted_domain_delta_nll": {
            domain: float(predicted[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "predicted_worst_delta_nll": float(predicted.max()),
        "predicted_domain_relative_delta": {
            domain: float(predicted[index] / bf16[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        # Diagnostic continuity with Stage 2B/2C; never enters the objective.
        "functional_specialist_coverage": {
            domain: float(functional_coverage[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "functional_specialist_coverage_min": float(functional_coverage.min()),
        "damage_sha256": damage_record["damage_sha256"],
        "calibration_fingerprint": scores.calibration_fingerprint,
        "score_hashes": scores.score_hashes,
        "solver_metadata": solution.metadata,
    }
    record = json_safe(record)
    record["allocation_sha256"] = canonical_sha256(record)
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def _reused_entries(
    registry: Mapping[str, Any],
    allocations_dir: Path,
    entry_key: str,
    source: str,
) -> list[dict[str, Any]]:
    """Reuse frozen allocations by hash, verifying every file on disk."""

    entries: list[dict[str, Any]] = []
    for entry in registry[entry_key]:
        path = allocations_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"Frozen allocation {entry['file']} is missing")
        if file_sha256(path) != entry["file_sha256"]:
            raise RuntimeError(
                f"Frozen allocation {entry['file']} does not match its registry "
                "file hash"
            )
        record = read_json(path)
        verify_allocation_record(record)
        if record["allocation_sha256"] != entry["allocation_sha256"]:
            raise RuntimeError(
                f"Frozen allocation {entry['file']} content hash mismatch"
            )
        entries.append(
            {
                "file": entry["file"],
                "source": source,
                "method": entry["method"],
                "method_label": entry["method_label"],
                "method_kind": entry["method_kind"],
                "regime": entry["regime"],
                "budget_fraction": entry["budget_fraction"],
                "allocation_sha256": entry["allocation_sha256"],
                "file_sha256": entry["file_sha256"],
            }
        )
    return entries


def generate_stage3_allocations(
    scores: Stage2BScoreArtifacts,
    memory: ExpertMemoryMatrix,
    damage_record: Mapping[str, Any],
    delta_nll: np.ndarray,
    allocations_dir: Path,
    stage2b_registry: Mapping[str, Any],
    stage2b_allocations_dir: Path,
    stage2c_registry: Mapping[str, Any],
    stage2c_allocations_dir: Path,
) -> dict[str, Any]:
    """Solve and freeze all eight Stage 3 allocations plus the registry.

    Every regime/budget combination is generated before any probe or
    development NLL is inspected; final-budget choices therefore cannot depend
    on outcomes. Comparators are reused from the frozen Stage 2B and Stage 2C
    registries by hash.
    """

    allocations_dir.mkdir(parents=True, exist_ok=True)
    reused_stage2b = _reused_entries(
        stage2b_registry, stage2b_allocations_dir, "entries", "stage2b_frozen"
    )
    reused_stage2c = _reused_entries(
        stage2c_registry, stage2c_allocations_dir, "new_entries", "stage2c_frozen"
    )
    reused_by_key = {
        (entry["method"], entry["regime"], entry["budget_fraction"]): entry
        for entry in reused_stage2b + reused_stage2c
    }
    comparator_methods = (
        REUSED_STAGE2B_METHODS
        + tuple(f"random_seed{seed}" for seed in RANDOM_ALLOCATION_SEEDS)
        + REUSED_STAGE2C_METHODS
    )
    new_entries: list[dict[str, Any]] = []
    for regime, base_bits in BASE_BITS_BY_REGIME.items():
        delta_bytes = memory.delta_protection_bytes(base_bits)
        for fraction in PROTECTION_FRACTIONS:
            budget = memory.protection_budget_bytes(base_bits, fraction)
            solution = solve_measured_damage_robust(
                delta_nll, regime, delta_bytes, budget
            )
            record = build_measured_allocation_record(
                regime, fraction, solution, scores, memory, damage_record, delta_nll
            )
            # Optimality sanity check: no reused comparator at the same
            # regime/budget may achieve a lower predicted maximum delta NLL
            # than the MILP optimum.
            for method in comparator_methods:
                entry = reused_by_key.get((method, regime, fraction))
                if entry is None:
                    raise RuntimeError(
                        f"Frozen comparator {method} is missing at "
                        f"{regime}/{fraction}"
                    )
                base = (
                    stage2c_allocations_dir
                    if entry["source"] == "stage2c_frozen"
                    else stage2b_allocations_dir
                )
                comparator = read_json(base / entry["file"])
                comparator_bits = np.asarray(
                    comparator["expert_bits"], dtype=np.int64
                )
                comparator_max = float(
                    predicted_domain_delta_nll(comparator_bits, delta_nll).max()
                )
                if comparator_max < solution.objective_value - 1e-6:
                    raise RuntimeError(
                        f"{method} at {regime}/{fraction} achieves predicted max "
                        "delta NLL below the Measured-Damage MILP optimum; the "
                        "solve cannot be optimal"
                    )
            file_name = allocation_file_name(MEASURED_DAMAGE_METHOD, regime, fraction)
            atomic_write_json(allocations_dir / file_name, record)
            new_entries.append(
                {
                    "file": file_name,
                    "source": "stage3_new",
                    "method": record["method"],
                    "method_label": record["method_label"],
                    "method_kind": record["method_kind"],
                    "regime": regime,
                    "budget_fraction": fraction,
                    "allocation_sha256": record["allocation_sha256"],
                    "file_sha256": file_sha256(allocations_dir / file_name),
                }
            )

    registry: dict[str, Any] = {
        "schema": STAGE3_REGISTRY_SCHEMA,
        "stage": STAGE3_STAGE,
        "frozen": True,
        "frozen_before_any_probe_or_heldout_nll": True,
        "all_regime_budget_allocations_frozen": True,
        "development_seed": STAGE3_DEVELOPMENT_SEED,
        "final_seed": STAGE3_FINAL_SEED,
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
        "profile_bits": list(STAGE3_PROFILE_BITS),
        "damage_sha256": damage_record["damage_sha256"],
        "calibration_fingerprint": scores.calibration_fingerprint,
        "score_hashes": scores.score_hashes,
        "stage2b_registry_sha256": stage2b_registry["registry_sha256"],
        "stage2c_registry_sha256": stage2c_registry["registry_sha256"],
        "comparators_reused_from": {
            "stage2b": str(stage2b_allocations_dir),
            "stage2c": str(stage2c_allocations_dir),
        },
        "new_entries": new_entries,
        "reused_stage2b_entries": reused_stage2b,
        "reused_stage2c_entries": reused_stage2c,
    }
    registry["registry_sha256"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    registry["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(allocations_dir / "allocation_registry.json", registry)
    return registry


def load_frozen_stage3_registry(
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    stage2c_allocations_dir: Path,
) -> dict[str, Any]:
    """Load the Stage 3 registry and verify every new and reused artifact."""

    registry = read_json(allocations_dir / "allocation_registry.json")
    if registry.get("schema") != STAGE3_REGISTRY_SCHEMA:
        raise RuntimeError("Unexpected Stage 3 registry schema")
    if registry.get("frozen") is not True:
        raise RuntimeError("The Stage 3 allocation registry is not marked frozen")
    expected = canonical_sha256(
        {
            key: value
            for key, value in registry.items()
            if key not in ("registry_sha256", "created_at_utc")
        }
    )
    if registry.get("registry_sha256") != expected:
        raise RuntimeError("Stage 3 allocation_registry.json failed its integrity hash")
    groups = (
        ("new_entries", allocations_dir),
        ("reused_stage2b_entries", stage2b_allocations_dir),
        ("reused_stage2c_entries", stage2c_allocations_dir),
    )
    for key, base in groups:
        for entry in registry[key]:
            path = base / entry["file"]
            if not path.is_file():
                raise RuntimeError(f"Frozen allocation {entry['file']} is missing")
            if file_sha256(path) != entry["file_sha256"]:
                raise RuntimeError(
                    f"Frozen allocation {entry['file']} file hash mismatch"
                )
            record = read_json(path)
            verify_allocation_record(record)
            if record["allocation_sha256"] != entry["allocation_sha256"]:
                raise RuntimeError(
                    f"Frozen allocation {entry['file']} content hash mismatch"
                )
    return registry


def stage3_registry_entries(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return (
        list(registry["new_entries"])
        + list(registry["reused_stage2b_entries"])
        + list(registry["reused_stage2c_entries"])
    )


def load_stage3_allocation(
    registry_entry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    stage2c_allocations_dir: Path,
) -> dict[str, Any]:
    """Load one allocation record from its frozen home directory."""

    base_by_source = {
        "stage3_new": allocations_dir,
        "stage2b_frozen": stage2b_allocations_dir,
        "stage2c_frozen": stage2c_allocations_dir,
    }
    base = base_by_source[registry_entry["source"]]
    record = read_json(base / registry_entry["file"])
    verify_allocation_record(record)
    if record["allocation_sha256"] != registry_entry["allocation_sha256"]:
        raise RuntimeError(
            f"Allocation {registry_entry['file']} does not match the Stage 3 registry"
        )
    return record


def write_stage3_allocation_summary(
    registry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    stage2c_allocations_dir: Path,
    delta_nll: np.ndarray,
    output_dir: Path,
) -> Path:
    """Optimization-space allocation table written before any evaluation."""

    rows: list[dict[str, Any]] = []
    for entry in stage3_registry_entries(registry):
        record = load_stage3_allocation(
            entry, allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir
        )
        if record["method_kind"] == "uniform_reference":
            continue
        predicted = predicted_domain_delta_nll(
            np.asarray(record["expert_bits"], dtype=np.int64), delta_nll
        )
        rows.append(
            {
                "source": entry["source"],
                "method": record["method"],
                "method_label": record["method_label"],
                "regime": record["regime"],
                "budget_fraction": record["budget_fraction"],
                "budget_bytes": record["budget_bytes"],
                "used_protection_bytes": record["used_protection_bytes"],
                "protected_expert_count": record["protected_expert_count"],
                "effective_bits_per_weight": record["effective_bits_per_weight"],
                **{
                    f"predicted_delta_{domain}": float(predicted[index])
                    for index, domain in enumerate(STAGE2B_DOMAINS)
                },
                "predicted_worst_delta": float(predicted.max()),
                "allocation_sha256": record["allocation_sha256"],
                "file": entry["file"],
            }
        )
    path = output_dir / "allocation_summary.csv"
    write_csv(
        path,
        rows,
        [
            "source", "method", "method_label", "regime", "budget_fraction",
            "budget_bytes", "used_protection_bytes", "protected_expert_count",
            "effective_bits_per_weight",
            *[f"predicted_delta_{domain}" for domain in STAGE2B_DOMAINS],
            "predicted_worst_delta", "allocation_sha256", "file",
        ],
    )
    return path
