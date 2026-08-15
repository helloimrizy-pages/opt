"""Stage 2C Fragility-Robust optimization, frozen allocations, and registry.

The PRIMARY Stage 2C method minimizes the largest fragility-weighted residual
risk across domains:

    minimize z
    subject to  q_norm[d] * (1 - Coverage_d(x)) <= z   for every domain
                sum_{l,e} DeltaM[l,e] * x[l,e] <= Budget
                x[l,e] binary

with fixed calibration fragility constants ``q_norm`` and the frozen Stage 2B
specialist coverage. The objective is exactly this linear program: no mean-risk
penalty, no tuned lambda, no additional term. All eight regime/budget
allocations are generated and frozen before any development NLL is inspected.
Comparator allocations are never regenerated; the frozen Stage 2B files are
reused by hash.
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
from .fragility import STAGE2C_STAGE, Stage2BScoreArtifacts, fragility_vector
from .io_utils import atomic_write_json, json_safe, read_json, write_csv
from .protection_allocations import (
    allocation_deterministic_content,
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

STAGE2C_ALLOCATION_SCHEMA = "stage2c_allocation_v1"
STAGE2C_REGISTRY_SCHEMA = "stage2c_allocation_registry_v1"
FRAGILITY_ROBUST_METHOD = "fragility_robust"
FRAGILITY_ROBUST_LABEL = "Fragility-Robust"
STAGE2C_DEVELOPMENT_SEED = 45
STAGE2C_FINAL_SEED = 44
# Frozen Stage 2B comparators reused by hash (plus the five random seeds).
REUSED_COMPARATOR_METHODS = (
    "robust_functional",
    "robust_routing",
    "average_specialization",
    "global_importance",
    "general_only",
    "math_only",
    "coding_only",
    "reasoning_only",
)


def predicted_residual_risk(q_norm: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """``ResidualRisk_d = q_norm[d] * (1 - Coverage_d)`` for fixed fragility."""

    fragility = np.asarray(q_norm, dtype=np.float64)
    covered = np.asarray(coverage, dtype=np.float64)
    if fragility.shape != (len(STAGE2B_DOMAINS),) or covered.shape != fragility.shape:
        raise ValueError("Fragility and coverage must both cover the four domains")
    if np.any(fragility < 0) or not np.all(np.isfinite(fragility)):
        raise ValueError("Normalized fragility must be finite and nonnegative")
    if np.any(covered < -1e-9) or np.any(covered > 1.0 + 1e-9):
        raise ValueError("Coverage must lie in [0, 1]")
    return fragility * (1.0 - covered)


def solve_fragility_robust(
    specialization: np.ndarray,
    q_norm: np.ndarray,
    delta_bytes: np.ndarray,
    budget_bytes: int,
    mip_rel_gap: float = 0.0,
) -> SolverResult:
    """Minimize the maximum fragility-weighted residual risk under the budget."""

    scores = np.asarray(specialization, dtype=np.float64)
    fragility = np.asarray(q_norm, dtype=np.float64)
    delta = np.asarray(delta_bytes, dtype=np.float64)
    if delta.ndim != 2 or np.any(delta <= 0):
        raise ValueError("Every protection increment must be positive")
    if budget_bytes < 0:
        raise ValueError("The protection budget cannot be negative")
    if scores.shape[:2] != delta.shape or scores.shape[2] != len(STAGE2B_DOMAINS):
        raise ValueError("Specialization scores and memory increments do not align")
    if fragility.shape != (len(STAGE2B_DOMAINS),):
        raise ValueError("Fragility must contain one value per domain")
    if np.any(fragility < 0) or not np.all(np.isfinite(fragility)):
        raise ValueError("Fragility values must be finite and nonnegative")
    if not np.any(fragility > 0):
        raise ValueError(
            "All-zero fragility marks the regime invalid; it must not be solved"
        )
    n = delta.size
    c = np.zeros(n + 1, dtype=np.float64)
    c[-1] = 1.0
    # q_d * (1 - S_d . x) <= z  <=>  q_d * S_d . x + z >= q_d
    weighted = scores.reshape(n, len(STAGE2B_DOMAINS)).T * fragility[:, None]
    risk_rows = np.concatenate(
        [weighted, np.ones((len(STAGE2B_DOMAINS), 1))], axis=1
    )
    memory_row = np.concatenate([delta.reshape(1, n), np.zeros((1, 1))], axis=1)
    constraints = [
        LinearConstraint(risk_rows, lb=fragility, ub=np.inf),
        LinearConstraint(memory_row, lb=-np.inf, ub=float(budget_bytes)),
    ]
    integrality = np.concatenate([np.ones(n), np.zeros(1)])
    bounds = Bounds(
        lb=np.zeros(n + 1),
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
        raise RuntimeError(f"Fragility-Robust MILP failed: {result.message}")
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
    coverage = specialist_coverage(scores, x)
    residual = predicted_residual_risk(fragility, coverage)
    achieved_z = float(residual.max())
    if abs(achieved_z - solver_z) > 1e-5:
        raise RuntimeError(
            f"Fragility-Robust MILP z={solver_z} disagrees with the recomputed "
            f"maximum residual risk {achieved_z} beyond solver tolerance"
        )
    metadata: dict[str, Any] = {
        "solver": "scipy.optimize.milp",
        "backend": "HiGHS",
        "scipy_version": scipy.__version__,
        "problem": "min_max_fragility_weighted_residual_risk",
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
        "objective_max_residual_risk": achieved_z,
        "solver_reported_z": solver_z,
        "fragility_q_norm_by_domain": {
            domain: float(fragility[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "coverage_by_domain": {
            domain: float(coverage[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "residual_risk_by_domain": {
            domain: float(residual[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "minimum_coverage": float(coverage.min()),
        "risk_constraint_residuals": (achieved_z - residual).tolist(),
    }
    return SolverResult(protected=x, objective_value=achieved_z, metadata=metadata)


def build_fragility_allocation_record(
    regime: str,
    fraction: float,
    solution: SolverResult,
    scores: Stage2BScoreArtifacts,
    memory: ExpertMemoryMatrix,
    fragility_record: Mapping[str, Any],
) -> dict[str, Any]:
    """One frozen Fragility-Robust allocation with full provenance."""

    base_bits = BASE_BITS_BY_REGIME[regime]
    delta = memory.delta_protection_bytes(base_bits)
    budget = memory.protection_budget_bytes(base_bits, fraction)
    x = solution.protected
    bits = bits_matrix_for_allocation(x, base_bits)
    used = int((delta * x.astype(np.int64)).sum())
    if used > budget:
        raise RuntimeError(
            f"Fragility-Robust allocation {regime}/{fraction} exceeds its budget"
        )
    q_norm = fragility_vector(fragility_record, regime)
    functional_coverage = specialist_coverage(scores.functional_specialization, x)
    routing_coverage = specialist_coverage(scores.routing_specialization, x)
    residual = predicted_residual_risk(q_norm, functional_coverage)
    record: dict[str, Any] = {
        "schema": STAGE2C_ALLOCATION_SCHEMA,
        "stage": STAGE2C_STAGE,
        "method": FRAGILITY_ROBUST_METHOD,
        "method_label": FRAGILITY_ROBUST_LABEL,
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
        "fragility_q_norm": {
            domain: float(q_norm[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "predicted_residual_risk": {
            domain: float(residual[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "predicted_max_residual_risk": float(residual.max()),
        "fragility_sha256": fragility_record["fragility_sha256"],
        "calibration_fingerprint": scores.calibration_fingerprint,
        "score_hashes": scores.score_hashes,
        "solver_metadata": solution.metadata,
    }
    # Serialize-safe first so the recorded hash always matches the JSON file
    # (non-finite solver diagnostics become null in both places).
    record = json_safe(record)
    record["allocation_sha256"] = canonical_sha256(record)
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def _reused_registry_entries(
    stage2b_registry: Mapping[str, Any], stage2b_allocations_dir: Path
) -> list[dict[str, Any]]:
    """Reuse every frozen Stage 2B allocation by hash, verifying files on disk."""

    entries: list[dict[str, Any]] = []
    for entry in stage2b_registry["entries"]:
        path = stage2b_allocations_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"Frozen Stage 2B allocation {entry['file']} is missing")
        if file_sha256(path) != entry["file_sha256"]:
            raise RuntimeError(
                f"Frozen Stage 2B allocation {entry['file']} does not match its "
                "registry file hash"
            )
        record = read_json(path)
        verify_allocation_record(record)
        if record["allocation_sha256"] != entry["allocation_sha256"]:
            raise RuntimeError(
                f"Frozen Stage 2B allocation {entry['file']} content hash mismatch"
            )
        entries.append(
            {
                "file": entry["file"],
                "source": "stage2b_frozen",
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


def generate_fragility_robust_allocations(
    scores: Stage2BScoreArtifacts,
    memory: ExpertMemoryMatrix,
    fragility_record: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_registry: Mapping[str, Any],
    stage2b_allocations_dir: Path,
) -> dict[str, Any]:
    """Solve and freeze all eight Fragility-Robust allocations plus the registry.

    Every regime/budget combination is generated before any development NLL is
    inspected; final-budget choices therefore cannot depend on development
    outcomes. Comparators are reused from the frozen Stage 2B registry by hash.
    """

    allocations_dir.mkdir(parents=True, exist_ok=True)
    reused = _reused_registry_entries(stage2b_registry, stage2b_allocations_dir)
    reused_by_key = {
        (entry["method"], entry["regime"], entry["budget_fraction"]): entry
        for entry in reused
    }
    valid_regimes = [
        regime
        for regime in BASE_BITS_BY_REGIME
        if fragility_record["regimes"][regime]["regime_valid"]
    ]
    if not valid_regimes:
        raise RuntimeError(
            "Both precision regimes have all-zero calibration fragility; no "
            "Stage 2C optimization is possible"
        )
    new_entries: list[dict[str, Any]] = []
    for regime, base_bits in BASE_BITS_BY_REGIME.items():
        if regime not in valid_regimes:
            continue
        q_norm = fragility_vector(fragility_record, regime)
        delta = memory.delta_protection_bytes(base_bits)
        for fraction in PROTECTION_FRACTIONS:
            budget = memory.protection_budget_bytes(base_bits, fraction)
            solution = solve_fragility_robust(
                scores.functional_specialization, q_norm, delta, budget
            )
            record = build_fragility_allocation_record(
                regime, fraction, solution, scores, memory, fragility_record
            )
            # Optimality sanity check: no reused comparator at the same
            # regime/budget may achieve a lower maximum residual risk than the
            # MILP optimum.
            for method in REUSED_COMPARATOR_METHODS + tuple(
                f"random_seed{seed}" for seed in RANDOM_ALLOCATION_SEEDS
            ):
                entry = reused_by_key.get((method, regime, fraction))
                if entry is None:
                    raise RuntimeError(
                        f"Frozen Stage 2B comparator {method} is missing at "
                        f"{regime}/{fraction}"
                    )
                comparator = read_json(stage2b_allocations_dir / entry["file"])
                comparator_coverage = np.asarray(
                    [
                        comparator["functional_specialist_coverage"][domain]
                        for domain in STAGE2B_DOMAINS
                    ]
                )
                comparator_max_risk = float(
                    predicted_residual_risk(q_norm, comparator_coverage).max()
                )
                if comparator_max_risk < solution.objective_value - 1e-6:
                    raise RuntimeError(
                        f"{method} at {regime}/{fraction} achieves max residual "
                        "risk below the Fragility-Robust MILP optimum; the solve "
                        "cannot be optimal"
                    )
            file_name = allocation_file_name(FRAGILITY_ROBUST_METHOD, regime, fraction)
            atomic_write_json(allocations_dir / file_name, record)
            new_entries.append(
                {
                    "file": file_name,
                    "source": "stage2c_new",
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
        "schema": STAGE2C_REGISTRY_SCHEMA,
        "stage": STAGE2C_STAGE,
        "frozen": True,
        "frozen_before_any_heldout_nll": True,
        "all_valid_regime_budget_allocations_frozen": True,
        "valid_regimes": valid_regimes,
        "invalid_regimes": [
            regime for regime in BASE_BITS_BY_REGIME if regime not in valid_regimes
        ],
        "development_seed": STAGE2C_DEVELOPMENT_SEED,
        "final_seed": STAGE2C_FINAL_SEED,
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
        "fragility_sha256": fragility_record["fragility_sha256"],
        "calibration_fingerprint": scores.calibration_fingerprint,
        "score_hashes": scores.score_hashes,
        "stage2b_registry_sha256": stage2b_registry["registry_sha256"],
        "comparators_reused_from": str(stage2b_allocations_dir),
        "new_entries": new_entries,
        "reused_entries": reused,
    }
    registry["registry_sha256"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    registry["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(allocations_dir / "allocation_registry.json", registry)
    return registry


def load_frozen_stage2c_registry(
    allocations_dir: Path, stage2b_allocations_dir: Path
) -> dict[str, Any]:
    """Load the Stage 2C registry and verify every new and reused artifact."""

    registry = read_json(allocations_dir / "allocation_registry.json")
    if registry.get("schema") != STAGE2C_REGISTRY_SCHEMA:
        raise RuntimeError("Unexpected Stage 2C registry schema")
    if registry.get("frozen") is not True:
        raise RuntimeError("The Stage 2C allocation registry is not marked frozen")
    expected = canonical_sha256(
        {
            key: value
            for key, value in registry.items()
            if key not in ("registry_sha256", "created_at_utc")
        }
    )
    if registry.get("registry_sha256") != expected:
        raise RuntimeError("Stage 2C allocation_registry.json failed its integrity hash")
    for entry in registry["new_entries"]:
        path = allocations_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"Frozen Stage 2C allocation {entry['file']} is missing")
        if file_sha256(path) != entry["file_sha256"]:
            raise RuntimeError(
                f"Frozen Stage 2C allocation {entry['file']} file hash mismatch"
            )
        record = read_json(path)
        verify_allocation_record(record)
        if record["allocation_sha256"] != entry["allocation_sha256"]:
            raise RuntimeError(
                f"Frozen Stage 2C allocation {entry['file']} content hash mismatch"
            )
    for entry in registry["reused_entries"]:
        path = stage2b_allocations_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"Reused Stage 2B allocation {entry['file']} is missing")
        if file_sha256(path) != entry["file_sha256"]:
            raise RuntimeError(
                f"Reused Stage 2B allocation {entry['file']} was modified"
            )
    return registry


def load_stage2c_allocation(
    registry_entry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
) -> dict[str, Any]:
    """Load one allocation record from its frozen home directory."""

    base = (
        allocations_dir
        if registry_entry["source"] == "stage2c_new"
        else stage2b_allocations_dir
    )
    record = read_json(base / registry_entry["file"])
    verify_allocation_record(record)
    if record["allocation_sha256"] != registry_entry["allocation_sha256"]:
        raise RuntimeError(
            f"Allocation {registry_entry['file']} does not match the Stage 2C registry"
        )
    return record


def write_stage2c_allocation_summary(
    registry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    output_dir: Path,
) -> Path:
    """Optimization-space allocation table written before any evaluation."""

    rows: list[dict[str, Any]] = []
    for entry in list(registry["new_entries"]) + list(registry["reused_entries"]):
        record = load_stage2c_allocation(entry, allocations_dir, stage2b_allocations_dir)
        if record["method_kind"] == "uniform_reference":
            continue
        coverage = record["functional_specialist_coverage"]
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
                **{f"coverage_{domain}": coverage[domain] for domain in STAGE2B_DOMAINS},
                "coverage_min": record["functional_specialist_coverage_min"],
                "coverage_mean": record["functional_specialist_coverage_mean"],
                "predicted_max_residual_risk": record.get(
                    "predicted_max_residual_risk"
                ),
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
            *[f"coverage_{domain}" for domain in STAGE2B_DOMAINS],
            "coverage_min", "coverage_mean", "predicted_max_residual_risk",
            "allocation_sha256", "file",
        ],
    )
    return path


def allocation_registry_deterministic_sha(registry: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in registry.items()
            if key not in ("registry_sha256", "created_at_utc")
        }
    )
