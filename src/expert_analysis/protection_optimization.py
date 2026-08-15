"""Stage 2B protection optimization: exact memory accounting and MILP solvers.

Every optimizer here selects a binary protection matrix ``x[l,e]`` under one
shared incremental-memory constraint. The objectives use only the frozen
calibration scores; no quantization loss, surrogate, or held-out NLL enters any
objective. All mixed-integer problems are solved with ``scipy.optimize.milp``
(HiGHS); greedy selection is used only for the score-independent random
baseline, which is preregistered as random.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from .quantization import projected_expert_storage
from .specialist_preservation import (
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
    specialist_coverage,
)

PROTECTED_BITS = 8
BASE_BITS_BY_REGIME = {"4to8": 4, "3to8": 3}
PROTECTION_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RANDOM_ALLOCATION_SEEDS = (1001, 1002, 1003, 1004, 1005)
GROUP_SIZE = 128
MEMORY_BIT_WIDTHS = (3, 4, 8, 16)


def expert_tensor_shapes_from_config(config: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Per-expert weight-matrix shapes for OLMoE's fused expert layout.

    ``down_proj`` per-expert slice is ``[hidden, intermediate]`` and the fused
    ``gate_up_proj`` slice is ``[2 * intermediate, hidden]``. The evaluation
    runner independently re-verifies these against the live model layout.
    """

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    if hidden < 1 or intermediate < 1:
        raise ValueError("Architecture config has invalid hidden/intermediate sizes")
    return [(hidden, intermediate), (2 * intermediate, hidden)]


@dataclass
class ExpertMemoryMatrix:
    """Exact projected per-expert storage bytes at every relevant bit width."""

    bytes_by_bits: dict[int, np.ndarray]
    weight_count: np.ndarray
    group_count: np.ndarray
    tensor_shapes: list[list[tuple[int, int]]]
    group_size: int

    @property
    def num_layers(self) -> int:
        return int(self.weight_count.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.weight_count.shape[1])

    def delta_protection_bytes(self, base_bits: int) -> np.ndarray:
        """``DeltaM_e = M_e(8) - M_e(base)`` in exact bytes."""

        if base_bits not in self.bytes_by_bits:
            raise KeyError(f"Base bit width {base_bits} is not accounted")
        delta = self.bytes_by_bits[PROTECTED_BITS] - self.bytes_by_bits[base_bits]
        if np.any(delta <= 0):
            raise RuntimeError("Protection must strictly increase expert storage")
        return delta

    def total_increment_bytes(self, base_bits: int) -> int:
        return int(self.delta_protection_bytes(base_bits).sum())

    def protection_budget_bytes(self, base_bits: int, fraction: float) -> int:
        if not 0.0 < fraction < 1.0:
            raise ValueError("Protection fraction must be in (0, 1)")
        return int(np.floor(fraction * self.total_increment_bytes(base_bits)))

    def allocation_bytes(self, bits_matrix: np.ndarray) -> int:
        """Total projected expert-weight bytes for a per-expert bit assignment."""

        bits = np.asarray(bits_matrix)
        if bits.shape != self.weight_count.shape:
            raise ValueError("Bit-assignment matrix has the wrong shape")
        total = 0
        for width in np.unique(bits):
            width = int(width)
            if width not in self.bytes_by_bits:
                raise ValueError(f"Bit width {width} is not accounted")
            total += int(self.bytes_by_bits[width][bits == width].sum())
        return total

    def effective_bits_per_weight(self, bits_matrix: np.ndarray) -> float:
        return self.allocation_bytes(bits_matrix) * 8.0 / float(self.weight_count.sum())


def build_expert_memory_matrix(
    per_layer_tensor_shapes: Sequence[Sequence[Sequence[int]]],
    group_size: int = GROUP_SIZE,
    num_experts: int = NUM_EXPERTS,
) -> ExpertMemoryMatrix:
    """Compute ``M_e(b)`` for every layer/expert with the Stage-1 accounting.

    ``per_layer_tensor_shapes`` supplies the per-expert weight shapes of every
    MoE layer, so unequal layers would be accounted exactly rather than assumed
    identical.
    """

    num_layers = len(per_layer_tensor_shapes)
    if num_layers < 1 or num_experts < 1:
        raise ValueError("At least one layer and expert are required")
    bytes_by_bits = {
        bits: np.zeros((num_layers, num_experts), dtype=np.int64)
        for bits in MEMORY_BIT_WIDTHS
    }
    weight_count = np.zeros((num_layers, num_experts), dtype=np.int64)
    group_count = np.zeros((num_layers, num_experts), dtype=np.int64)
    shapes: list[list[tuple[int, int]]] = []
    for layer, layer_shapes in enumerate(per_layer_tensor_shapes):
        normalized = [tuple(int(v) for v in shape) for shape in layer_shapes]
        shapes.append([tuple(shape) for shape in normalized])
        for expert in range(num_experts):
            for bits in MEMORY_BIT_WIDTHS:
                accounting = projected_expert_storage(normalized, bits, group_size)
                bytes_by_bits[bits][layer, expert] = accounting["projected_bytes"]
                if bits == MEMORY_BIT_WIDTHS[0]:
                    weight_count[layer, expert] = accounting["weight_count"]
                group_count[layer, expert] = max(
                    group_count[layer, expert], accounting["number_of_groups"]
                )
    return ExpertMemoryMatrix(
        bytes_by_bits=bytes_by_bits,
        weight_count=weight_count,
        group_count=group_count,
        tensor_shapes=shapes,
        group_size=group_size,
    )


@dataclass
class SolverResult:
    """A validated binary protection matrix plus complete solver metadata."""

    protected: np.ndarray
    objective_value: float
    metadata: dict[str, Any]


def _validate_inputs(delta_bytes: np.ndarray, budget_bytes: int) -> None:
    if delta_bytes.ndim != 2:
        raise ValueError("delta_bytes must have shape [layer, expert]")
    if np.any(delta_bytes <= 0):
        raise ValueError("Every protection increment must be positive")
    if budget_bytes < 0:
        raise ValueError("The protection budget cannot be negative")


def _finalize_binary(
    solution: np.ndarray, delta_bytes: np.ndarray, budget_bytes: int
) -> np.ndarray:
    rounded = np.round(solution)
    if np.max(np.abs(solution - rounded)) > 1e-6:
        raise RuntimeError("MILP returned a non-integral protection variable")
    x = rounded.astype(np.uint8).reshape(delta_bytes.shape)
    used = int((delta_bytes * x).sum())
    if used > budget_bytes:
        raise RuntimeError(
            f"MILP solution uses {used} bytes, exceeding the {budget_bytes}-byte budget"
        )
    return x

def _solver_metadata(
    result: Any,
    started: float,
    delta_bytes: np.ndarray,
    budget_bytes: int,
    x: np.ndarray,
    objective_sense: str,
    problem: str,
) -> dict[str, Any]:
    used = int((delta_bytes * x).sum())
    return {
        "solver": "scipy.optimize.milp",
        "backend": "HiGHS",
        "scipy_version": scipy.__version__,
        "problem": problem,
        "objective_sense": objective_sense,
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
    }


def solve_max_min_coverage(
    specialization: np.ndarray,
    delta_bytes: np.ndarray,
    budget_bytes: int,
    mip_rel_gap: float = 0.0,
) -> SolverResult:
    """Maximize ``z`` with ``Coverage_d(x) >= z`` for all domains under the budget."""

    scores = np.asarray(specialization, dtype=np.float64)
    delta = np.asarray(delta_bytes, dtype=np.float64)
    _validate_inputs(delta, budget_bytes)
    if scores.shape[:2] != delta.shape or scores.shape[2] != len(STAGE2B_DOMAINS):
        raise ValueError("Specialization scores and memory increments do not align")
    n = delta.size
    c = np.zeros(n + 1, dtype=np.float64)
    c[-1] = -1.0
    coverage_rows = np.concatenate(
        [
            scores.reshape(n, len(STAGE2B_DOMAINS)).T,
            -np.ones((len(STAGE2B_DOMAINS), 1)),
        ],
        axis=1,
    )
    memory_row = np.concatenate([delta.reshape(1, n), np.zeros((1, 1))], axis=1)
    constraints = [
        LinearConstraint(coverage_rows, lb=0.0, ub=np.inf),
        LinearConstraint(memory_row, lb=-np.inf, ub=float(budget_bytes)),
    ]
    integrality = np.concatenate([np.ones(n), np.zeros(1)])
    bounds = Bounds(lb=np.zeros(n + 1), ub=np.ones(n + 1))
    started = time.monotonic()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"mip_rel_gap": mip_rel_gap},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Max-min coverage MILP failed: {result.message}")
    x = _finalize_binary(result.x[:n], delta, budget_bytes)
    solver_z = float(result.x[-1])
    coverage = specialist_coverage(scores, x)
    # The exact achieved objective is the recomputed minimum coverage of the
    # binary solution; the solver's continuous z must agree within its MIP
    # feasibility tolerance.
    achieved_z = float(coverage.min())
    if abs(achieved_z - solver_z) > 1e-5:
        raise RuntimeError(
            f"Max-min MILP z={solver_z} disagrees with the recomputed minimum "
            f"coverage {achieved_z} beyond solver tolerance"
        )
    metadata = _solver_metadata(
        result, started, delta, budget_bytes, x, "maximize", "max_min_specialist_coverage"
    )
    metadata["objective_z"] = achieved_z
    metadata["solver_reported_z"] = solver_z
    metadata["coverage_by_domain"] = {
        domain: float(coverage[index]) for index, domain in enumerate(STAGE2B_DOMAINS)
    }
    metadata["minimum_coverage"] = achieved_z
    metadata["coverage_constraint_residuals"] = (coverage - achieved_z).tolist()
    return SolverResult(protected=x, objective_value=achieved_z, metadata=metadata)


def solve_weighted_selection(
    weights: np.ndarray,
    delta_bytes: np.ndarray,
    budget_bytes: int,
    problem: str,
    mip_rel_gap: float = 0.0,
) -> SolverResult:
    """Maximize ``sum w[l,e] x[l,e]`` under the shared memory budget."""

    profit = np.asarray(weights, dtype=np.float64)
    delta = np.asarray(delta_bytes, dtype=np.float64)
    _validate_inputs(delta, budget_bytes)
    if profit.shape != delta.shape:
        raise ValueError("Objective weights and memory increments do not align")
    if not np.all(np.isfinite(profit)):
        raise ValueError("Objective weights contain non-finite values")
    n = delta.size
    constraints = [
        LinearConstraint(delta.reshape(1, n), lb=-np.inf, ub=float(budget_bytes))
    ]
    started = time.monotonic()
    result = milp(
        c=-profit.reshape(n),
        constraints=constraints,
        integrality=np.ones(n),
        bounds=Bounds(lb=np.zeros(n), ub=np.ones(n)),
        options={"mip_rel_gap": mip_rel_gap},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Weighted-selection MILP failed for {problem}: {result.message}")
    x = _finalize_binary(result.x, delta, budget_bytes)
    objective = float((profit * x).sum())
    metadata = _solver_metadata(
        result, started, delta, budget_bytes, x, "maximize", problem
    )
    metadata["objective_value"] = objective
    return SolverResult(protected=x, objective_value=objective, metadata=metadata)


def random_allocation(
    delta_bytes: np.ndarray, budget_bytes: int, seed: int
) -> SolverResult:
    """Score-independent deterministic random feasible protection allocation.

    Experts are visited in one seeded permutation and included whenever the
    remaining budget allows. The procedure never reads any expert score.
    """

    delta = np.asarray(delta_bytes, dtype=np.int64)
    _validate_inputs(delta, budget_bytes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(delta.size)
    flat = delta.reshape(-1)
    x = np.zeros(delta.size, dtype=np.uint8)
    remaining = int(budget_bytes)
    for index in order:
        cost = int(flat[index])
        if cost <= remaining:
            x[index] = 1
            remaining -= cost
    protected = x.reshape(delta.shape)
    used = int((delta * protected).sum())
    metadata = {
        "solver": "seeded_random_permutation_fill",
        "score_independent": True,
        "seed": int(seed),
        "budget_bytes": int(budget_bytes),
        "used_protection_bytes": used,
        "budget_utilization": used / budget_bytes if budget_bytes > 0 else 0.0,
        "memory_constraint_residual_bytes": int(budget_bytes - used),
        "protected_expert_count": int(protected.sum()),
    }
    return SolverResult(
        protected=protected, objective_value=float("nan"), metadata=metadata
    )


def bits_matrix_for_allocation(
    protected: np.ndarray, base_bits: int, protected_bits: int = PROTECTED_BITS
) -> np.ndarray:
    """Per-expert bit assignment: base precision or protected 8-bit only."""

    x = np.asarray(protected)
    if not np.all(np.isin(x, (0, 1))):
        raise ValueError("Protection matrix must be binary")
    if base_bits not in BASE_BITS_BY_REGIME.values():
        raise ValueError(f"Base precision {base_bits} is not preregistered")
    if protected_bits != PROTECTED_BITS:
        raise ValueError("Protected precision is fixed at 8-bit")
    bits = np.full(x.shape, base_bits, dtype=np.int64)
    bits[x == 1] = protected_bits
    return bits


def uniform_bits_matrix(bits: int, num_layers: int = NUM_MOE_LAYERS, num_experts: int = NUM_EXPERTS) -> np.ndarray:
    return np.full((num_layers, num_experts), bits, dtype=np.int64)
