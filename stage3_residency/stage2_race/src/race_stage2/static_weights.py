"""Calibration-only learning of a fixed RACE adviser-weight vector.

The objective is a convex logistic surrogate of the pairwise future-use ranking loss
of the *combined* score, minimized over the probability simplex by projected
gradient descent with deterministic Armijo backtracking. It never touches an
evaluation sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .advisers import uniform_weights


TAU = 0.1
ITERATIONS = 300
INITIAL_STEP = 1.0
ARMIJO_C = 1e-4
MAX_BACKTRACKS = 40


@dataclass(frozen=True)
class PairDataset:
    """Flattened comparable-pair differences with per-example normalized weights."""

    differences: np.ndarray
    weights: np.ndarray
    examples: int

    @property
    def pairs(self) -> int:
        return int(self.differences.shape[0])

    @property
    def advisers(self) -> int:
        return int(self.differences.shape[1])


def build_pair_dataset(
    normalized: Sequence[np.ndarray], distances: Sequence[np.ndarray]
) -> PairDataset:
    """Stack every comparable candidate pair of every example into one matrix."""

    if len(normalized) != len(distances):
        raise ValueError("Example score and distance lists must align")
    blocks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used = 0
    advisers: int | None = None
    for scores, values in zip(normalized, distances):
        matrix = np.asarray(scores, dtype=np.float64)
        target = np.asarray(values, dtype=np.int64)
        if matrix.ndim != 2:
            raise ValueError("Example adviser scores must be [adviser, candidate]")
        if advisers is None:
            advisers = int(matrix.shape[0])
        elif int(matrix.shape[0]) != advisers:
            raise ValueError("Examples disagree on the adviser-pool size")
        if matrix.shape[1] != target.shape[0]:
            raise ValueError("Example candidate counts differ")
        first, second = np.nonzero(target[:, None] < target[None, :])
        if first.size == 0:
            continue
        difference = (matrix[:, first] - matrix[:, second]).T
        blocks.append(difference)
        weights.append(np.full(first.size, 1.0 / float(first.size), dtype=np.float64))
        used += 1
    if not blocks:
        raise ValueError("No comparable calibration pairs were collected")
    return PairDataset(np.concatenate(blocks), np.concatenate(weights), used)


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""

    values = np.asarray(vector, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    candidates = np.nonzero(ordered * indices > (cumulative - 1.0))[0]
    rho = int(candidates[-1]) if candidates.size else 0
    theta = (cumulative[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)
    total = projected.sum()
    if total <= 0:
        return uniform_weights(values.size)
    return projected / total


def logistic_objective(
    weights: np.ndarray, dataset: PairDataset, tau: float = TAU
) -> tuple[float, np.ndarray]:
    margin = (dataset.differences @ np.asarray(weights, dtype=np.float64)) / float(tau)
    mass = float(dataset.weights.sum())
    value = float(dataset.weights @ np.logaddexp(0.0, -margin)) / mass
    gradient = -(dataset.differences.T @ (dataset.weights / (1.0 + np.exp(margin)))) / (
        float(tau) * mass
    )
    return value, gradient


def zero_one_pairwise_loss(weights: np.ndarray, dataset: PairDataset) -> float:
    """Weighted fraction of misordered comparable pairs, ties counted as one half."""

    margin = dataset.differences @ np.asarray(weights, dtype=np.float64)
    mass = float(dataset.weights.sum())
    inverted = float(dataset.weights[margin < 0.0].sum())
    tied = float(dataset.weights[margin == 0.0].sum())
    return (inverted + 0.5 * tied) / mass


def learn_static_weights(
    dataset: PairDataset,
    *,
    tau: float = TAU,
    iterations: int = ITERATIONS,
    initial_step: float = INITIAL_STEP,
    armijo_c: float = ARMIJO_C,
    max_backtracks: int = MAX_BACKTRACKS,
) -> dict[str, Any]:
    weights = uniform_weights(dataset.advisers)
    value, gradient = logistic_objective(weights, dataset, tau)
    best_weights = weights.copy()
    best_value = value
    best_iteration = 0
    history = [value]
    backtracks_used = 0
    step = float(initial_step)
    stopped = "iterations_exhausted"
    for iteration in range(1, int(iterations) + 1):
        trial = step * 2.0
        improved = False
        candidate = weights
        candidate_value = value
        candidate_gradient = gradient
        for _attempt in range(int(max_backtracks)):
            proposal = project_to_simplex(weights - trial * gradient)
            move = proposal - weights
            squared = float(move @ move)
            if squared > 0.0:
                proposal_value, proposal_gradient = logistic_objective(proposal, dataset, tau)
                if proposal_value <= value - armijo_c * squared / trial:
                    candidate = proposal
                    candidate_value = proposal_value
                    candidate_gradient = proposal_gradient
                    improved = True
                    break
            trial *= 0.5
            backtracks_used += 1
        if not improved:
            # No admissible projected-gradient step remains: for this convex
            # objective that is the first-order optimality condition.
            stopped = "projected_gradient_stationary"
            break
        step = trial
        weights, value, gradient = candidate, candidate_value, candidate_gradient
        history.append(value)
        if value < best_value:
            best_value = value
            best_weights = weights.copy()
            best_iteration = iteration
    uniform = uniform_weights(dataset.advisers)
    return {
        "weights": best_weights.tolist(),
        "objective": float(best_value),
        "objective_at_uniform": float(logistic_objective(uniform, dataset, tau)[0]),
        "zero_one_pairwise_loss": zero_one_pairwise_loss(best_weights, dataset),
        "zero_one_pairwise_loss_at_uniform": zero_one_pairwise_loss(uniform, dataset),
        "best_iteration": int(best_iteration),
        "iterations_run": len(history) - 1,
        "stopping_reason": stopped,
        "backtracks_used": int(backtracks_used),
        "objective_history_first": history[: min(len(history), 10)],
        "objective_history_last": history[-min(len(history), 10) :],
        "pairs": dataset.pairs,
        "examples": dataset.examples,
        "tau": float(tau),
    }
