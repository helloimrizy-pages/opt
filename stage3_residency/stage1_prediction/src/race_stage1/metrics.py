from __future__ import annotations

from typing import Any


def baseline_improvement(simple: float, method: float) -> float:
    if simple <= 0:
        raise ValueError("Simple-baseline cost must be positive")
    return (float(simple) - float(method)) / float(simple)


def oracle_gap_closed(simple: float, method: float, oracle: float) -> float | None:
    denominator = float(simple) - float(oracle)
    if abs(denominator) <= 1e-12:
        return None
    return (float(simple) - float(method)) / denominator


def residual_headroom(simple: float, method: float, oracle: float) -> float:
    if simple <= 0:
        raise ValueError("Simple-baseline cost must be positive")
    return (float(method) - float(oracle)) / float(simple)


def comparison_metrics(simple: float, method: float, oracle: float) -> dict[str, Any]:
    return {
        "baseline_improvement": baseline_improvement(simple, method),
        "oracle_gap_closed": oracle_gap_closed(simple, method, oracle),
        "residual_headroom": residual_headroom(simple, method, oracle),
        "absolute_improvement": float(simple) - float(method),
        "absolute_residual": float(method) - float(oracle),
    }
