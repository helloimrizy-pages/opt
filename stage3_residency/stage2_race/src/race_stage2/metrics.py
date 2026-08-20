"""Stage 2 comparison metrics.

The oracle-gap denominator is always the *original* Stage 0 strongest-simple cost,
never a Stage-1-relative redefinition, so gap-closure numbers stay comparable across
Stage 0, Stage 1 and Stage 2.
"""

from __future__ import annotations

from typing import Any


def race_improvement(stage1: float, race: float) -> float:
    """Primary practical metric: fractional cost reduction against Stage 1."""

    if stage1 <= 0:
        raise ValueError("Stage 1 cost must be positive")
    return (float(stage1) - float(race)) / float(stage1)


def original_oracle_gap_closed(simple: float, race: float, oracle: float) -> float | None:
    denominator = float(simple) - float(oracle)
    if abs(denominator) <= 1e-12:
        return None
    return (float(simple) - float(race)) / denominator


def stage1_residual_recovered(stage1: float, race: float, oracle: float) -> float | None:
    denominator = float(stage1) - float(oracle)
    if abs(denominator) <= 1e-12:
        return None
    return (float(stage1) - float(race)) / denominator


def residual_headroom(simple: float, race: float, oracle: float) -> float:
    if simple <= 0:
        raise ValueError("Stage 0 simple-baseline cost must be positive")
    return (float(race) - float(oracle)) / float(simple)


def regression_ratio(stage1: float, race: float) -> float:
    if stage1 <= 0:
        raise ValueError("Stage 1 cost must be positive")
    return float(race) / float(stage1)


def comparison_metrics(
    *, simple: float, stage1: float, race: float, oracle: float
) -> dict[str, Any]:
    return {
        "stage0_simple_cost": float(simple),
        "stage1_cost": float(stage1),
        "race_cost": float(race),
        "oracle_cost": float(oracle),
        "race_improvement_over_stage1": race_improvement(stage1, race),
        "original_oracle_gap_closed": original_oracle_gap_closed(simple, race, oracle),
        "stage1_original_oracle_gap_closed": original_oracle_gap_closed(
            simple, stage1, oracle
        ),
        "stage1_residual_recovered": stage1_residual_recovered(stage1, race, oracle),
        "residual_headroom": residual_headroom(simple, race, oracle),
        "regression_ratio": regression_ratio(stage1, race),
        "normalized_cost_vs_stage0_simple": float(race) / float(simple),
    }
