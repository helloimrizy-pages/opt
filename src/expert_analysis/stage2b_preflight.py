"""Stage 2B preflight: verify the frozen scientific state before any new work.

Stage 2B may begin only if the balanced causal validation is STRONG GO, the
Stage-1 quantization pilot is GO, and the Stage-2A surrogate validation is
SURROGATE_NO_GO. The surrogate failure is frozen: no cost matrix, no new
delta-NLL surrogate, and no post-hoc promotion of AOD/GQS/APD is permitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json

EXPECTED_SURROGATE_VALUES = {
    "aod_overall_spearman": 0.1225,
    "gqs_overall_spearman": 0.0975,
    "aod_specificity_spearman": 0.4059,
    "gqs_specificity_spearman": 0.4676,
}


def verify_frozen_upstream_decisions(results_root: Path) -> dict[str, Any]:
    """Check all three frozen upstream decisions and return an evidence report."""

    balanced = read_json(
        results_root / "expert_domain_balanced_causal_validation" / "results.json"
    )
    balanced_decision = balanced["balanced_analysis"]["decision"]
    if balanced_decision.get("label") != "STRONG GO":
        raise RuntimeError(
            "Balanced causal validation is not STRONG GO; Stage 2B is not authorized"
        )
    correlations = {
        row["predictor"]: row
        for row in balanced["balanced_analysis"]["correlation_results"]
    }
    expected_correlations = {
        "functional_specialization_margin": 0.753,
        "routing_specialization_margin": 0.711,
        "target_routing_frequency": 0.417,
    }
    for predictor, expected in expected_correlations.items():
        observed = correlations[predictor]["spearman"]
        if abs(observed - expected) > 5e-3:
            raise RuntimeError(
                f"Balanced correlation {predictor}={observed:.4f} does not match the "
                f"frozen value {expected}"
            )

    stage1 = read_json(
        results_root / "expert_quantization_pilot" / "stage1_decision.json"
    )
    if stage1.get("decision") != "GO":
        raise RuntimeError("Stage-1 quantization pilot decision is not GO")
    if stage1.get("selected_bit_width") != 4:
        raise RuntimeError("Stage-1 GO was expected at 4-bit")

    surrogate = read_json(
        results_root / "quantization_cost_surrogate" / "surrogate_decision.json"
    )
    if surrogate.get("decision") != "SURROGATE_NO_GO":
        raise RuntimeError("Stage-2A surrogate decision is not SURROGATE_NO_GO")
    if surrogate.get("full_cost_matrix_authorized") is not False:
        raise RuntimeError("The full cost matrix must remain unauthorized")
    checks = {
        "aod_overall_spearman": surrogate["aod_gates"]["gate_a"]["overall_spearman"],
        "gqs_overall_spearman": surrogate["gqs_gates"]["gate_a"]["overall_spearman"],
        "aod_specificity_spearman": surrogate["aod_gates"]["gate_c"][
            "specificity_spearman"
        ],
        "gqs_specificity_spearman": surrogate["gqs_gates"]["gate_c"][
            "specificity_spearman"
        ],
    }
    for name, expected in EXPECTED_SURROGATE_VALUES.items():
        if abs(checks[name] - expected) > 5e-4:
            raise RuntimeError(
                f"Frozen Stage-2A value {name}={checks[name]:.6f} does not match the "
                f"recorded {expected}; refusing to reinterpret the surrogate result"
            )
    if not (
        surrogate["aod_gates"]["gate_a"]["passed"] is False
        and surrogate["aod_gates"]["gate_b"]["passed"] is False
        and surrogate["gqs_gates"]["gate_a"]["passed"] is False
        and surrogate["gqs_gates"]["gate_b"]["passed"] is False
    ):
        raise RuntimeError("Stage-2A gates A/B must be failed for both AOD and GQS")

    return {
        "passed": True,
        "balanced_causal_validation": {
            "decision": "STRONG GO",
            "spearman_by_predictor": {
                predictor: row["spearman"] for predictor, row in correlations.items()
            },
        },
        "stage1_quantization_pilot": {
            "decision": "GO",
            "selected_bit_width": 4,
        },
        "stage2a_surrogate": {
            "decision": "SURROGATE_NO_GO",
            "values": checks,
            "frozen": True,
        },
        "forbidden_by_preregistration": [
            "running build_quantization_cost_matrix.py",
            "promoting APD or any surrogate post hoc",
            "fitting new delta-NLL surrogates or regressors",
            "using Stage-1 delta NLL or Stage-2A scores to select experts",
            "modifying the Stage 2B objective after seeing development results",
        ],
    }
