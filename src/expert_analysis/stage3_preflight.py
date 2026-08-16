"""Stage 3 preflight: verify the frozen prior state before any new work.

Stage 3 (measured expert-damage preservation) may begin only if the balanced
causal validation is STRONG GO, the Stage-1 pilot is GO, Stage 2A is
SURROGATE_NO_GO, Stage 2B is ROBUST_PRESERVATION_NO_GO, and Stage 2C is
FRAGILITY_ROBUST_NO_GO, each with its frozen registry and gate values intact.

Scope note, recorded here and in the Stage 3 preregistration: the Stage 2A
restriction blocks SURROGATE-predicted expert-level delta NLL (AOD, GQS, APD,
fitted regressors, and the surrogate-derived full cost matrix). Stage 3 does
not estimate per-expert damage. It MEASURES it directly, one expert at a time,
with the audited Stage-1 QDQ on the frozen calibration data — the same
ground-truth quantity the Stage 1 pilot already measured for 16 experts. The
Stage 2C rule that no alternative fragility weighting may be searched is also
respected: Stage 3 does not reweight the Stage 2C objective; it replaces
score-based coverage with measured damage under a new, explicitly authorized
preregistration. All frozen negative decisions are preserved unchanged.

The seed-44 final split must remain unevaluated until the new seed-46
development gate passes; this module verifies its hashes without any model
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json
from .stage2c_preflight import (
    STAGE2C_RESULTS_DIRNAME,
    verify_seed44_untouched,
    verify_stage2c_upstream,
)

STAGE3_RESULTS_DIRNAME = "measured_damage_preservation"

EXPECTED_STAGE2C_DECISION = "FRAGILITY_ROBUST_NO_GO"
EXPECTED_STAGE2C_REGISTRY_SHA256 = (
    "b1b6b9a68c0840e60b1d080678ed7a8fb7f56a0595100a76223c4f3860b52caf"
)
EXPECTED_STAGE2C_PREREGISTRATION_SHA256 = (
    "8001fb770c1883d1bd887c282694559c9924500dbf5a08751c7d3f1d87f17385"
)
EXPECTED_STAGE2C_FRAGILITY_SHA256 = (
    "82db7846ebfdedd62c6d2403a3af2869de507449dc7c42e63790e697291bb624"
)
# Frozen Stage 2C development gate values, used only to prove the negative
# result is unchanged (tolerance covers the rounding of the recorded digits).
EXPECTED_STAGE2C_GATE_VALUES = {
    "4to8": {
        "fragility_robust_worst_relative_delta": 0.025616,
        "robust_functional_worst_relative_delta": 0.033205,
        "random_mean_worst_relative_delta": 0.030074,
        "global_importance_worst_relative_delta": 0.012976,
        "average_specialization_worst_relative_delta": 0.031682,
    },
    "3to8": {
        "fragility_robust_worst_relative_delta": 0.052080,
        "robust_functional_worst_relative_delta": 0.050717,
        "random_mean_worst_relative_delta": 0.068232,
        "global_importance_worst_relative_delta": 0.054973,
        "average_specialization_worst_relative_delta": 0.054626,
    },
}
STAGE2C_GATE_VALUE_TOLERANCE = 5e-6


def verify_stage3_upstream(results_root: Path) -> dict[str, Any]:
    """Verify every frozen upstream decision required before Stage 3 work."""

    upstream = verify_stage2c_upstream(results_root)
    stage2c_dir = results_root / STAGE2C_RESULTS_DIRNAME
    decision = read_json(stage2c_dir / "stage2c_decision.json")
    if decision.get("decision") != EXPECTED_STAGE2C_DECISION:
        raise RuntimeError(
            f"Stage 2C decision is {decision.get('decision')!r}; Stage 3 requires "
            f"the frozen {EXPECTED_STAGE2C_DECISION} result"
        )
    if decision.get("development_decision", {}).get("authorized_regimes") != []:
        raise RuntimeError("Stage 2C must have no authorized development regimes")
    if decision.get("registry_sha256") != EXPECTED_STAGE2C_REGISTRY_SHA256:
        raise RuntimeError(
            "Stage 2C decision references an unexpected allocation registry hash"
        )
    if decision.get("preregistration_sha256") != EXPECTED_STAGE2C_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "Stage 2C decision references an unexpected preregistration hash"
        )
    if decision.get("fragility_sha256") != EXPECTED_STAGE2C_FRAGILITY_SHA256:
        raise RuntimeError(
            "Stage 2C decision references an unexpected fragility record hash"
        )
    gates = decision["development_gates"]
    for regime, expected in EXPECTED_STAGE2C_GATE_VALUES.items():
        regime_gates = gates[regime]
        observed = {
            "fragility_robust_worst_relative_delta": regime_gates["gate_a"][
                "fragility_robust_worst_relative_delta"
            ],
            "robust_functional_worst_relative_delta": regime_gates["gate_a"][
                "robust_functional_worst_relative_delta"
            ],
            "random_mean_worst_relative_delta": regime_gates["gate_b"][
                "random_mean_worst_relative_delta"
            ],
            "global_importance_worst_relative_delta": regime_gates["gate_c"][
                "global_importance_worst_relative_delta"
            ],
            "average_specialization_worst_relative_delta": regime_gates["gate_c"][
                "average_specialization_worst_relative_delta"
            ],
        }
        for name, expected_value in expected.items():
            if abs(observed[name] - expected_value) > STAGE2C_GATE_VALUE_TOLERANCE:
                raise RuntimeError(
                    f"Frozen Stage 2C value {regime}/{name}={observed[name]:.8f} does "
                    f"not match the recorded {expected_value}; refusing to reinterpret "
                    "the Stage 2C negative result"
                )
        if regime_gates["all_passed"] is not False:
            raise RuntimeError(
                f"Stage 2C regime {regime} is recorded as passing; the frozen "
                "NO_GO state is inconsistent"
            )

    audit_path = stage2c_dir / "audits" / "independent_audit.json"
    audit = read_json(audit_path)
    if audit.get("passed") is not True or audit.get("checks_failed") != 0:
        raise RuntimeError(
            "The frozen Stage 2C independent audit is not a clean pass; Stage 3 "
            "may not build on an unaudited or failed prior stage"
        )

    return {
        "passed": True,
        "upstream": upstream,
        "stage2c": {
            "decision": EXPECTED_STAGE2C_DECISION,
            "registry_sha256": EXPECTED_STAGE2C_REGISTRY_SHA256,
            "preregistration_sha256": EXPECTED_STAGE2C_PREREGISTRATION_SHA256,
            "fragility_sha256": EXPECTED_STAGE2C_FRAGILITY_SHA256,
            "gate_values_verified": EXPECTED_STAGE2C_GATE_VALUES,
            "independent_audit_passed": True,
            "historical_motivation_only": True,
            "stage2c_values_never_enter_stage3_objective": True,
        },
        "measured_not_estimated": {
            "stage2a_blocks": "surrogate-predicted expert-level delta NLL",
            "stage3_uses": (
                "directly measured per-expert delta NLL from single-expert QDQ "
                "on frozen calibration data; no AOD, GQS, APD, fitted model, or "
                "any predictive surrogate enters any Stage 3 quantity"
            ),
            "stage2c_no_reweighting_respected": True,
        },
        "forbidden_by_preregistration": [
            "using AOD, GQS, APD, or any predictive delta-NLL surrogate",
            "using seed-43 or seed-45 outcomes to tune or validate Stage 3",
            "reweighting or refitting the Stage 2C fragility objective",
            "modifying the measured-damage objective after seeing seed-46 results",
            "evaluating seed 44 before FINAL_CONFIRMATION_GO",
            "adding bit-width regimes or protection budgets after evaluation",
        ],
    }


def verify_seed44_untouched_stage3(
    results_root: Path,
    stage3_dir: Path | None = None,
    allow_authorized_final: bool = False,
) -> dict[str, Any]:
    """Verify seed-44 isolation across Stage 2B, Stage 2C, and Stage 3.

    Reuses the Stage 2C hash-only verification (which also proves no Stage 2B
    or Stage 2C seed-44 outputs exist) and additionally proves no Stage 3
    seed-44 model outputs exist unless a final evaluation was authorized.
    """

    report = verify_seed44_untouched(
        results_root, results_root / STAGE2C_RESULTS_DIRNAME
    )
    stage3_outputs_exist = False
    if stage3_dir is not None:
        stage3_final_losses = stage3_dir / "final_seed44" / "losses"
        stage3_outputs_exist = stage3_final_losses.exists() and any(
            stage3_final_losses.iterdir()
        )
        if stage3_outputs_exist and not allow_authorized_final:
            raise RuntimeError(
                "Stage 3 seed-44 model outputs exist without an authorized final "
                "evaluation; the temporal separation rule was violated"
            )
    report["stage3_final_outputs_exist"] = stage3_outputs_exist
    return report
