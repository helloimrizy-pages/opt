"""Stage 2C preflight: verify the frozen prior state before any new work.

Stage 2C (fragility-weighted robust specialist preservation) may begin only if
the balanced causal validation is STRONG GO, the Stage-1 pilot is GO, Stage 2A
is SURROGATE_NO_GO, and the Stage 2B development decision is
ROBUST_PRESERVATION_NO_GO with its frozen registry and gate values intact.

The Stage 2B development numbers are verified here purely to prove the frozen
negative result was not altered. They are historical motivation only and never
enter any Stage 2C objective, fragility value, or allocation calculation.

The seed-44 final split must remain unevaluated until the new seed-45
development gate passes; this module verifies its hashes without any model
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .balanced import array_sha256
from .io_utils import read_json
from .protection_allocations import load_frozen_registry
from .specialist_preservation import STAGE2B_DOMAINS
from .stage2b_preflight import verify_frozen_upstream_decisions

STAGE2B_RESULTS_DIRNAME = "robust_specialist_preservation"
STAGE2C_RESULTS_DIRNAME = "fragility_robust_preservation"

EXPECTED_STAGE2B_DECISION = "ROBUST_PRESERVATION_NO_GO"
EXPECTED_STAGE2B_REGISTRY_SHA256 = (
    "b0221262f0e51700cc16fa5e6a681f63ab6507a9d768714f853f3dfc3f87aa34"
)
# Frozen Stage 2B development gate values, used only to prove the negative
# result is unchanged (tolerance covers the rounding of the recorded digits).
EXPECTED_STAGE2B_GATE_VALUES = {
    "4to8": {
        "robust_worst_relative_delta": 0.031427,
        "random_mean_worst_relative_delta": 0.026288,
        "global_importance_worst_relative_delta": 0.011853,
        "average_specialization_worst_relative_delta": 0.028731,
    },
    "3to8": {
        "robust_worst_relative_delta": 0.053559,
        "random_mean_worst_relative_delta": 0.064823,
        "global_importance_worst_relative_delta": 0.056027,
        "average_specialization_worst_relative_delta": 0.050999,
    },
}
STAGE2B_GATE_VALUE_TOLERANCE = 5e-6


def verify_stage2c_upstream(results_root: Path) -> dict[str, Any]:
    """Verify every frozen upstream decision required before Stage 2C work."""

    upstream = verify_frozen_upstream_decisions(results_root)
    stage2b_dir = results_root / STAGE2B_RESULTS_DIRNAME
    decision = read_json(stage2b_dir / "stage2b_decision.json")
    if decision.get("decision") != EXPECTED_STAGE2B_DECISION:
        raise RuntimeError(
            f"Stage 2B decision is {decision.get('decision')!r}; Stage 2C requires "
            f"the frozen {EXPECTED_STAGE2B_DECISION} result"
        )
    if decision.get("development_decision", {}).get("passing_regimes") != []:
        raise RuntimeError("Stage 2B must have no passing development regimes")
    if decision.get("registry_sha256") != EXPECTED_STAGE2B_REGISTRY_SHA256:
        raise RuntimeError(
            "Stage 2B decision references an unexpected allocation registry hash"
        )
    gates = decision["development_gates"]
    for regime, expected in EXPECTED_STAGE2B_GATE_VALUES.items():
        regime_gates = gates[regime]
        observed = {
            "robust_worst_relative_delta": regime_gates["gate_a"][
                "robust_worst_relative_delta"
            ],
            "random_mean_worst_relative_delta": regime_gates["gate_a"][
                "random_mean_worst_relative_delta"
            ],
            "global_importance_worst_relative_delta": regime_gates["gate_b"][
                "global_importance_worst_relative_delta"
            ],
            "average_specialization_worst_relative_delta": regime_gates["gate_b"][
                "average_specialization_worst_relative_delta"
            ],
        }
        for name, expected_value in expected.items():
            if abs(observed[name] - expected_value) > STAGE2B_GATE_VALUE_TOLERANCE:
                raise RuntimeError(
                    f"Frozen Stage 2B value {regime}/{name}={observed[name]:.8f} does "
                    f"not match the recorded {expected_value}; refusing to reinterpret "
                    "the Stage 2B negative result"
                )
        if regime_gates["all_passed"] is not False:
            raise RuntimeError(
                f"Stage 2B regime {regime} is recorded as passing; the frozen "
                "NO_GO state is inconsistent"
            )

    registry = load_frozen_registry(stage2b_dir / "allocations")
    if registry["registry_sha256"] != EXPECTED_STAGE2B_REGISTRY_SHA256:
        raise RuntimeError("The frozen Stage 2B allocation registry hash changed")

    return {
        "passed": True,
        "upstream": upstream,
        "stage2b": {
            "decision": EXPECTED_STAGE2B_DECISION,
            "registry_sha256": registry["registry_sha256"],
            "gate_values_verified": EXPECTED_STAGE2B_GATE_VALUES,
            "historical_motivation_only": True,
            "stage2b_values_never_enter_stage2c_objective": True,
        },
        "forbidden_by_preregistration": [
            "using Stage 2B development outcomes to tune the Stage 2C objective",
            "using seed-43 outcomes to validate Stage 2C",
            "searching alternative fragility formulas or fitting coefficients",
            "using AOD, GQS, APD, or any expert-level delta-NLL surrogate",
            "evaluating seed 44 before FINAL_CONFIRMATION_GO",
            "adding bit-width regimes or protection budgets after evaluation",
        ],
    }


def verify_seed44_untouched(
    results_root: Path,
    stage2c_dir: Path | None = None,
    allow_authorized_final: bool = False,
) -> dict[str, Any]:
    """Verify the seed-44 final split metadata/hashes without model evaluation.

    Confirms the frozen seed-44 inputs still match the Stage 2B split manifest
    and that no seed-44 model outputs exist anywhere, unless a final evaluation
    was explicitly authorized by a FINAL_CONFIRMATION_GO decision.
    """

    stage2b_dir = results_root / STAGE2B_RESULTS_DIRNAME
    manifest = read_json(stage2b_dir / "splits" / "split_manifest.json")
    if manifest["final_seed"] != 44:
        raise RuntimeError("The Stage 2B final split seed is not 44")
    domains_report: dict[str, Any] = {}
    for domain in STAGE2B_DOMAINS:
        entry = manifest["domains"][domain]["final"]
        path = stage2b_dir / "splits" / "final" / f"{domain}.npz"
        with np.load(path, allow_pickle=False) as data:
            input_ids = np.asarray(data["input_ids"])
            mask = np.asarray(data["measurement_mask"])
        if array_sha256(input_ids) != entry["input_ids_sha256"]:
            raise RuntimeError(f"Seed-44 input hash changed for {domain}")
        if array_sha256(mask) != entry["measurement_mask_sha256"]:
            raise RuntimeError(f"Seed-44 measurement-mask hash changed for {domain}")
        domains_report[domain] = {
            "num_examples": int(input_ids.shape[0]),
            "input_ids_sha256": entry["input_ids_sha256"],
            "hash_verified_without_model_evaluation": True,
        }

    stage2b_final_losses = stage2b_dir / "final" / "losses"
    if stage2b_final_losses.exists() and any(stage2b_final_losses.iterdir()):
        raise RuntimeError(
            "Stage 2B final losses exist; the seed-44 split was evaluated, which "
            "the frozen ROBUST_PRESERVATION_NO_GO decision forbids"
        )
    stage2c_outputs_exist = False
    if stage2c_dir is not None:
        stage2c_final_losses = stage2c_dir / "final_seed44" / "losses"
        stage2c_outputs_exist = stage2c_final_losses.exists() and any(
            stage2c_final_losses.iterdir()
        )
        if stage2c_outputs_exist and not allow_authorized_final:
            raise RuntimeError(
                "Stage 2C seed-44 model outputs exist without an authorized final "
                "evaluation; the temporal separation rule was violated"
            )
    return {
        "passed": True,
        "final_seed": 44,
        "domains": domains_report,
        "stage2b_final_outputs_exist": False,
        "stage2c_final_outputs_exist": stage2c_outputs_exist,
        "verified_without_model_evaluation": True,
    }
