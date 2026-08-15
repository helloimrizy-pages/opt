"""Stage 2C development gates, decisions, final criteria, and mechanism checks.

All decision constants are frozen in the Stage 2C preregistration before any
seed-45 NLL is computed. Point-estimate metrics, paired bootstrap machinery,
and the worst-domain recomputation inside every replicate are reused unchanged
from the audited Stage 2B statistics module.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .protection_statistics import (
    MethodStatistics,
    random_baseline_statistics,
)
from .specialist_preservation import STAGE2B_DOMAINS
from .statistics import safe_spearman

STAGE2C_BOOTSTRAP_REPLICATES = 1000
STAGE2C_BOOTSTRAP_SEED = 20260815
STAGE2C_DEVELOPMENT_BUDGET_FRACTION = 0.20
# Gate E: Fragility-Robust MeanRelativeDelta may not be more than 10%
# relatively worse than the lower comparator mean. The epsilon exists only for
# denominator safety when the comparator magnitude is essentially zero; it is
# not an absolute tolerance floor and must not be redefined.
GATE_E_RELATIVE_TOLERANCE = 0.10
GATE_E_DENOMINATOR_EPSILON = 1e-12
GATE_D_REQUIRED_POSITIVE_DOMAINS = 3
# Final criteria (preregistered before seed-45 evaluation).
FINAL_REQUIRED_POINT_BUDGETS = 3
QUALIFIED_REQUIRED_POINT_BUDGETS = 2
FINAL_REQUIRED_CI_WINS_VS_AVERAGE = 2
FINAL_REQUIRED_POINT_WINS_VS_GLOBAL = 2
# "Majority of budgets" for systematic negative recovery: at least 3 of 4.
SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS = 3

DEVELOPMENT_GO_DECISION = "FINAL_CONFIRMATION_GO"
DEVELOPMENT_NO_GO_DECISION = "FRAGILITY_ROBUST_NO_GO"


def stage2c_development_gates(
    fragility_robust: MethodStatistics,
    robust_functional: MethodStatistics,
    randoms: list[MethodStatistics],
    global_importance: MethodStatistics,
    average_specialization: MethodStatistics,
) -> dict[str, Any]:
    """The five preregistered Stage 2C development gates for one regime."""

    random_stats = random_baseline_statistics(randoms)
    gate_a = {
        "name": "fixes_stage2b_beats_robust_functional",
        "fragility_robust_worst_relative_delta": fragility_robust.worst_relative_delta,
        "robust_functional_worst_relative_delta": robust_functional.worst_relative_delta,
        "passed": bool(
            fragility_robust.worst_relative_delta
            < robust_functional.worst_relative_delta
        ),
    }
    gate_b = {
        "name": "beats_random_mean",
        "fragility_robust_worst_relative_delta": fragility_robust.worst_relative_delta,
        "random_mean_worst_relative_delta": random_stats["mean_worst_relative_delta"],
        "passed": bool(
            fragility_robust.worst_relative_delta
            < random_stats["mean_worst_relative_delta"]
        ),
    }
    gate_c = {
        "name": "beats_both_strong_simple_baselines",
        "fragility_robust_worst_relative_delta": fragility_robust.worst_relative_delta,
        "global_importance_worst_relative_delta": global_importance.worst_relative_delta,
        "average_specialization_worst_relative_delta": (
            average_specialization.worst_relative_delta
        ),
        "passed": bool(
            fragility_robust.worst_relative_delta
            < global_importance.worst_relative_delta
            and fragility_robust.worst_relative_delta
            < average_specialization.worst_relative_delta
        ),
    }
    positive_recovery = [
        domain
        for domain in STAGE2B_DOMAINS
        if fragility_robust.recovery[domain] > 0
    ]
    gate_d = {
        "name": "broad_recovery_vs_uniform_base",
        "recovery_by_domain": dict(fragility_robust.recovery),
        "positive_recovery_domains": positive_recovery,
        "required_positive_domains": GATE_D_REQUIRED_POSITIVE_DOMAINS,
        "passed": bool(len(positive_recovery) >= GATE_D_REQUIRED_POSITIVE_DOMAINS),
    }
    comparator_mean = min(
        global_importance.mean_relative_delta,
        average_specialization.mean_relative_delta,
    )
    denominator = max(abs(comparator_mean), GATE_E_DENOMINATOR_EPSILON)
    relative_worseness = (
        fragility_robust.mean_relative_delta - comparator_mean
    ) / denominator
    gate_e = {
        "name": "mean_quality_within_ten_percent_of_better_baseline",
        "fragility_robust_mean_relative_delta": fragility_robust.mean_relative_delta,
        "global_importance_mean_relative_delta": global_importance.mean_relative_delta,
        "average_specialization_mean_relative_delta": (
            average_specialization.mean_relative_delta
        ),
        "comparator_mean_relative_delta": comparator_mean,
        "relative_worseness": relative_worseness,
        "relative_tolerance": GATE_E_RELATIVE_TOLERANCE,
        "denominator_epsilon": GATE_E_DENOMINATOR_EPSILON,
        "epsilon_applied": bool(
            abs(comparator_mean) < GATE_E_DENOMINATOR_EPSILON
        ),
        "passed": bool(relative_worseness <= GATE_E_RELATIVE_TOLERANCE),
    }
    gates = {
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_d": gate_d,
        "gate_e": gate_e,
    }
    gates["all_passed"] = bool(
        all(gates[key]["passed"] for key in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"))
    )
    gates["random_baseline_statistics"] = random_stats
    return gates


def stage2c_development_decision(
    gates_by_regime: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    passing = [
        regime for regime, gates in gates_by_regime.items() if gates["all_passed"]
    ]
    decision = DEVELOPMENT_GO_DECISION if passing else DEVELOPMENT_NO_GO_DECISION
    return {
        "decision": decision,
        "authorized_regimes": passing,
        "regimes_evaluated": {
            regime: "PASS" if gates["all_passed"] else "FAIL"
            for regime, gates in gates_by_regime.items()
        },
        "rule": (
            "FINAL_CONFIRMATION_GO if at least one precision regime passes all "
            "five development gates on the new seed-45 split; only passing "
            "regimes are authorized for seed-44 final confirmation. A failing "
            "regime is never final-tested. On FRAGILITY_ROBUST_NO_GO the "
            "negative result is preserved, seed 44 stays untouched, and no "
            "alternative weighting may be searched."
        ),
    }


def stage2c_final_regime_assessment(
    statistics: Mapping[tuple[str, float], Mapping[str, MethodStatistics]],
    comparisons_vs_average: Mapping[float, Mapping[str, Any]],
    comparisons_vs_global: Mapping[float, Mapping[str, Any]],
    regime: str,
    budgets: tuple[float, ...],
) -> dict[str, Any]:
    """Apply the frozen Stage 2C final success requirements to one regime."""

    point_wins: dict[str, Any] = {}
    improvements_over_average: list[float] = []
    ci_wins_vs_average = 0
    point_wins_vs_global = 0
    beats_rf_budgets = 0
    beats_random_budgets = 0
    beats_both_simple_budgets = 0
    for budget in budgets:
        methods = statistics[(regime, budget)]
        fragility_robust = methods["fragility_robust"]
        randoms = [
            item for name, item in methods.items() if name.startswith("random_seed")
        ]
        random_mean_worst = random_baseline_statistics(randoms)[
            "mean_worst_relative_delta"
        ]
        beats_rf = (
            fragility_robust.worst_relative_delta
            < methods["robust_functional"].worst_relative_delta
        )
        beats_random = fragility_robust.worst_relative_delta < random_mean_worst
        beats_global = (
            fragility_robust.worst_relative_delta
            < methods["global_importance"].worst_relative_delta
        )
        beats_average = (
            fragility_robust.worst_relative_delta
            < methods["average_specialization"].worst_relative_delta
        )
        point_wins[str(budget)] = {
            "beats_robust_functional": bool(beats_rf),
            "beats_random_mean": bool(beats_random),
            "beats_global_importance": bool(beats_global),
            "beats_average_specialization": bool(beats_average),
            "all_four": bool(beats_rf and beats_random and beats_global and beats_average),
        }
        beats_rf_budgets += int(beats_rf)
        beats_random_budgets += int(beats_random)
        beats_both_simple_budgets += int(beats_global and beats_average)
        improvements_over_average.append(
            methods["average_specialization"].worst_relative_delta
            - fragility_robust.worst_relative_delta
        )
        if comparisons_vs_average[budget]["ci_favors_first"]:
            ci_wins_vs_average += 1
        if comparisons_vs_global[budget]["difference"] < 0:
            point_wins_vs_global += 1

    systematic_negative_domains = []
    for domain in STAGE2B_DOMAINS:
        negative_budgets = sum(
            1
            for budget in budgets
            if statistics[(regime, budget)]["fragility_robust"].recovery[domain] < 0
        )
        if negative_budgets >= SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS:
            systematic_negative_domains.append(domain)

    all_four_count = sum(1 for value in point_wins.values() if value["all_four"])
    average_improvement = float(np.mean(improvements_over_average))
    requirement_1 = all_four_count >= FINAL_REQUIRED_POINT_BUDGETS
    requirement_2 = average_improvement > 0
    requirement_3 = ci_wins_vs_average >= FINAL_REQUIRED_CI_WINS_VS_AVERAGE
    requirement_4 = point_wins_vs_global >= FINAL_REQUIRED_POINT_WINS_VS_GLOBAL
    requirement_5 = not systematic_negative_domains
    strong = bool(
        requirement_1
        and requirement_2
        and requirement_3
        and requirement_4
        and requirement_5
    )
    # Preregistered SUCCESS WITH QUALIFICATIONS rule: not strong, positive
    # average improvement over Average-Specialization, no systematically
    # negative domain, and either wins over both strong simple baselines at
    # >= 2 of 4 budgets, or a clear Stage 2B fix (beats Robust-Functional and
    # the random mean at >= 3 of 4 budgets).
    qualified = bool(
        not strong
        and requirement_2
        and requirement_5
        and (
            beats_both_simple_budgets >= QUALIFIED_REQUIRED_POINT_BUDGETS
            or (
                beats_rf_budgets >= FINAL_REQUIRED_POINT_BUDGETS
                and beats_random_budgets >= FINAL_REQUIRED_POINT_BUDGETS
            )
        )
    )
    return {
        "regime": regime,
        "point_wins_by_budget": point_wins,
        "budgets_with_all_four_point_wins": all_four_count,
        "budgets_beating_robust_functional": beats_rf_budgets,
        "budgets_beating_random_mean": beats_random_budgets,
        "budgets_beating_both_simple_baselines": beats_both_simple_budgets,
        "average_improvement_over_average_specialization": average_improvement,
        "ci_wins_vs_average_specialization": ci_wins_vs_average,
        "point_wins_vs_global_importance": point_wins_vs_global,
        "systematic_negative_recovery_domains": systematic_negative_domains,
        "requirement_1_point_wins": bool(requirement_1),
        "requirement_2_positive_average_improvement": bool(requirement_2),
        "requirement_3_ci_wins_vs_average": bool(requirement_3),
        "requirement_4_point_wins_vs_global": bool(requirement_4),
        "requirement_5_no_systematic_negative_recovery": bool(requirement_5),
        "strong_success": strong,
        "qualified_success": qualified,
    }


def stage2c_final_decision(
    regime_assessments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    strong = [r for r, a in regime_assessments.items() if a["strong_success"]]
    qualified = [r for r, a in regime_assessments.items() if a["qualified_success"]]
    if strong:
        label = "STRONG SUCCESS"
    elif qualified:
        label = "SUCCESS WITH QUALIFICATIONS"
    else:
        label = "NEGATIVE RESULT"
    return {
        "decision": label,
        "strong_success_regimes": strong,
        "qualified_success_regimes": qualified,
        "rule": (
            "STRONG SUCCESS requires, in at least one authorized regime: "
            f"all-four point wins at >= {FINAL_REQUIRED_POINT_BUDGETS} of 4 "
            "budgets (vs Robust-Functional, random mean, Global-Importance, "
            "Average-Specialization), positive average worst-domain improvement "
            f"over Average-Specialization, >= {FINAL_REQUIRED_CI_WINS_VS_AVERAGE} "
            "worst-domain CIs entirely favoring Fragility-Robust vs "
            f"Average-Specialization, >= {FINAL_REQUIRED_POINT_WINS_VS_GLOBAL} "
            "point wins vs Global-Importance, and no domain with negative "
            f"recovery at >= {SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS} of 4 "
            "budgets. SUCCESS WITH QUALIFICATIONS follows the preregistered "
            "qualified rule. Anything else is a NEGATIVE RESULT; goalposts are "
            "never moved."
        ),
    }


def protection_shift_analysis(
    fragility_coverage: Mapping[str, Mapping[str, float]],
    robust_functional_coverage: Mapping[str, Mapping[str, float]],
    q_norm_by_regime: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Descriptive mechanism check: did fragile domains gain coverage?

    ``ProtectionShift_d = Coverage_FragilityRobust_d - Coverage_RobustFunctional_d``
    compared with calibration ``q_norm[d]`` via Spearman across the four
    domains. Descriptive only; never used to change the optimizer.
    """

    output: dict[str, Any] = {"descriptive_only": True, "regimes": {}}
    for regime, q_norm in q_norm_by_regime.items():
        shifts = {
            domain: float(
                fragility_coverage[regime][domain]
                - robust_functional_coverage[regime][domain]
            )
            for domain in STAGE2B_DOMAINS
        }
        fragility_values = np.asarray([q_norm[d] for d in STAGE2B_DOMAINS])
        shift_values = np.asarray([shifts[d] for d in STAGE2B_DOMAINS])
        output["regimes"][regime] = {
            "protection_shift_by_domain": shifts,
            "q_norm_by_domain": {d: float(q_norm[d]) for d in STAGE2B_DOMAINS},
            "spearman_fragility_vs_shift": safe_spearman(
                fragility_values, shift_values
            ),
        }
    return output


def fragility_transfer_check(
    q_norm_by_regime: Mapping[str, Mapping[str, float]],
    final_uniform_base_relative_delta: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Compare calibration fragility ranking with seed-44 uniform-base damage.

    Perfect agreement is not required for success; this only tests whether
    calibration-level domain fragility transfers to the held-out final split.
    """

    output: dict[str, Any] = {"regimes": {}}
    for regime, q_norm in q_norm_by_regime.items():
        observed = final_uniform_base_relative_delta[regime]
        fragility_values = np.asarray([q_norm[d] for d in STAGE2B_DOMAINS])
        observed_values = np.asarray([observed[d] for d in STAGE2B_DOMAINS])
        calibration_rank = [
            STAGE2B_DOMAINS[i] for i in np.argsort(-fragility_values)
        ]
        final_rank = [STAGE2B_DOMAINS[i] for i in np.argsort(-observed_values)]
        output["regimes"][regime] = {
            "calibration_q_norm": {d: float(q_norm[d]) for d in STAGE2B_DOMAINS},
            "final_uniform_base_relative_delta": {
                d: float(observed[d]) for d in STAGE2B_DOMAINS
            },
            "spearman": safe_spearman(fragility_values, observed_values),
            "calibration_rank_most_fragile_first": calibration_rank,
            "final_rank_most_degraded_first": final_rank,
            "most_fragile_domain_calibration": calibration_rank[0],
            "most_degraded_domain_final": final_rank[0],
            "most_fragile_domain_matches": bool(
                calibration_rank[0] == final_rank[0]
            ),
        }
    return output
