"""Stage 3 development gates, decisions, final criteria, and mechanism checks.

All decision constants are frozen in the Stage 3 preregistration before any
probe or seed-46 NLL is computed. Point-estimate metrics, paired bootstrap
machinery, and the worst-domain recomputation inside every replicate are
reused unchanged from the audited Stage 2B statistics module; the shared gate
tolerances are imported from the audited Stage 2C constants.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .fragility_statistics import (
    FINAL_REQUIRED_CI_WINS_VS_AVERAGE,
    FINAL_REQUIRED_POINT_BUDGETS,
    FINAL_REQUIRED_POINT_WINS_VS_GLOBAL,
    GATE_D_REQUIRED_POSITIVE_DOMAINS,
    GATE_E_DENOMINATOR_EPSILON,
    GATE_E_RELATIVE_TOLERANCE,
    QUALIFIED_REQUIRED_POINT_BUDGETS,
    SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS,
)
from .protection_statistics import MethodStatistics, random_baseline_statistics
from .specialist_preservation import STAGE2B_DOMAINS
from .statistics import safe_spearman

STAGE3_DEVELOPMENT_GO_DECISION = "FINAL_CONFIRMATION_GO"
STAGE3_DEVELOPMENT_NO_GO_DECISION = "MEASURED_DAMAGE_NO_GO"


def stage3_development_gates(
    measured_damage: MethodStatistics,
    robust_functional: MethodStatistics,
    fragility_robust: MethodStatistics,
    randoms: list[MethodStatistics],
    global_importance: MethodStatistics,
    average_specialization: MethodStatistics,
) -> dict[str, Any]:
    """The five preregistered Stage 3 development gates for one regime."""

    random_stats = random_baseline_statistics(randoms)
    gate_a = {
        "name": "fixes_prior_robust_attempts",
        "measured_damage_worst_relative_delta": measured_damage.worst_relative_delta,
        "robust_functional_worst_relative_delta": robust_functional.worst_relative_delta,
        "fragility_robust_worst_relative_delta": fragility_robust.worst_relative_delta,
        "passed": bool(
            measured_damage.worst_relative_delta
            < robust_functional.worst_relative_delta
            and measured_damage.worst_relative_delta
            < fragility_robust.worst_relative_delta
        ),
    }
    gate_b = {
        "name": "beats_random_mean",
        "measured_damage_worst_relative_delta": measured_damage.worst_relative_delta,
        "random_mean_worst_relative_delta": random_stats["mean_worst_relative_delta"],
        "passed": bool(
            measured_damage.worst_relative_delta
            < random_stats["mean_worst_relative_delta"]
        ),
    }
    gate_c = {
        "name": "beats_both_strong_simple_baselines",
        "measured_damage_worst_relative_delta": measured_damage.worst_relative_delta,
        "global_importance_worst_relative_delta": global_importance.worst_relative_delta,
        "average_specialization_worst_relative_delta": (
            average_specialization.worst_relative_delta
        ),
        "passed": bool(
            measured_damage.worst_relative_delta
            < global_importance.worst_relative_delta
            and measured_damage.worst_relative_delta
            < average_specialization.worst_relative_delta
        ),
    }
    positive_recovery = [
        domain
        for domain in STAGE2B_DOMAINS
        if measured_damage.recovery[domain] > 0
    ]
    gate_d = {
        "name": "broad_recovery_vs_uniform_base",
        "recovery_by_domain": dict(measured_damage.recovery),
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
        measured_damage.mean_relative_delta - comparator_mean
    ) / denominator
    gate_e = {
        "name": "mean_quality_within_ten_percent_of_better_baseline",
        "measured_damage_mean_relative_delta": measured_damage.mean_relative_delta,
        "global_importance_mean_relative_delta": global_importance.mean_relative_delta,
        "average_specialization_mean_relative_delta": (
            average_specialization.mean_relative_delta
        ),
        "comparator_mean_relative_delta": comparator_mean,
        "relative_worseness": relative_worseness,
        "relative_tolerance": GATE_E_RELATIVE_TOLERANCE,
        "denominator_epsilon": GATE_E_DENOMINATOR_EPSILON,
        "epsilon_applied": bool(abs(comparator_mean) < GATE_E_DENOMINATOR_EPSILON),
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
        all(
            gates[key]["passed"]
            for key in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e")
        )
    )
    gates["random_baseline_statistics"] = random_stats
    return gates


def stage3_development_decision(
    gates_by_regime: Mapping[str, Mapping[str, Any]],
    additivity_authorized_regimes: list[str],
) -> dict[str, Any]:
    passing = [
        regime for regime, gates in gates_by_regime.items() if gates["all_passed"]
    ]
    decision = (
        STAGE3_DEVELOPMENT_GO_DECISION
        if passing
        else STAGE3_DEVELOPMENT_NO_GO_DECISION
    )
    return {
        "decision": decision,
        "authorized_regimes": passing,
        "additivity_authorized_regimes": list(additivity_authorized_regimes),
        "regimes_evaluated": {
            regime: "PASS" if gates["all_passed"] else "FAIL"
            for regime, gates in gates_by_regime.items()
        },
        "rule": (
            "FINAL_CONFIRMATION_GO if at least one additivity-authorized "
            "precision regime passes all five development gates on the new "
            "seed-46 split; only passing regimes are authorized for seed-44 "
            "final confirmation. A failing regime is never final-tested. On "
            "MEASURED_DAMAGE_NO_GO the negative result is preserved, seed 44 "
            "stays untouched, and no alternative objective may be searched."
        ),
    }


def stage3_final_regime_assessment(
    statistics: Mapping[tuple[str, float], Mapping[str, MethodStatistics]],
    comparisons_vs_average: Mapping[float, Mapping[str, Any]],
    comparisons_vs_global: Mapping[float, Mapping[str, Any]],
    regime: str,
    budgets: tuple[float, ...],
) -> dict[str, Any]:
    """Apply the frozen Stage 3 final success requirements to one regime."""

    point_wins: dict[str, Any] = {}
    improvements_over_average: list[float] = []
    ci_wins_vs_average = 0
    point_wins_vs_global = 0
    beats_prior_robust_budgets = 0
    beats_random_budgets = 0
    beats_both_simple_budgets = 0
    for budget in budgets:
        methods = statistics[(regime, budget)]
        measured = methods["measured_damage_robust"]
        randoms = [
            item for name, item in methods.items() if name.startswith("random_seed")
        ]
        random_mean_worst = random_baseline_statistics(randoms)[
            "mean_worst_relative_delta"
        ]
        beats_rf = (
            measured.worst_relative_delta
            < methods["robust_functional"].worst_relative_delta
        )
        beats_fragility = (
            measured.worst_relative_delta
            < methods["fragility_robust"].worst_relative_delta
        )
        beats_random = measured.worst_relative_delta < random_mean_worst
        beats_global = (
            measured.worst_relative_delta
            < methods["global_importance"].worst_relative_delta
        )
        beats_average = (
            measured.worst_relative_delta
            < methods["average_specialization"].worst_relative_delta
        )
        all_five = bool(
            beats_rf and beats_fragility and beats_random and beats_global
            and beats_average
        )
        point_wins[str(budget)] = {
            "beats_robust_functional": bool(beats_rf),
            "beats_fragility_robust": bool(beats_fragility),
            "beats_random_mean": bool(beats_random),
            "beats_global_importance": bool(beats_global),
            "beats_average_specialization": bool(beats_average),
            "all_five": all_five,
        }
        beats_prior_robust_budgets += int(beats_rf and beats_fragility)
        beats_random_budgets += int(beats_random)
        beats_both_simple_budgets += int(beats_global and beats_average)
        improvements_over_average.append(
            methods["average_specialization"].worst_relative_delta
            - measured.worst_relative_delta
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
            if statistics[(regime, budget)]["measured_damage_robust"].recovery[domain]
            < 0
        )
        if negative_budgets >= SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS:
            systematic_negative_domains.append(domain)

    all_five_count = sum(1 for value in point_wins.values() if value["all_five"])
    average_improvement = float(np.mean(improvements_over_average))
    requirement_1 = all_five_count >= FINAL_REQUIRED_POINT_BUDGETS
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
    qualified = bool(
        not strong
        and requirement_2
        and requirement_5
        and (
            beats_both_simple_budgets >= QUALIFIED_REQUIRED_POINT_BUDGETS
            or (
                beats_prior_robust_budgets >= FINAL_REQUIRED_POINT_BUDGETS
                and beats_random_budgets >= FINAL_REQUIRED_POINT_BUDGETS
            )
        )
    )
    return {
        "regime": regime,
        "point_wins_by_budget": point_wins,
        "budgets_with_all_five_point_wins": all_five_count,
        "budgets_beating_prior_robust_methods": beats_prior_robust_budgets,
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


def stage3_final_decision(
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
            f"all-five point wins at >= {FINAL_REQUIRED_POINT_BUDGETS} of 4 "
            "budgets (vs Robust-Functional, Fragility-Robust, random mean, "
            "Global-Importance, Average-Specialization), positive average "
            "worst-domain improvement over Average-Specialization, >= "
            f"{FINAL_REQUIRED_CI_WINS_VS_AVERAGE} worst-domain CIs entirely "
            "favoring Measured-Damage-Robust vs Average-Specialization, >= "
            f"{FINAL_REQUIRED_POINT_WINS_VS_GLOBAL} point wins vs "
            "Global-Importance, and no domain with negative recovery at >= "
            f"{SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS} of 4 budgets. SUCCESS "
            "WITH QUALIFICATIONS follows the preregistered qualified rule. "
            "Anything else is a NEGATIVE RESULT; goalposts are never moved."
        ),
    }


def prediction_transfer_check(
    predicted_by_slug: Mapping[str, Mapping[str, float]],
    realized_by_slug: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Descriptive check: does calibration-predicted damage transfer held-out?

    Compares the additive calibration-based predicted delta NLL of every
    evaluated allocation with its realized held-out delta NLL, per domain and
    for the worst domain. Descriptive only; never used to change the optimizer
    or any gate.
    """

    slugs = sorted(set(predicted_by_slug) & set(realized_by_slug))
    if len(slugs) < 3:
        raise ValueError("Transfer check requires at least three allocations")
    predicted = np.asarray(
        [[predicted_by_slug[s][d] for d in STAGE2B_DOMAINS] for s in slugs]
    )
    realized = np.asarray(
        [[realized_by_slug[s][d] for d in STAGE2B_DOMAINS] for s in slugs]
    )
    return {
        "descriptive_only": True,
        "allocations": slugs,
        "spearman_by_domain": {
            domain: safe_spearman(predicted[:, index], realized[:, index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "worst_delta_spearman": safe_spearman(
            predicted.max(axis=1), realized.max(axis=1)
        ),
        "overall_spearman": safe_spearman(predicted.reshape(-1), realized.reshape(-1)),
        "total_predicted_over_realized_ratio": float(
            predicted.sum() / realized.sum() if realized.sum() != 0 else float("nan")
        ),
    }
