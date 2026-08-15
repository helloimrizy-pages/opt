"""Stage 2B metrics, paired bootstrap, development gates, and decisions.

All decision constants here were frozen before any held-out NLL was computed.
The worst-domain metric recomputes its ``max`` over domains inside every
bootstrap replicate rather than freezing the point-estimate worst domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .datasets import DOMAIN_SEED_OFFSETS
from .specialist_preservation import STAGE2B_DOMAINS
from .statistics import safe_spearman

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260815
DEVELOPMENT_BUDGET_FRACTION = 0.20
# Gate C tolerance: 10% of the Average-Specialization degradation magnitude,
# with a preregistered absolute floor for the near-zero case (relative-NLL
# units; 1e-4 corresponds to 0.01% of baseline NLL).
GATE_C_RELATIVE_TOLERANCE = 0.10
GATE_C_ABSOLUTE_TOLERANCE = 1e-4
GATE_D_REQUIRED_POSITIVE_DOMAINS = 3
FINAL_REQUIRED_POINT_BUDGETS = 3
QUALIFIED_REQUIRED_POINT_BUDGETS = 2
FINAL_REQUIRED_CI_WINS_VS_AVERAGE = 2


def build_replicate_indices(
    example_counts: Mapping[str, int],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """One shared index matrix per domain so every comparison is paired."""

    indices: dict[str, np.ndarray] = {}
    for domain in STAGE2B_DOMAINS:
        count = int(example_counts[domain])
        if count < 2:
            raise ValueError(f"Domain {domain} has too few examples to bootstrap")
        rng = np.random.default_rng([seed, DOMAIN_SEED_OFFSETS[domain]])
        indices[domain] = rng.integers(0, count, size=(replicates, count))
    return indices


def _quantile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


@dataclass
class MethodStatistics:
    """Point estimates and bootstrap replicate arrays for one allocation."""

    method: str
    method_label: str
    method_kind: str
    regime: str | None
    budget_fraction: float | None
    relative_delta: dict[str, float]
    relative_delta_ci: dict[str, tuple[float, float]]
    delta_nll: dict[str, float]
    recovery: dict[str, float]
    recovery_ci: dict[str, tuple[float, float]]
    mean_relative_delta: float
    mean_relative_delta_ci: tuple[float, float]
    median_relative_delta: float
    worst_relative_delta: float
    worst_relative_delta_ci: tuple[float, float]
    worst_domain: str
    worst_raw_delta_nll: float
    mean_raw_delta_nll: float
    mean_recovery: float
    min_recovery: float
    replicate_relative: np.ndarray = field(repr=False)
    replicate_worst: np.ndarray = field(repr=False)
    replicate_mean: np.ndarray = field(repr=False)
    replicate_recovery: np.ndarray = field(repr=False)

    def summary_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": self.method,
            "method_label": self.method_label,
            "method_kind": self.method_kind,
            "regime": self.regime,
            "budget_fraction": self.budget_fraction,
            "mean_relative_delta": self.mean_relative_delta,
            "mean_relative_delta_ci_low": self.mean_relative_delta_ci[0],
            "mean_relative_delta_ci_high": self.mean_relative_delta_ci[1],
            "median_relative_delta": self.median_relative_delta,
            "worst_relative_delta": self.worst_relative_delta,
            "worst_relative_delta_ci_low": self.worst_relative_delta_ci[0],
            "worst_relative_delta_ci_high": self.worst_relative_delta_ci[1],
            "worst_domain": self.worst_domain,
            "worst_raw_delta_nll": self.worst_raw_delta_nll,
            "mean_raw_delta_nll": self.mean_raw_delta_nll,
            "mean_recovery": self.mean_recovery,
            "min_recovery": self.min_recovery,
        }
        for domain in STAGE2B_DOMAINS:
            row[f"relative_delta_{domain}"] = self.relative_delta[domain]
            row[f"relative_delta_{domain}_ci_low"] = self.relative_delta_ci[domain][0]
            row[f"relative_delta_{domain}_ci_high"] = self.relative_delta_ci[domain][1]
            row[f"delta_nll_{domain}"] = self.delta_nll[domain]
            row[f"recovery_{domain}"] = self.recovery[domain]
            row[f"recovery_{domain}_ci_low"] = self.recovery_ci[domain][0]
            row[f"recovery_{domain}_ci_high"] = self.recovery_ci[domain][1]
        return row


def compute_method_statistics(
    record: Mapping[str, Any],
    allocation_nll: Mapping[str, np.ndarray],
    bf16_nll: Mapping[str, np.ndarray],
    base_uniform_nll: Mapping[str, np.ndarray],
    replicate_indices: Mapping[str, np.ndarray],
) -> MethodStatistics:
    """Compute all preregistered metrics for one evaluated allocation.

    ``base_uniform_nll`` is the uniform base-precision model of the record's
    regime (uniform 4-bit for 4->8, uniform 3-bit for 3->8).
    """

    relative: dict[str, float] = {}
    relative_ci: dict[str, tuple[float, float]] = {}
    delta: dict[str, float] = {}
    recovery: dict[str, float] = {}
    recovery_ci: dict[str, tuple[float, float]] = {}
    replicates = next(iter(replicate_indices.values())).shape[0]
    replicate_relative = np.zeros((replicates, len(STAGE2B_DOMAINS)))
    replicate_recovery = np.zeros((replicates, len(STAGE2B_DOMAINS)))
    for domain_index, domain in enumerate(STAGE2B_DOMAINS):
        allocation = np.asarray(allocation_nll[domain], dtype=np.float64)
        baseline = np.asarray(bf16_nll[domain], dtype=np.float64)
        base_uniform = np.asarray(base_uniform_nll[domain], dtype=np.float64)
        if allocation.shape != baseline.shape or allocation.shape != base_uniform.shape:
            raise ValueError(f"Per-example NLL arrays are misaligned for {domain}")
        if np.mean(baseline) <= 0:
            raise ValueError(f"BF16 baseline NLL is non-positive for {domain}")
        relative[domain] = float(
            (allocation.mean() - baseline.mean()) / baseline.mean()
        )
        delta[domain] = float(allocation.mean() - baseline.mean())
        recovery[domain] = float(base_uniform.mean() - allocation.mean())
        indices = replicate_indices[domain]
        allocation_reps = allocation[indices].mean(axis=1)
        baseline_reps = baseline[indices].mean(axis=1)
        base_reps = base_uniform[indices].mean(axis=1)
        replicate_relative[:, domain_index] = (
            allocation_reps - baseline_reps
        ) / baseline_reps
        replicate_recovery[:, domain_index] = base_reps - allocation_reps
        relative_ci[domain] = _quantile_ci(replicate_relative[:, domain_index])
        recovery_ci[domain] = _quantile_ci(replicate_recovery[:, domain_index])

    point_relative = np.asarray([relative[d] for d in STAGE2B_DOMAINS])
    point_delta = np.asarray([delta[d] for d in STAGE2B_DOMAINS])
    point_recovery = np.asarray([recovery[d] for d in STAGE2B_DOMAINS])
    replicate_worst = replicate_relative.max(axis=1)
    replicate_mean = replicate_relative.mean(axis=1)
    worst_index = int(point_relative.argmax())
    return MethodStatistics(
        method=record["method"],
        method_label=record["method_label"],
        method_kind=record["method_kind"],
        regime=record["regime"],
        budget_fraction=record["budget_fraction"],
        relative_delta=relative,
        relative_delta_ci=relative_ci,
        delta_nll=delta,
        recovery=recovery,
        recovery_ci=recovery_ci,
        mean_relative_delta=float(point_relative.mean()),
        mean_relative_delta_ci=_quantile_ci(replicate_mean),
        median_relative_delta=float(np.median(point_relative)),
        worst_relative_delta=float(point_relative.max()),
        worst_relative_delta_ci=_quantile_ci(replicate_worst),
        worst_domain=STAGE2B_DOMAINS[worst_index],
        worst_raw_delta_nll=float(point_delta.max()),
        mean_raw_delta_nll=float(point_delta.mean()),
        mean_recovery=float(point_recovery.mean()),
        min_recovery=float(point_recovery.min()),
        replicate_relative=replicate_relative,
        replicate_worst=replicate_worst,
        replicate_mean=replicate_mean,
        replicate_recovery=replicate_recovery,
    )


def paired_comparison(
    first: MethodStatistics, second: MethodStatistics, metric: str
) -> dict[str, Any]:
    """``first - second`` with a CI from the shared-replicate difference.

    Negative differences mean ``first`` degrades less than ``second``.
    """

    if metric == "worst_relative_delta":
        point = first.worst_relative_delta - second.worst_relative_delta
        difference = first.replicate_worst - second.replicate_worst
    elif metric == "mean_relative_delta":
        point = first.mean_relative_delta - second.mean_relative_delta
        difference = first.replicate_mean - second.replicate_mean
    else:
        raise ValueError(f"Unsupported comparison metric {metric!r}")
    low, high = _quantile_ci(difference)
    return {
        "first": first.method,
        "second": second.method,
        "regime": first.regime,
        "budget_fraction": first.budget_fraction,
        "metric": metric,
        "difference": float(point),
        "difference_ci_low": low,
        "difference_ci_high": high,
        "favors_first": bool(point < 0),
        "ci_excludes_zero": bool(high < 0 or low > 0),
        "ci_favors_first": bool(high < 0),
    }


def random_baseline_statistics(
    randoms: list[MethodStatistics],
) -> dict[str, Any]:
    worst = np.asarray([item.worst_relative_delta for item in randoms])
    mean = np.asarray([item.mean_relative_delta for item in randoms])
    return {
        "count": len(randoms),
        "mean_worst_relative_delta": float(worst.mean()),
        "std_worst_relative_delta": float(worst.std(ddof=1)) if len(randoms) > 1 else 0.0,
        "best_random_worst_relative_delta": float(worst.min()),
        "worst_random_worst_relative_delta": float(worst.max()),
        "mean_mean_relative_delta": float(mean.mean()),
        "individual_worst_relative_delta": {
            item.method: item.worst_relative_delta for item in randoms
        },
        "individual_mean_relative_delta": {
            item.method: item.mean_relative_delta for item in randoms
        },
    }


def mean_random_replicate_worst(randoms: list[MethodStatistics]) -> np.ndarray:
    return np.stack([item.replicate_worst for item in randoms], axis=0).mean(axis=0)


def development_gates(
    robust: MethodStatistics,
    randoms: list[MethodStatistics],
    global_importance: MethodStatistics,
    average_specialization: MethodStatistics,
) -> dict[str, Any]:
    """The four preregistered development gates for one precision regime."""

    random_stats = random_baseline_statistics(randoms)
    gate_a = {
        "name": "better_than_random",
        "robust_worst_relative_delta": robust.worst_relative_delta,
        "random_mean_worst_relative_delta": random_stats["mean_worst_relative_delta"],
        "passed": bool(
            robust.worst_relative_delta < random_stats["mean_worst_relative_delta"]
        ),
    }
    gate_b = {
        "name": "better_than_non_robust_preservation",
        "robust_worst_relative_delta": robust.worst_relative_delta,
        "global_importance_worst_relative_delta": global_importance.worst_relative_delta,
        "average_specialization_worst_relative_delta": (
            average_specialization.worst_relative_delta
        ),
        "passed": bool(
            robust.worst_relative_delta < global_importance.worst_relative_delta
            and robust.worst_relative_delta
            < average_specialization.worst_relative_delta
        ),
    }
    average_magnitude = abs(average_specialization.mean_relative_delta)
    tolerance = max(
        GATE_C_RELATIVE_TOLERANCE * average_magnitude, GATE_C_ABSOLUTE_TOLERANCE
    )
    gate_c = {
        "name": "no_catastrophic_mean_tradeoff",
        "robust_mean_relative_delta": robust.mean_relative_delta,
        "average_specialization_mean_relative_delta": (
            average_specialization.mean_relative_delta
        ),
        "relative_tolerance": GATE_C_RELATIVE_TOLERANCE,
        "absolute_tolerance_floor": GATE_C_ABSOLUTE_TOLERANCE,
        "tolerance_used": tolerance,
        "absolute_floor_applied": bool(
            GATE_C_ABSOLUTE_TOLERANCE > GATE_C_RELATIVE_TOLERANCE * average_magnitude
        ),
        "passed": bool(
            robust.mean_relative_delta
            <= average_specialization.mean_relative_delta + tolerance
        ),
    }
    positive_recovery = [
        domain for domain in STAGE2B_DOMAINS if robust.recovery[domain] > 0
    ]
    gate_d = {
        "name": "broad_recovery",
        "recovery_by_domain": dict(robust.recovery),
        "positive_recovery_domains": positive_recovery,
        "required_positive_domains": GATE_D_REQUIRED_POSITIVE_DOMAINS,
        "passed": bool(len(positive_recovery) >= GATE_D_REQUIRED_POSITIVE_DOMAINS),
    }
    gates = {"gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c, "gate_d": gate_d}
    gates["all_passed"] = bool(all(gates[key]["passed"] for key in
                                   ("gate_a", "gate_b", "gate_c", "gate_d")))
    gates["random_baseline_statistics"] = random_stats
    return gates


def development_decision(gates_by_regime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passing = [
        regime for regime, gates in gates_by_regime.items() if gates["all_passed"]
    ]
    decision = "FULL_EVALUATION_GO" if passing else "ROBUST_PRESERVATION_NO_GO"
    return {
        "decision": decision,
        "passing_regimes": passing,
        "regimes_evaluated": list(gates_by_regime),
        "rule": (
            "FULL_EVALUATION_GO if at least one precision regime passes all four "
            "development gates; the gate is a compute-saving rule, not the final "
            "hypothesis test, and never modifies the method"
        ),
    }


def final_regime_assessment(
    statistics: Mapping[tuple[str, float], Mapping[str, MethodStatistics]],
    comparisons_vs_average: Mapping[float, Mapping[str, Any]],
    regime: str,
    budgets: tuple[float, ...],
) -> dict[str, Any]:
    """Apply the frozen final success conditions to one precision regime.

    ``statistics[(regime, budget)]`` maps method name to MethodStatistics;
    ``comparisons_vs_average[budget]`` is the Robust-Functional minus
    Average-Specialization worst-domain comparison at that budget.
    """

    point_wins = {}
    improvements_over_average = []
    ci_wins = 0
    catastrophic_domains = []
    for budget in budgets:
        methods = statistics[(regime, budget)]
        robust = methods["robust_functional"]
        randoms = [
            item for name, item in methods.items() if name.startswith("random_seed")
        ]
        random_mean_worst = random_baseline_statistics(randoms)[
            "mean_worst_relative_delta"
        ]
        beats_random = robust.worst_relative_delta < random_mean_worst
        beats_global = (
            robust.worst_relative_delta
            < methods["global_importance"].worst_relative_delta
        )
        beats_average = (
            robust.worst_relative_delta
            < methods["average_specialization"].worst_relative_delta
        )
        point_wins[str(budget)] = {
            "beats_random_mean": bool(beats_random),
            "beats_global_importance": bool(beats_global),
            "beats_average_specialization": bool(beats_average),
            "all_three": bool(beats_random and beats_global and beats_average),
        }
        improvements_over_average.append(
            methods["average_specialization"].worst_relative_delta
            - robust.worst_relative_delta
        )
        if comparisons_vs_average[budget]["ci_favors_first"]:
            ci_wins += 1
    for domain in STAGE2B_DOMAINS:
        always_negative = all(
            statistics[(regime, budget)]["robust_functional"].recovery[domain] < 0
            and statistics[(regime, budget)]["robust_functional"].recovery_ci[domain][1]
            < 0
            for budget in budgets
        )
        if always_negative:
            catastrophic_domains.append(domain)
    all_three_count = sum(
        1 for value in point_wins.values() if value["all_three"]
    )
    average_improvement = float(np.mean(improvements_over_average))
    return {
        "regime": regime,
        "point_wins_by_budget": point_wins,
        "budgets_with_all_three_point_wins": all_three_count,
        "average_improvement_over_average_specialization": average_improvement,
        "ci_wins_vs_average_specialization": ci_wins,
        "catastrophic_domains": catastrophic_domains,
        "strong_success": bool(
            all_three_count >= FINAL_REQUIRED_POINT_BUDGETS
            and average_improvement > 0
            and ci_wins >= FINAL_REQUIRED_CI_WINS_VS_AVERAGE
            and not catastrophic_domains
        ),
        "qualified_success": bool(
            all_three_count >= QUALIFIED_REQUIRED_POINT_BUDGETS
            and average_improvement > 0
            and not catastrophic_domains
        ),
    }


def final_decision(regime_assessments: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
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
            "STRONG SUCCESS requires, in at least one regime: all-three point wins "
            f"at >= {FINAL_REQUIRED_POINT_BUDGETS} of 4 budgets, positive average "
            "worst-domain improvement over Average-Specialization, >= "
            f"{FINAL_REQUIRED_CI_WINS_VS_AVERAGE} budget comparisons with 95% CIs "
            "favoring Robust-Functional, and no domain with systematically "
            "catastrophic recovery. SUCCESS WITH QUALIFICATIONS requires all-three "
            f"point wins at >= {QUALIFIED_REQUIRED_POINT_BUDGETS} budgets with "
            "positive average improvement in some regime. Anything else is a "
            "NEGATIVE RESULT."
        ),
    }


def coverage_recovery_diagnostic(
    statistics: Mapping[str, MethodStatistics],
    coverage_min_by_method: Mapping[str, float],
) -> dict[str, Any]:
    """Diagnostic-only Spearman between minimum coverage and empirical outcome."""

    methods = [
        name
        for name in statistics
        if name in coverage_min_by_method
    ]
    coverage = np.asarray([coverage_min_by_method[name] for name in methods])
    worst = np.asarray([statistics[name].worst_relative_delta for name in methods])
    min_recovery = np.asarray([statistics[name].min_recovery for name in methods])
    return {
        "methods": methods,
        "spearman_min_coverage_vs_negative_worst_relative_delta": safe_spearman(
            coverage, -worst
        ),
        "spearman_min_coverage_vs_min_recovery": safe_spearman(coverage, min_recovery),
        "diagnostic_only": True,
        "not_a_fitted_model": True,
    }
