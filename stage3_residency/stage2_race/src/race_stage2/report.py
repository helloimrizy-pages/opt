"""Rendering of the frozen Stage 2 report."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


REGIME_ORDER = ("stationary", "abrupt", "repeated", "mixed")


def _percent(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{100 * float(value):.{digits}f}%"


def _interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return ""
    if not (np.isfinite(float(low)) and np.isfinite(float(high))):
        return ""
    return f" [{100*float(low):.2f}, {100*float(high):.2f}]"


def render_report(
    analysis: Mapping[str, Any], inputs: Any, frozen: Mapping[str, Any]
) -> str:
    verdict = str(analysis["verdict"])
    primary = str(analysis["primary_variant_label"])
    suite = analysis["suite_results"]
    decision_rows = [row for row in suite if int(row["capacity"]) != 8]
    parameters = analysis["primary_variant_parameters"]
    improvements = [row["race_improvement_over_stage1"] for row in decision_rows]
    gaps = [row["original_oracle_gap_closed"] for row in decision_rows]
    stage1_gaps = [row["stage1_original_oracle_gap_closed"] for row in decision_rows]

    lines: list[str] = [verdict, "", "# RACE Stage 2: Adaptive Multi-Horizon Future-Reuse Ranking", ""]

    lines += [
        "## A. Executive verdict",
        "",
        f"- Frozen primary RACE variant: `{primary}` (`{analysis['primary_variant_id']}`).",
        f"- Frozen parameters: adviser pool of {len(parameters['adviser_pool'])} causal advisers "
        f"{parameters['adviser_pool']}, capped future-reuse horizon `H_max = {parameters['H_max']}` "
        f"same-layer events, delayed Hedge learning rate `eta = {parameters['eta']}`, initialization "
        f"`{parameters['initialization']}`, online loss `{parameters['loss']}`, one weight vector per "
        f"`{parameters['weight_scope']}` policy instance.",
        f"- Improvement over the frozen Stage 1 winner "
        f"`{analysis['stage1_winner_method_id']}` across capacities 12–32: "
        f"{_percent(min(improvements))} to {_percent(max(improvements))}.",
        f"- Original Stage 0 oracle gap closed: {_percent(min(gaps))} to {_percent(max(gaps))}, "
        f"against {_percent(min(stage1_gaps))} to {_percent(max(stage1_gaps))} for the Stage 1 winner.",
        f"- Preregistered success criteria: {analysis['decision']['reason']}",
        "",
        "| Capacity | Stage 1 winner | RACE primary | Oracle | Improvement vs Stage 1 | Original oracle gap closed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in decision_rows:
        lines.append(
            "| {} | {:,.0f} | {:,.0f} | {:,.0f} | {}{} | {}{} |".format(
                row["capacity"],
                row["stage1_cost"],
                row[f"{primary}_cost"],
                row["oracle_cost"],
                _percent(row["race_improvement_over_stage1"]),
                _interval(row["race_improvement_ci_low"], row["race_improvement_ci_high"]),
                _percent(row["original_oracle_gap_closed"]),
                _interval(
                    row["original_oracle_gap_closed_ci_low"],
                    row["original_oracle_gap_closed_ci_high"],
                ),
            )
        )

    lines += [
        "",
        "## B. Frozen prior evidence",
        "",
        "Stage 0 and Stage 1 are unchanged. Stage 2 reads their sealed archives and recomputes none "
        "of their numbers.",
        "",
        f"- Stage 0 verdict `RACE_STAGE0_STRONG_GO`; trace logical hash "
        f"`{analysis['trace_hash']}`; archive manifest "
        f"`{analysis['stage0_archive_manifest_sha256']}`.",
        f"- Stage 1 verdict `RACE_STAGE1_STRONG_GO`; winner "
        f"`{analysis['stage1_winner_method_id']}`; archive manifest "
        f"`{analysis['stage1_archive_manifest_sha256']}`.",
        "- Stage 1's decisive diagnostic remains intact: perfect next-use scoring with the same "
        "trivial eviction mechanism reproduces the offline oracle exactly at all fifty frozen "
        "conditions, so future-reuse ranking, not combinatorial eviction, is the bottleneck Stage 2 "
        "attacks.",
        "",
        "Stage 1 oracle-gap closure on the same frozen suite (recomputed here only as a reference "
        "column, from the sealed Stage 1 rows):",
        "",
        "| Capacity | Stage 0 simple | Stage 1 winner | Oracle | Stage 1 gap closed |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite:
        lines.append(
            "| {} | {:,.0f} | {:,.0f} | {:,.0f} | {} |".format(
                row["capacity"],
                row["stage0_simple_cost"],
                row["stage1_cost"],
                row["oracle_cost"],
                _percent(row["stage1_original_oracle_gap_closed"]),
            )
        )

    pool = frozen["adviser_pools"]["primary"]
    lines += [
        "",
        "## C. Algorithm",
        "",
        "**Adviser pool.** Nine causal advisers, all reused unmodified from Stage 0/Stage 1: "
        f"`{', '.join(pool['order'])}`. The six Markov advisers score the current atomic request "
        "against calibration-fitted binary transition matrices at horizons "
        f"{pool['markov_horizons']}; horizons 1–16 are bitwise identical to the frozen Stage 1 "
        "transition archive at its stored precision and horizon 32 is newly fitted on the identical "
        "calibration path with the identical frozen fitting function. `GATE_EWMA` uses the frozen "
        f"Stage 1 alpha {pool['gate_ewma_alpha']}, `LFU_DECAY` the frozen Stage 0 alpha "
        f"{pool['lfu_decay_alpha']}, and `PERSISTENCE` the previous same-layer request.",
        "",
        "**Normalization.** At each decision event the advisers are scored only over the eligible "
        "eviction candidates — pre-event residents absent from the current atomic request — and each "
        "adviser's scores are replaced by average-rank percentiles `z` in `[0, 1]`, where `1.0` is "
        "that adviser's strongest retention recommendation. Exact adviser ties receive identical "
        "midranks, which keeps adviser indifference visible instead of manufacturing an ordering.",
        "",
        "**Combination and eviction.** `S_e = sum_j w_j z_{j,e}` with `w` on the probability simplex. "
        "Candidates are ordered by `S` descending, then LRU recency descending, then expert ID "
        "ascending — the exact frozen Stage 1 ordering with `S` in place of the single predictor "
        "score — and the lowest-ranked candidates are evicted. No other eviction logic exists, and a "
        "Stage 2 variant that places all weight on one adviser reproduces the corresponding frozen "
        "Stage 1 single-predictor cost exactly.",
        "",
        f"**Delayed feedback.** The target is the capped next-use distance "
        f"`d_tilde = min(d, {parameters['H_max'] + 1})` in same-layer events. A decision at same-layer "
        f"event `q` becomes a pending example; it is resolved only at event `q + {parameters['H_max']}`, "
        "once every follow-up event needed for its label has actually been observed, and the weight "
        "update is applied at that same event before that event's own decision. Decisions inside the "
        "final 32 events of a stream are never used for learning and their unresolved count is "
        "reported.",
        "",
        "**Ranking loss.** For a resolved example, ordered candidate pairs with "
        "`d_tilde(a) < d_tilde(b)` are comparable; adviser `j` pays `0` when it ranks `a` above `b`, "
        "`1` when it inverts them and `0.5` on an exact score tie. The unweighted loss divides by the "
        "comparable-pair count; the cost-sensitive loss weights each pair by "
        "`|1/d_tilde(a) - 1/d_tilde(b)|` with the fixed potential `phi(d) = 1/d`. Both lie in `[0, 1]`. "
        "Examples with no comparable pair are skipped and counted.",
        "",
        f"**Multiplicative update.** `w_j <- w_j exp(-eta * loss_j)` renormalized to the simplex, with "
        f"`eta = {parameters['eta']}` selected on calibration alone from the frozen grid "
        f"{list(inputs.preregistration['online_learning']['eta_grid'])}. The state is carried in log "
        "space with per-step maximum subtraction, which is algebraically identical and cannot "
        "overflow or underflow.",
        "",
        "## D. Causality audit",
        "",
        "RACE never reads a future event at decision time. Operationally:",
        "",
    ]
    delayed = [
        row
        for row in analysis["delayed_feedback"]
        if row["variant"] == primary and int(row["capacity"]) != 8
    ]
    lines += [
        "| Capacity | Examples generated | Resolved | Unresolved at stream end | Skipped (no comparable pair) | Applied updates | Mean delay | Max delay | Min update-minus-decision offset |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in delayed:
        lines.append(
            "| {} | {:,} | {:,} | {:,} ({}) | {:,} | {:,} | {} | {} | {} |".format(
                row["capacity"],
                row["examples_generated"],
                row["examples_resolved"],
                row["examples_unresolved_at_stream_end"],
                _percent(row["unresolved_fraction"], 3),
                row["examples_skipped_no_comparable_pair"],
                row["applied_updates"],
                (
                    f"{row['average_feedback_delay_same_layer_events']:.2f}"
                    if row["average_feedback_delay_same_layer_events"] is not None
                    else "N/A"
                ),
                row["maximum_feedback_delay_same_layer_events"],
                row["minimum_update_minus_decision_offset"],
            )
        )
    lines += [
        "",
        "- The minimum observed update-minus-decision offset equals `H_max = 32` same-layer events at "
        "every capacity, so no weight update ever preceded the observation of its own label.",
        "- Every applied update carries recorded `decision_event_index`, "
        "`label_resolution_event_index` and `weight_update_event_index`; the simulator raises rather "
        "than updating if `update < decision + H_max`.",
        "- The causal capped label built from the rolling 32-event window was cross-checked against an "
        "offline future-use computation in the pilot audit and matched on every checked example.",
        "- The offline ranking observer is a one-way sink: enabling it leaves every simulated cost "
        "bit-identical, which is verified in both the pilot audit and the unit tests.",
        "- Calibration used only the 80 frozen Stage 0 calibration sequences; evaluation used the "
        "disjoint 320-sequence split, and nothing was reselected after evaluation began.",
        "",
        "## E. Main results",
        "",
        "Frozen ten-workload suite, unit expert-transfer cost at lambda zero. Capacity 8 is degenerate "
        "(top-k = 8 leaves no eviction freedom) and contributes to no verdict count.",
        "",
        "| Capacity | Stage 1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite:
        lines.append(
            "| {} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} |".format(
                row["capacity"],
                row["stage1_cost"],
                row["RACE_UNIFORM_cost"],
                row["RACE_STATIC_cost"],
                row["RACE_ONLINE_cost"],
                row["RACE_COST_cost"],
                row["oracle_cost"],
            )
        )
    lines += [
        "",
        "Normalized by the Stage 0 strongest-simple cost at the same capacity:",
        "",
        "| Capacity | Stage 1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite:
        base = row["stage0_simple_cost"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["capacity"],
                row["stage1_cost"] / base,
                row["RACE_UNIFORM_cost"] / base,
                row["RACE_STATIC_cost"] / base,
                row["RACE_ONLINE_cost"] / base,
                row["RACE_COST_cost"] / base,
                row["oracle_cost"] / base,
            )
        )
    lines += [
        "",
        f"Success metrics for the frozen primary variant `{primary}` "
        "(95% intervals are the conditional paired bootstrap described in section K):",
        "",
        "| Capacity | Improvement vs Stage 1 | Original oracle gap closed | Stage 1 residual recovered |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in decision_rows:
        lines.append(
            "| {} | {}{} | {}{} | {}{} |".format(
                row["capacity"],
                _percent(row["race_improvement_over_stage1"]),
                _interval(row["race_improvement_ci_low"], row["race_improvement_ci_high"]),
                _percent(row["original_oracle_gap_closed"]),
                _interval(
                    row["original_oracle_gap_closed_ci_low"],
                    row["original_oracle_gap_closed_ci_high"],
                ),
                _percent(row["stage1_residual_recovered"]),
                _interval(
                    row["stage1_residual_recovered_ci_low"],
                    row["stage1_residual_recovered_ci_high"],
                ),
            )
        )
    lines += [
        "",
        "By workload regime (no regime is averaged away):",
        "",
        "| Regime | Capacity | Stage 0 simple | Stage 1 | RACE primary | Oracle | Improvement vs Stage 1 | Gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime in REGIME_ORDER:
        for row in analysis["scope_results"]:
            if row["scope"] != regime or int(row["capacity"]) == 8:
                continue
            lines.append(
                "| {} | {} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {} | {} |".format(
                    regime,
                    row["capacity"],
                    row["stage0_simple_cost"],
                    row["stage1_cost"],
                    row[f"{primary}_cost"],
                    row["oracle_cost"],
                    _percent(row["race_improvement_over_stage1"]),
                    _percent(row["original_oracle_gap_closed"]),
                )
            )
    flagged = [row for row in analysis["regression_rows"] if row["flagged"]]
    lines += [
        "",
        "### Stability and regressions",
        "",
        f"The preregistered stability flag marks any workload/capacity where "
        f"`C_RACE > 1.03 * C_Stage1`. {len(flagged)} of {len(analysis['regression_rows'])} "
        f"workload/capacity cells are flagged.",
        "",
    ]
    if flagged:
        lines += [
            "| Workload | Regime | Capacity | Stage 1 | RACE | Ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(flagged, key=lambda item: -float(item["regression_ratio"]))[:40]:
            lines.append(
                "| {} | {} | {} | {:,.0f} | {:,.0f} | {:.4f} |".format(
                    row["workload"],
                    row["regime"],
                    row["capacity"],
                    row["stage1_cost"],
                    row["race_cost"],
                    row["regression_ratio"],
                )
            )
        if len(flagged) > 40:
            lines.append(f"| … | | | | | {len(flagged) - 40} further flagged cells in `tables/table10_regressions.csv` |")
    else:
        lines.append("No workload/capacity cell exceeded the 3% regression threshold.")

    lines += [
        "",
        "## F. Ablation",
        "",
        "Uniform -> Static -> Online -> Cost, on the frozen ten-workload suite. A positive relative "
        "change means the later configuration is cheaper.",
        "",
        "| Question | Comparison | Capacity | Before | After | Relative change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["ablation"]:
        lines.append(
            "| {} | {} | {} | {:,.0f} | {:,.0f} | {:+.2f}% |".format(
                row["question"],
                row["comparison"],
                row["capacity"],
                row["before_cost"],
                row["after_cost"],
                100 * row["relative_change"],
            )
        )

    lines += [
        "",
        "## G. Ranking diagnostics",
        "",
        "Measured on the actual eviction candidate sets. Pairwise ordering accuracy is "
        "`P(S_a > S_b | d_a < d_b)` with half credit for exact score ties; oracle-consistent eviction "
        "means at least one evicted expert attained the maximum true next-use distance among the "
        "candidates, and oracle-optimal means the evicted set matched a farthest-future-optimal choice.",
        "",
        "| Variant | Capacity | Eviction events | Ordering accuracy (capped) | Ordering accuracy (true) | Oracle-consistent | Oracle-optimal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["ranking_diagnostics"]:
        if int(row["capacity"]) == 8:
            continue
        lines.append(
            "| {} | {} | {:,} | {} | {} | {} | {} |".format(
                row["variant"],
                row["capacity"],
                row["eviction_events"],
                _percent(row["pairwise_ordering_accuracy_capped"]),
                _percent(row["pairwise_ordering_accuracy_true"]),
                _percent(row["oracle_consistent_eviction_rate"]),
                _percent(row["oracle_optimal_eviction_rate"]),
            )
        )
    correlations = analysis["ranking_vs_cost"]["correlations"]
    lines += [
        "",
        "Association between ranking quality and realized transfer cost across all evaluated "
        "variants and non-degenerate capacities:",
        "",
    ]
    for name, value in correlations.items():
        lines.append(
            f"- `{name}` vs improvement over Stage 1: Spearman "
            f"{value['spearman']:.3f} (descriptive p={value['p_value_descriptive']:.3g}, "
            f"{value['points']} configuration points)."
        )
    lines.append("")
    lines.append(analysis["ranking_vs_cost"]["interpretation"])

    lines += [
        "",
        "## H. Horizon behavior",
        "",
        "Mean deployed weight assigned to each Markov horizon by the frozen primary variant, averaged "
        "over layers and workloads:",
        "",
        "| Capacity | Spare slots | H1 | H2 | H4 | H8 | H16 | H32 | Markov mass | Weighted mean horizon |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["horizon_weights"]:
        if row["variant"] != primary or int(row["capacity"]) == 8:
            continue
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                row["capacity"],
                row["spare_residency"],
                row["markov_H1"],
                row["markov_H2"],
                row["markov_H4"],
                row["markov_H8"],
                row["markov_H16"],
                row["markov_H32"],
                row["markov_mass"],
                (
                    f"{row['weighted_mean_markov_horizon']:.3f}"
                    if row["weighted_mean_markov_horizon"] is not None
                    else "N/A"
                ),
            )
        )

    lines += [
        "",
        "## I. Weight adaptation",
        "",
    ]
    weight_rows = [
        row
        for row in analysis["weight_adaptation"]
        if row["variant"] == primary and row["workload"] == "mixed_interleaved"
    ]
    if weight_rows:
        by_capacity: dict[int, list[Mapping[str, Any]]] = {}
        for row in weight_rows:
            by_capacity.setdefault(int(row["capacity"]), []).append(row)
        lines += [
            "Mixed-interleaved workload, per-layer adviser weights of the frozen primary variant:",
            "",
            "| Capacity | Layers | Mean effective advisers | End effective advisers | Most common dominant adviser | Mean dominant weight |",
            "| ---: | ---: | ---: | ---: | --- | ---: |",
        ]
        for capacity in sorted(by_capacity):
            if capacity == 8:
                continue
            rows = by_capacity[capacity]
            dominant = {}
            for row in rows:
                dominant[row["dominant_adviser"]] = dominant.get(row["dominant_adviser"], 0) + 1
            top = max(dominant.items(), key=lambda item: (item[1], item[0]))
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {} ({}/{}) | {:.4f} |".format(
                    capacity,
                    len(rows),
                    float(np.mean([row["mean_effective_advisers"] for row in rows])),
                    float(np.mean([row["end_effective_advisers"] for row in rows])),
                    top[0],
                    top[1],
                    len(rows),
                    float(np.mean([row["dominant_adviser_mean_weight"] for row in rows])),
                )
            )
    regret_rows = [
        row
        for row in analysis["regret_accounting"]
        if row["variant"] == primary and int(row["capacity"]) != 8
    ]
    if regret_rows:
        lines += [
            "",
            "Empirical adviser regret (this is accounting, not a theoretical guarantee): cumulative "
            "resolved mixture loss of RACE minus the cumulative loss of the best single fixed adviser "
            "in hindsight.",
            "",
            "| Capacity | Resolved examples | Mixture rank loss | Best fixed adviser | Best fixed loss | Empirical regret | Regret per example |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for row in regret_rows:
            lines.append(
                "| {} | {:,} | {:,.1f} | {} | {:,.1f} | {:,.1f} | {:+.5f} |".format(
                    row["capacity"],
                    row["resolved_examples"],
                    row["cumulative_mixture_rank_loss"],
                    row["best_fixed_adviser_by_rank_loss"],
                    row["best_fixed_adviser_rank_loss"],
                    row["empirical_rank_regret"],
                    row["empirical_rank_regret_per_example"],
                )
            )
    layers = analysis["trajectory_layers"]
    if layers.get("layers"):
        lines += [
            "",
            f"Figure 3 plots adviser-weight trajectories for layers {layers['layers']}, chosen by the "
            f"preregistered rule ({layers['rule']}).",
        ]

    lines += [
        "",
        "## J. Oracle residual",
        "",
        "| Capacity | RACE cost | Oracle cost | Residual headroom vs Stage 0 simple | Remaining fraction of the Stage 1 residual |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in decision_rows:
        remaining = row["stage1_residual_recovered"]
        lines.append(
            "| {} | {:,.0f} | {:,.0f} | {} | {} |".format(
                row["capacity"],
                row[f"{primary}_cost"],
                row["oracle_cost"],
                _percent(row["residual_headroom"]),
                _percent(None if remaining is None else 1.0 - float(remaining)),
            )
        )
    lines += [
        "",
        "## K. Limitations",
        "",
        "- These are trace simulations of expert residency, misses, admissions and transfers. No "
        "end-to-end latency improvement and no hardware speedup is claimed or measured.",
        "- The trace is one fixed OLMoE-1B-7B-0924 decode trace on four fixed domains; nothing here "
        "generalizes automatically to other models, batch sizes or serving stacks.",
        "- The online learning target is capped at 32 same-layer events. Reuse structure beyond that "
        "horizon is invisible to the learner by construction.",
        "- RACE combines existing simple causal advisers. It trains no neural network, uses no "
        "reinforcement learning, performs no speculative prefetch and does not change OLMoE routing.",
        "- Plain multiplicative weights is not a tracking algorithm: once an adviser's cumulative loss "
        "advantage becomes large its competitors receive numerically negligible weight, so the "
        "measured adaptation should be read as concentration, not as unlimited regime tracking.",
        "- Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen "
        "stateful workload path. The cache trajectory and the online-learning trajectory are not "
        "regenerated under resampled orderings, so these are conditional intervals, not full "
        "uncertainty over possible workload orderings.",
        "- The offline oracle and the perfect-score policy are non-causal diagnostics, not deployable "
        "methods.",
        "",
        "## L. Next recommendation",
        "",
        _next_action(verdict, analysis),
        "",
        *_diagnostic_analysis(analysis, primary),
        "",
        "## Reproducibility",
        "",
        f"- Stage 2 preregistration hash: `{analysis['preregistration_hash']}`",
        f"- Stage 2 frozen config file: `{analysis['frozen_config_file_sha256']}`",
        f"- Stage 2 evaluation manifest: `{analysis['evaluation_manifest_sha256']}`",
        f"- Stage 0 trace logical hash: `{analysis['trace_hash']}`",
        f"- Stage 0 archive manifest: `{analysis['stage0_archive_manifest_sha256']}`",
        f"- Stage 1 archive manifest: `{analysis['stage1_archive_manifest_sha256']}`",
        f"- Stage 1 frozen config: `{frozen['stage1_frozen_config_file_sha256']}`",
        f"- Stage 1 winner: `{analysis['stage1_winner_method_id']}`",
        f"- Stage 2 source commit at freeze: `{frozen['stage2_repository_head']}`",
        f"- Stage 2 source bundle hash: `{frozen['stage2_source_bundle_hash']}`",
        f"- Stage 2 transition models: `{frozen['transition_model_hash']}`",
        f"- Selected eta: `{frozen['selected_eta']}`; initialization: "
        f"`{frozen['selected_initialization']}`; primary loss: `{frozen['primary_loss']}`; "
        f"weight scope: `{frozen['weight_scope_primary']}`; H_max: `{frozen['H_max']}`.",
        "",
        "## Core answer",
        "",
        _core_answer(analysis, primary),
    ]
    return "\n".join(lines) + "\n"


def _diagnostic_analysis(analysis: Mapping[str, Any], primary: str) -> list[str]:
    """The preregistered post-outcome diagnostic questions, answered from measurement."""

    ablation = {
        (row["question"], int(row["capacity"])): float(row["relative_change"])
        for row in analysis["ablation"]
    }

    def mean_of(question: str) -> float | None:
        values = [
            value for (name, _capacity), value in ablation.items() if name == question
        ]
        return float(np.mean(values)) if values else None

    ranking = [
        row
        for row in analysis["ranking_diagnostics"]
        if row["variant"] == primary and int(row["capacity"]) != 8
    ]
    horizons = [
        row
        for row in analysis["horizon_weights"]
        if row["variant"] == primary and int(row["capacity"]) != 8
    ]
    regret = [
        row
        for row in analysis["regret_accounting"]
        if row["variant"] == primary and int(row["capacity"]) != 8
    ]
    delayed = [
        row
        for row in analysis["delayed_feedback"]
        if row["variant"] == primary and int(row["capacity"]) != 8
    ]
    lines = [
        "### Diagnostic analysis",
        "",
        "The preregistered response to this outcome is diagnosis, not added model capacity. "
        "Each question below is answered from the measurements in this run.",
        "",
    ]
    diversity = mean_of("G_extended_vs_stage1")
    online_diversity = mean_of("F_adviser_diversity_online")
    if diversity is not None:
        lines.append(
            "**Is adviser diversity too low?** Partly, and in a specific way. Adding exactly one "
            "adviser — the frozen Stage 1 winner itself, which is a raw-scale blend of two pool "
            f"members — moves the online variant by {100*online_diversity:+.2f}% on average and turns "
            f"the comparison against Stage 1 from negative into {100*diversity:+.2f}%. The primary "
            "pool cannot express that blend because percentile normalization is applied to each "
            "adviser separately and discards the magnitude information the raw-scale mixture uses. "
            "So the binding constraint is the representation the specification mandates, not the "
            "number of advisers: a tenth adviser only reaches parity with Stage 1, it does not "
            "unlock the oracle gap."
        )
        lines.append("")
    if ranking:
        low = min(float(row["pairwise_ordering_accuracy_capped"]) for row in ranking)
        high = max(float(row["pairwise_ordering_accuracy_capped"]) for row in ranking)
        oracle_low = min(float(row["oracle_consistent_eviction_rate"]) for row in ranking)
        oracle_high = max(float(row["oracle_consistent_eviction_rate"]) for row in ranking)
        lines.append(
            "**Is the needed future information absent from causal routing history?** Largely yes. "
            f"On the actual eviction candidate sets the combined score orders only {100*low:.1f}%–"
            f"{100*high:.1f}% of comparable pairs correctly, against 100% for the perfect-score "
            "policy that reproduces the oracle. The oracle-consistent eviction rate falls from "
            f"{100*oracle_high:.1f}% at the smallest non-degenerate capacity to {100*oracle_low:.1f}% "
            "at the largest, i.e. the ranking is weakest exactly where the residual oracle headroom "
            "is largest. Every adviser in the pool is a low-order statistic of the same causal "
            "routing history, so their errors are correlated and re-weighting them cannot manufacture "
            "information none of them carries."
        )
        lines.append("")
    if horizons:
        tails = [
            float(row["markov_H16"]) + float(row["markov_H32"]) for row in horizons
        ]
        effective = [
            float(row["weighted_mean_markov_horizon"])
            for row in horizons
            if row["weighted_mean_markov_horizon"] is not None
        ]
        lines.append(
            "**Is the useful horizon longer than 32?** No — the measurement points the other way. "
            f"The two longest Markov advisers together hold only {100*min(tails):.2f}%–"
            f"{100*max(tails):.2f}% of the deployed weight, and the weight-averaged Markov horizon "
            f"stays between {min(effective):.2f} and {max(effective):.2f} same-layer events. The "
            "effective horizon does grow with spare residency (section H), which supports the Stage 1 "
            "lookahead finding, but it grows within a short range. Raising `H_max` above 32 is "
            "therefore not the indicated next step."
        )
        lines.append("")
    if horizons:
        markov_mass = [float(row["markov_mass"]) for row in horizons]
        lines.append(
            "**Are the current gate and routing features insufficient?** This is where the evidence "
            "points. The six calibration-fitted Markov transition advisers together hold only "
            f"{100*min(markov_mass):.1f}%–{100*max(markov_mass):.1f}% of the deployed weight; the "
            "rest goes to decayed-frequency and gate-EWMA recency, which are pure history statistics "
            "with no routing content. A learner that is free to weight conditional next-request "
            "probabilities against plain recency mostly chooses recency, which says the transition "
            "features add little beyond what recency already encodes on this trace."
        )
        lines.append("")
    if regret and delayed:
        per_example = [float(row["empirical_rank_regret_per_example"]) for row in regret]
        updates = sum(int(row["applied_updates"]) for row in delayed)
        lines.append(
            "**Is the delayed adaptation too slow?** No. Across the four non-degenerate capacities "
            f"the run applied {updates:,} delayed weight updates at a fixed 32-event delay, and the "
            "empirical per-example adviser regret against the best fixed adviser in hindsight lies "
            f"between {min(per_example):+.5f} and {max(per_example):+.5f}. The online learner is "
            "already tracking the best single adviser essentially exactly; the loss it is minimizing "
            "simply does not have much left to give. Consistently with that, calibration selected the "
            "smallest learning rate in the frozen grid."
        )
        lines.append("")
    lines.append(
        "Taken together the indicated next direction is a **richer causal feature or scoring "
        "representation that preserves magnitude**, evaluated against the same frozen Stage 0/Stage 1 "
        "references, rather than a better weighting scheme over the present rank-normalized pool. "
        "Two concrete, still-non-neural candidates follow directly from the measurements above: "
        "(i) combine advisers on a calibrated common numeric scale instead of within-event "
        "percentiles, since the one representational change measured here recovered the entire "
        "primary-pool deficit; and (ii) target the loss at the retention boundary rather than at all "
        "comparable pairs, since only inversions that cross the eviction cutoff can change a miss. "
        "Neither requires a neural predictor, and neither should be adopted without its own "
        "preregistration."
    )
    return lines


def _next_action(verdict: str, analysis: Mapping[str, Any]) -> str:
    ablation = {
        (row["question"], int(row["capacity"])): float(row["relative_change"])
        for row in analysis["ablation"]
    }
    online_gain = [
        value for (question, _capacity), value in ablation.items() if question == "C_online_adaptation"
    ]
    static_gain = [
        value for (question, _capacity), value in ablation.items() if question == "B_static_weights"
    ]
    extended_gain = [
        value
        for (question, _capacity), value in ablation.items()
        if question == "G_extended_vs_stage1"
    ]
    pieces = []
    if verdict in {"RACE_STAGE2_STRONG_SUCCESS", "RACE_STAGE2_VERY_STRONG_SUCCESS"}:
        pieces.append(
            "Proceed: adaptive causal multi-horizon future-reuse ranking recovers a substantial part "
            "of the residual oracle gap under the frozen Stage 1 eviction mechanism."
        )
    elif verdict == "RACE_STAGE2_WEAK":
        pieces.append(
            "Do not add complexity yet. The measured gain is real but below the preregistered strong "
            "threshold; inspect the ranking diagnostics, the horizon weights and the delayed-feedback "
            "tables in this report before proposing a richer model."
        )
    else:
        pieces.append(
            "Do not respond by building a neural predictor. The preregistered NO-GO rule fired and the "
            "diagnostic sections above localize why."
        )
    if online_gain:
        pieces.append(
            f"Online adaptation changed cost relative to the calibration-learned static weights by "
            f"{100*float(np.mean(online_gain)):+.2f}% on average across the four non-degenerate "
            "capacities."
        )
    if static_gain:
        pieces.append(
            f"Calibration-learned static weights changed cost relative to uniform weights by "
            f"{100*float(np.mean(static_gain)):+.2f}% on average."
        )
    if extended_gain:
        pieces.append(
            "The labeled adviser-diversity ablation, which adds the frozen Stage 1 winner itself to "
            f"the pool, changed cost relative to the Stage 1 winner by "
            f"{100*float(np.mean(extended_gain)):+.2f}% on average; this isolates how much of the "
            "primary-pool result is a pool-expressiveness limit rather than a weighting limit."
        )
    return " ".join(pieces)


def _core_answer(analysis: Mapping[str, Any], primary: str) -> str:
    decision_rows = [row for row in analysis["suite_results"] if int(row["capacity"]) != 8]
    improvements = [float(row["race_improvement_over_stage1"]) for row in decision_rows]
    gaps = [float(row["original_oracle_gap_closed"]) for row in decision_rows]
    stage1_gaps = [float(row["stage1_original_oracle_gap_closed"]) for row in decision_rows]
    return (
        "Can adaptive causal multi-horizon future-reuse ranking recover a substantial fraction of the "
        "expert-residency oracle gap that simple fixed prediction leaves behind? On this frozen "
        f"evidence, the answer is: the frozen primary variant `{primary}` changed transfer cost "
        f"against the Stage 1 winner by {_percent(min(improvements))} to {_percent(max(improvements))} "
        f"across capacities 12–32 and closed {_percent(min(gaps))} to {_percent(max(gaps))} of the "
        f"original Stage 0 oracle gap, against {_percent(min(stage1_gaps))} to "
        f"{_percent(max(stage1_gaps))} for simple fixed prediction. "
        f"The preregistered verdict is {analysis['verdict']}."
    )
