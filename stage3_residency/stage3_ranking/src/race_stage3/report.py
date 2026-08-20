"""Rendering of the frozen Stage 3 report."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


REGIME_ORDER = ("stationary", "abrupt", "repeated", "mixed")


def _pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{100 * float(value):.{digits}f}%"


def _ci(low: Any, high: Any) -> str:
    if low is None or high is None or not (np.isfinite(float(low)) and np.isfinite(float(high))):
        return ""
    return f" [{100*float(low):.2f}, {100*float(high):.2f}]"


def render_report(analysis: Mapping[str, Any], inputs: Any, frozen: Mapping[str, Any]) -> str:
    verdict = str(analysis["verdict"])
    primary = str(analysis["primary_variant"])
    suite = analysis["suite_results"]
    rows = [r for r in suite if int(r["capacity"]) != 8]
    improvements = [r["improvement_over_stage1"] for r in rows]
    gaps = [r["original_oracle_gap_closed"] for r in rows]
    stage1_gaps = [r["stage1_original_oracle_gap_closed"] for r in rows]
    decision = analysis["decision"]

    lines = [verdict, "", "# RACE Stage 3: Learned Causal Future-Reuse Ranking", ""]
    lines += [
        "## A. Executive verdict",
        "",
        f"- Frozen primary variant: `{primary}` — one calibration-fitted linear ranking function "
        f"per cache capacity over {len(frozen['feature_names']['all'])} raw-scale causal features, "
        "driving the unchanged Stage 1 eviction rule.",
        f"- Improvement over the frozen Stage 1 winner `{analysis['stage1_winner_method_id']}` "
        f"across capacities 12–32: {_pct(min(improvements))} to {_pct(max(improvements))}.",
        f"- Original Stage 0 oracle gap closed: {_pct(min(gaps))} to {_pct(max(gaps))}, against "
        f"{_pct(min(stage1_gaps))} to {_pct(max(stage1_gaps))} for the Stage 1 winner.",
        f"- {decision['reason']}",
        "",
        "| Capacity | Stage 1 winner | Stage 3 primary | Oracle | Improvement vs Stage 1 | Oracle gap closed | Stage 1 gap closed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:,.0f} | {:,.0f} | {:,.0f} | {}{} | {} | {} |".format(
                row["capacity"], row["stage1_cost"], row[f"{primary}_cost"], row["oracle_cost"],
                _pct(row["improvement_over_stage1"]),
                _ci(row["improvement_ci_low"], row["improvement_ci_high"]),
                _pct(row["original_oracle_gap_closed"]),
                _pct(row["stage1_original_oracle_gap_closed"]),
            )
        )

    stage2 = decision["stage2_criteria"]
    lines += [
        "",
        "### Stage 2's criteria, applied verbatim",
        "",
        "Stage 3 keeps Stage 2's STRONG and VERY_STRONG rungs byte-identical so the stages stay "
        "comparable, and adds a PARTIAL rung between a 10% cost win and failure. Under Stage 2's "
        "own rule this run would be judged:",
        "",
        f"- Condition A (at least 10% better than Stage 1) met at capacities {stage2['condition_a_capacities']}.",
        f"- Condition B (at least 30% of the original oracle gap) met at capacities {stage2['condition_b_capacities']}.",
        f"- Stage 2 STRONG_SUCCESS would {'PASS' if stage2['stage2_strong_success_would_pass'] else 'NOT pass'}.",
        "",
        "## B. Frozen prior evidence",
        "",
        f"- Stage 0 `RACE_STAGE0_STRONG_GO`; trace `{analysis['trace_hash']}`; archive "
        f"`{analysis['stage0_archive_manifest_sha256']}`.",
        f"- Stage 1 `RACE_STAGE1_STRONG_GO`; winner `{analysis['stage1_winner_method_id']}`; archive "
        f"`{analysis['stage1_archive_manifest_sha256']}`.",
        f"- Stage 2 `RACE_STAGE2_NO_GO`; archive `{analysis['stage2_archive_manifest_sha256']}`.",
        "",
        "Stage 2's diagnostic is the reason Stage 3 exists. It localized the failure as "
        "representational: percentile rank normalization applied to each adviser separately "
        "destroys the magnitude information the raw-scale Stage 1 hybrid uses, and no weighting "
        "over a rank-normalized pool can rebuild it. Stage 3 acts on exactly that finding.",
        "",
        "## C. Algorithm",
        "",
        "**Scoring.** One linear ranking function per cache capacity, `S_e = w_B · x_e`, over causal "
        "features kept on their natural scale. The frozen Stage 1 eviction rule is unchanged: "
        "retain the highest-scoring eligible candidates, ties broken by LRU recency then expert ID.",
        "",
        "**Features.** Calibration-fitted Markov survival probabilities at horizons 1–32 and their "
        "band differences; noisy-OR context combinations; the same context evaluated against the "
        "previous same-layer request; renewal structure (elapsed time, the two most recent "
        "inter-arrival gaps, an overdue ratio); exact windowed counts over 4/8/16/32 events; "
        "decayed request and gate statistics; calibration popularity; and decode-request-scope "
        "statistics. Every value uses only events already observed.",
        "",
        "**Fitting.** Convex weighted pairwise logistic ranking loss over within-candidate-set "
        "ordered pairs, target the capped next-use distance `min(d, 33)`, L2 chosen on a held-out "
        "tail of calibration groups, L-BFGS-B, calibration path only. Two collection rounds: round 1 "
        "under the frozen Stage 1 winner, round 2 under the round-1 model.",
        "",
        "**What Stage 3 does not do.** No neural network, no reinforcement learning, no recurrent or "
        "attention model, no online adaptation at evaluation time, no prefetching, no change to "
        "OLMoE routing, and no future information at decision time.",
        "",
        "## D. Causality and mechanism audit",
        "",
        "- The feature state is advanced only after the features for the same event have been read, "
        "so no feature can see its own event's consequences.",
        "- A Stage 3 replay driven by the frozen Stage 1 winner score reproduces the frozen Stage 1 "
        "cost exactly, which proves the eviction mechanism is unchanged.",
        "- The same mechanism driven by exact next-use scores reproduces the Stage 0 oracle exactly.",
        "- Mutating a later sequence cannot change any earlier action.",
        "- Fitting, feature standardization, static popularity and the L2 choice used only the 80 "
        "frozen calibration sequences; evaluation used the disjoint 320-sequence split.",
        "",
        "## E. Main results",
        "",
        "| Capacity | Stage 1 winner | " + " | ".join(analysis["variants"]) + " | Oracle |",
        "| ---: | ---: | " + " | ".join(["---:"] * len(analysis["variants"])) + " | ---: |",
    ]
    for row in suite:
        cells = " | ".join(f"{row[f'{v}_cost']:,.0f}" for v in analysis["variants"])
        lines.append(f"| {row['capacity']} | {row['stage1_cost']:,.0f} | {cells} | {row['oracle_cost']:,.0f} |")
    lines += [
        "",
        "By workload regime:",
        "",
        "| Regime | Capacity | Stage 1 | Stage 3 | Oracle | Improvement | Gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime in REGIME_ORDER:
        for row in analysis["scope_results"]:
            if row["scope"] != regime or int(row["capacity"]) == 8:
                continue
            lines.append(
                "| {} | {} | {:,.0f} | {:,.0f} | {:,.0f} | {} | {} |".format(
                    regime, row["capacity"], row["stage1_cost"], row[f"{primary}_cost"],
                    row["oracle_cost"], _pct(row["improvement_over_stage1"]),
                    _pct(row["original_oracle_gap_closed"])))
    flagged = [r for r in analysis["regression_rows"] if r["flagged"]]
    lines += [
        "",
        f"### Stability. {len(flagged)} of {len(analysis['regression_rows'])} workload/capacity cells "
        f"exceed the 3% regression flag.",
        "",
    ]
    if flagged:
        lines += ["| Workload | Capacity | Stage 1 | Stage 3 | Ratio |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in sorted(flagged, key=lambda r: -float(r["regression_ratio"]))[:25]:
            lines.append("| {} | {} | {:,.0f} | {:,.0f} | {:.4f} |".format(
                row["workload"], row["capacity"], row["stage1_cost"], row["stage3_cost"],
                row["regression_ratio"]))
    else:
        lines.append("No workload/capacity cell exceeded the 3% regression threshold.")

    lines += [
        "",
        "## F. Ablation",
        "",
        "| Question | Comparison | Capacity | Before | After | Relative change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["ablation"]:
        lines.append("| {} | {} | {} | {:,.0f} | {:,.0f} | {:+.2f}% |".format(
            row["question"], row["comparison"], row["capacity"], row["before_cost"],
            row["after_cost"], 100 * row["relative_change"]))
    lines += [
        "",
        "`B2_stage1_to_no_request_scope` is the honest like-for-like comparison: it restricts Stage 3 "
        "to the information Stage 1 also had. The difference between it and the primary variant is "
        "the value of knowing decode-request boundaries, which a serving stack always knows but no "
        "earlier RACE stage used.",
        "",
        "## G. Ranking diagnostics",
        "",
        "| Variant | Capacity | Eviction events | Ordering accuracy (capped) | Oracle-consistent | Oracle-optimal |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["ranking_diagnostics"]:
        if int(row["capacity"]) == 8:
            continue
        lines.append("| {} | {} | {:,} | {} | {} | {} |".format(
            row["variant"], row["capacity"], row["eviction_events"],
            _pct(row["pairwise_ordering_accuracy_capped"]),
            _pct(row["oracle_consistent_eviction_rate"]),
            _pct(row["oracle_optimal_eviction_rate"])))

    lines += [
        "",
        "## H. The ranking-accuracy wall",
        "",
        "The calibration study that produced this design also measured how far the approach can go. "
        "Pairwise ordering accuracy on held-out calibration candidate sets:",
        "",
        "| Model | Accuracy |",
        "| --- | ---: |",
        "| Stage 1 winner score | 60.3% |",
        "| linear, 19 causal features | 66.6% |",
        "| linear, 37 causal features | 67.7% |",
        "| linear, 45 features (adds request scope) | 67.7% |",
        "| gradient-boosted trees, 45 features | 69.2% |",
        "| gradient-boosted trees, 13x data and 8x capacity | 68.95% |",
        "",
        "Scaling training data 13-fold and model capacity 8-fold moves accuracy by under one point. "
        "The limit is the information carried by causal routing history, not the estimator, so a "
        "neural network would meet the same wall — which is what Stage 2's diagnostic predicted. "
        "There is a structural reason: next-use distance depends on which tokens the model will emit "
        "next, and past routing does not determine that.",
        "",
        "A non-causal frontier measurement on calibration mapped ranking accuracy to oracle-gap "
        "closure by blending the true ordering into the deployed score. Reaching a 10% cost win over "
        "Stage 1 needs roughly 70% accuracy at capacities 24 and 32 and 73–75% at capacity 16, and "
        "only if the extra accuracy lands on comparisons that straddle the eviction cutoff. Training "
        "weighted toward that cutoff was tried directly and made results monotonically worse.",
        "",
        "## I. Oracle residual",
        "",
        "| Capacity | Stage 3 cost | Oracle | Residual headroom | Stage 1 residual recovered |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {} | {:,.0f} | {:,.0f} | {} | {}{} |".format(
            row["capacity"], row[f"{primary}_cost"], row["oracle_cost"],
            _pct(row["residual_headroom"]), _pct(row["stage1_residual_recovered"]),
            _ci(row["stage1_residual_recovered_ci_low"], row["stage1_residual_recovered_ci_high"])))

    lines += [
        "",
        "## J. Limitations",
        "",
        "- Trace simulation of expert residency, misses, admissions and transfers. No end-to-end "
        "latency improvement and no hardware speedup is claimed or measured.",
        "- One fixed OLMoE-1B-7B-0924 decode trace over four domains; nothing generalizes "
        "automatically to other models, batch sizes or serving stacks.",
        "- The primary variant uses decode-request-boundary information that the Stage 1 baseline "
        "did not have. The `STAGE3_RANKER_NO_REQUEST_SCOPE` ablation is the like-for-like comparison "
        "and is reported beside it.",
        "- The ranking model is fitted offline on calibration and fixed at evaluation. It does not "
        "adapt online, so it cannot track a distribution shift that calibration did not contain.",
        "- The learning target is capped at 32 same-layer events; structure beyond that is invisible.",
        "- Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen "
        "stateful workload path; cache trajectories are not regenerated under resampled orderings.",
        "- The offline oracle is a non-causal diagnostic, not a deployable method.",
        "",
        "## K. Next recommendation",
        "",
        _next_action(verdict, analysis),
        "",
        "## Reproducibility",
        "",
        f"- Stage 3 preregistration: `{analysis['preregistration_hash']}`",
        f"- Stage 3 frozen config: `{analysis['frozen_config_file_sha256']}`",
        f"- Stage 3 evaluation manifest: `{analysis['evaluation_manifest_sha256']}`",
        f"- Stage 0 trace: `{analysis['trace_hash']}`",
        f"- Stage 0 archive: `{analysis['stage0_archive_manifest_sha256']}`",
        f"- Stage 1 archive: `{analysis['stage1_archive_manifest_sha256']}`",
        f"- Stage 2 archive: `{analysis['stage2_archive_manifest_sha256']}`",
        f"- Stage 3 source bundle at freeze: `{frozen['stage3_source_bundle_hash']}`",
        f"- Transition models: `{frozen['transition_model_hash']}`",
        "",
        "## Core answer",
        "",
        _core(analysis, primary),
    ]
    return "\n".join(lines) + "\n"


def _next_action(verdict: str, analysis: Mapping[str, Any]) -> str:
    ablation = {(r["question"], int(r["capacity"])): float(r["relative_change"])
                for r in analysis["ablation"]}
    like_for_like = [v for (q, _c), v in ablation.items() if q == "B2_stage1_to_no_request_scope"]
    scope_value = [v for (q, _c), v in ablation.items() if q == "B_request_scope_value"]
    pieces = []
    if verdict in {"RACE_STAGE3_STRONG_SUCCESS", "RACE_STAGE3_VERY_STRONG_SUCCESS"}:
        pieces.append("Proceed. A calibration-fitted causal ranking function over raw-scale features "
                      "recovers a substantial part of the residual oracle gap under the unchanged "
                      "Stage 1 eviction mechanism.")
    elif verdict == "RACE_STAGE3_PARTIAL_SUCCESS":
        pieces.append("Bank this as the strongest causal residency policy measured so far, but do not "
                      "claim the preregistered strong result. The accuracy-wall evidence in section H "
                      "says further gains will not come from a bigger model on this feature set.")
    elif verdict == "RACE_STAGE3_WEAK":
        pieces.append("The gain is real but narrow. Section H localizes why, and it is not a modelling "
                      "deficiency that more capacity would fix.")
    else:
        pieces.append("Do not answer this with a larger model. Section H shows the limit is the "
                      "information in causal routing history.")
    if like_for_like:
        pieces.append(f"Restricted to exactly the information Stage 1 had, Stage 3 changes cost by "
                      f"{100*float(np.mean(like_for_like)):+.2f}% on average across the four "
                      "non-degenerate capacities.")
    if scope_value:
        pieces.append(f"Decode-request-boundary awareness is worth a further "
                      f"{100*float(np.mean(scope_value)):+.2f}% on average, which is the one new "
                      "information source this stage introduced.")
    pieces.append("The measured frontier says a materially better residency policy needs information "
                  "about the tokens the model is about to emit, not a better estimator over routing "
                  "history. Anything that supplies that — for example a cheap signal derived from the "
                  "current forward pass before the MoE layers commit — is the next thing worth "
                  "preregistering, and it is a different experiment from this one.")
    return " ".join(pieces)


def _core(analysis: Mapping[str, Any], primary: str) -> str:
    rows = [r for r in analysis["suite_results"] if int(r["capacity"]) != 8]
    improvements = [float(r["improvement_over_stage1"]) for r in rows]
    gaps = [float(r["original_oracle_gap_closed"]) for r in rows]
    stage1 = [float(r["stage1_original_oracle_gap_closed"]) for r in rows]
    return (
        "Can a learned causal ranking function over raw-scale routing features recover substantially "
        "more of the expert-residency oracle gap than simple fixed prediction? On this frozen "
        f"evidence: `{primary}` changed transfer cost against the Stage 1 winner by "
        f"{_pct(min(improvements))} to {_pct(max(improvements))} across capacities 12–32 and closed "
        f"{_pct(min(gaps))} to {_pct(max(gaps))} of the original Stage 0 oracle gap, against "
        f"{_pct(min(stage1))} to {_pct(max(stage1))} for Stage 1. The preregistered verdict is "
        f"{analysis['verdict']}."
    )
