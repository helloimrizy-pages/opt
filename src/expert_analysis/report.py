from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_summary(results: dict[str, Any], output_path: Path) -> str:
    domains = results["experiment"]["domains"]
    correlations = results["cross_domain_correlations"]
    topk = results["topk_overlap"]
    routing_functional = results["routing_vs_functional_correlation"]
    specialized = results["domain_specialized_experts"]
    split_half = results.get("same_domain_split_half", [])
    controlled = results.get("controlled_corpus") or results["experiment"].get(
        "controlled_input"
    )
    masking_rows = results.get("expert_masking_loss", [])
    masking_contrasts = results.get("expert_masking_domain_contrasts", [])
    tested_positions = int(results["architecture"].get("num_moe_layers", 0)) * int(
        results["architecture"].get("num_experts", 0)
    )
    functional_aggregate = [
        row
        for row in correlations
        if row["metric"] == "functional_contribution" and row["layer"] == "average"
    ]
    functional_layers = [
        row
        for row in correlations
        if row["metric"] == "functional_contribution" and row["layer"] != "average"
    ]
    top25_functional = [
        row
        for row in topk
        if row["metric"] == "functional_contribution"
        and row["threshold"] == 0.25
        and row["layer"] == "average"
    ]
    minimum_examples = min(
        item["examples"] for item in results["token_counts"].values()
    )
    assessment, decision, assessment_reason = _assessment(
        functional_layers,
        top25_functional,
        minimum_examples=minimum_examples,
    )
    metric_stability = {
        metric: _finite_mean(
            row["spearman"]
            for row in correlations
            if row["metric"] == metric and row["layer"] == "average"
        )
        for metric in ("routing_frequency", "gate_mass", "functional_contribution")
    }
    metric_top25 = {
        metric: _finite_mean(
            row["jaccard_similarity"]
            for row in topk
            if row["metric"] == metric
            and row["threshold"] == 0.25
            and row["layer"] == "average"
        )
        for metric in ("routing_frequency", "gate_mass", "functional_contribution")
    }

    lines = [
        "# OLMoE Expert Importance Across Domains",
        "",
        "## Experimental setup",
        "",
        f"- Checkpoint: {results['model']['checkpoint']}",
        "- Resolved model revision: "
        f"{results['model'].get('resolved_revision') or 'not available'}",
        f"- Device: {results['experiment']['device_description']} "
        f"({results['experiment']['device']}, {results['experiment']['dtype']})",
        f"- Seed: {results['experiment']['seed']}",
        f"- Maximum sequence length: {results['experiment']['max_sequence_length']} tokens",
        f"- Bootstrap replicates: {results['experiment']['bootstrap_replicates']}",
        f"- Reference answers included: {results['experiment']['include_reference_answers']}",
    ]
    if controlled:
        lines.extend(
            [
                f"- Prompt style: {controlled.get('prompt_style', 'neutral fixed-token control')}",
                f"- Shared neutral prefix: `{_inline_code(controlled['neutral_prefix'])}`",
                "- Measured source positions per example: "
                f"{controlled['measured_tokens_per_example']}",
                "- Look-ahead next-token labels per example: "
                f"{controlled.get('lookahead_tokens_per_example', 1)}",
                "- Exact measured token budget per domain: "
                f"{controlled['measured_tokens_per_domain']}",
            ]
        )
    lines.extend(
        [
        "",
        "The model was evaluated without generation or weight updates. Routing utilization, "
        "selected gate mass, and the L2 magnitude of each weighted expert output were collected "
        "in the same backbone forward pass. Functional contribution is an activation-magnitude "
        "proxy, not a causal intervention.",
        "",
        "## Dataset and token summary",
        "",
        "| Domain | Dataset | Split | Examples | Tokens | Mean tokens/example | Substitution |",
        "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for domain in domains:
        dataset = results["datasets"][domain]
        counts = results["token_counts"][domain]
        lines.append(
            f"| {domain.title()} | {dataset['repository']} "
            f"({dataset.get('config') or 'default'}) | {dataset['split']} | "
            f"{counts['examples']} | {counts['tokens']} | "
            f"{counts['mean_tokens_per_example']:.1f} | "
            f"{'yes' if dataset.get('substituted') else 'no'} |"
        )
    capped_domains = [
        domain
        for domain in domains
        if results["token_counts"][domain]["mean_tokens_per_example"]
        >= 0.99 * results["experiment"]["max_sequence_length"]
    ]
    if capped_domains:
        lines.extend(
            [
                "",
                "**Length warning:** The mean token count reached the configured maximum "
                f"for {', '.join(domain.title() for domain in capped_domains)}. This indicates "
                "widespread truncation and should be revisited at the intended 512-token cap.",
            ]
        )

    lines.extend(
        [
            "",
            "## Main cross-domain correlations",
            "",
            "| Metric | Mean layer-wise Spearman across domain pairs | "
            "Mean top-25% Jaccard across domain pairs |",
            "|---|---:|---:|",
        ]
    )
    for metric, label in (
        ("routing_frequency", "Routing frequency"),
        ("gate_mass", "Gate mass"),
        ("functional_contribution", "Functional contribution"),
    ):
        lines.append(
            f"| {label} | {_fmt(metric_stability[metric])} | "
            f"{_fmt(metric_top25[metric])} |"
        )

    lowest_pairs = sorted(
        functional_aggregate, key=lambda row: _sort_float(row["spearman"])
    )
    lines.extend(
        [
            "",
            "## Lowest-correlation domain pairs",
            "",
            "| Pair | Functional Spearman | Kendall tau | Top-25% Jaccard |",
            "|---|---:|---:|---:|",
        ]
    )
    top25_by_pair = {
        (row["domain_a"], row["domain_b"]): row for row in top25_functional
    }
    for row in lowest_pairs:
        overlap = top25_by_pair[(row["domain_a"], row["domain_b"])]
        lines.append(
            f"| {row['domain_a'].title()} vs {row['domain_b'].title()} | "
            f"{_fmt(row['spearman'])} | {_fmt(row['kendall_tau'])} | "
            f"{_fmt(overlap['jaccard_similarity'])} |"
        )

    strongest_layers = _strongest_layers(functional_layers)
    lines.extend(
        [
            "",
            "## Layers with the strongest domain dependence",
            "",
            "| MoE layer | Mean functional Spearman across pairs | Lowest pair | "
            "Lowest Spearman |",
            "|---:|---:|---|---:|",
        ]
    )
    for row in strongest_layers[: min(8, len(strongest_layers))]:
        lines.append(
            f"| {row['layer']} | {_fmt(row['mean'])} | "
            f"{row['lowest_pair']} | {_fmt(row['lowest'])} |"
        )

    lines.extend(
        [
            "",
            "## Domain-specialized expert examples",
            "",
            "Specialization is based on normalized functional contribution. The reported ratio "
            "uses a 1e-12 numerical floor; the absolute range and rank change should be used "
            "to guard against ratios driven by tiny denominators.",
            "",
            "| Layer | Expert | Maximum domain | Minimum domain | Ratio | "
            "Normalized range | Rank range |",
            "|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        specialized,
        key=lambda item: (
            item["rank_range"],
            item["absolute_normalized_range"],
        ),
        reverse=True,
    )[:12]:
        lines.append(
            f"| {row['layer']} | {row['expert_id']} | {row['max_domain'].title()} | "
            f"{row['min_domain'].title()} | {_fmt_specialization(row)} | "
            f"{_fmt(row['absolute_normalized_range'], 4)} | "
            f"{_fmt(row['rank_range'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Routing frequency versus functional contribution",
            "",
            "| Domain | Mean within-domain Spearman across layers |",
            "|---|---:|",
        ]
    )
    for domain in domains:
        row = next(
            item
            for item in routing_functional
            if item["domain"] == domain and item["layer"] == "average"
        )
        lines.append(f"| {domain.title()} | {_fmt(row['spearman'])} |")
    stability_difference = (
        metric_stability["routing_frequency"]
        - metric_stability["functional_contribution"]
    )
    if np.isfinite(stability_difference):
        if stability_difference > 0.05:
            comparison = (
                "Routing-frequency rankings were more stable across domains than the functional "
                f"proxy by {stability_difference:.3f} mean Spearman."
            )
        elif stability_difference < -0.05:
            comparison = (
                "Functional-contribution rankings were more stable across domains than routing "
                f"frequency by {-stability_difference:.3f} mean Spearman."
            )
        else:
            comparison = (
                "Routing-frequency and functional-contribution rankings had similar average "
                f"cross-domain stability (difference {stability_difference:.3f})."
            )
        lines.extend(["", comparison])

    if split_half:
        lines.extend(
            [
                "",
                "## Same-domain split-half reliability",
                "",
                "Repeated disjoint half-splits estimate how reproducible each domain's ranking "
                "is at half the available sample size. The Spearman–Brown column projects the "
                "mean split-half value to the full sample size; it is a reliability diagnostic, "
                "not a cross-domain correction.",
                "",
                "| Domain | Metric | Mean split-half Spearman | 95% interval | "
                "Spearman–Brown |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for domain in domains:
            for metric, label in (
                ("routing_frequency", "Routing frequency"),
                ("gate_mass", "Gate mass"),
                ("functional_contribution", "Functional contribution"),
            ):
                row = next(
                    item
                    for item in split_half
                    if item["domain"] == domain
                    and item["metric"] == metric
                    and item["layer"] == "average"
                )
                lines.append(
                    f"| {domain.title()} | {label} | "
                    f"{_fmt(row['split_half_mean_spearman'])} | "
                    f"[{_fmt(row['split_half_ci_low'])}, "
                    f"{_fmt(row['split_half_ci_high'])}] | "
                    f"{_fmt(row['spearman_brown_corrected'])} |"
                )

    lines.extend(
        [
            "",
            "## Bootstrap uncertainty",
            "",
            "Examples were resampled independently within each domain from the stored "
            "per-example expert vectors; the model was not rerun. Aggregate intervals are for "
            "the mean layer-wise Spearman in each domain pair.",
            "",
            "| Domain pair | Observed | Bootstrap mean | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in lowest_pairs:
        lines.append(
            f"| {row['domain_a'].title()} vs {row['domain_b'].title()} | "
            f"{_fmt(row['spearman'])} | {_fmt(row['bootstrap_mean_spearman'])} | "
            f"[{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}] |"
        )

    if masking_rows:
        lines.extend(
            [
                "",
                "## Controlled expert-masking loss effects",
                "",
                "For each pre-registered layer/expert pair, its selected gate coefficient was "
                "set to zero at the measured source positions only. Tokens were not rerouted, "
                "and model weights were not changed. Positive delta NLL means masking made "
                "next-token prediction worse.",
                "",
                "| Layer/expert | Domain | Proxy rank | Routed-token fraction | "
                "Delta NLL (nats/token) | 95% paired-bootstrap CI |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in masking_rows:
            lines.append(
                f"| L{row['layer']}/E{row['expert_id']} | {row['domain'].title()} | "
                f"{_fmt(row['functional_rank'], 1)} | "
                f"{_fmt(row['fraction_tokens_routed'], 4)} | "
                f"{_fmt(row['delta_nll'], 6)} | "
                f"[{_fmt(row['delta_nll_ci_low'], 6)}, "
                f"{_fmt(row['delta_nll_ci_high'], 6)}] |"
            )

    if masking_contrasts:
        lines.extend(
            [
                "",
                "## Proxy-versus-causal domain contrasts",
                "",
                "For pre-registered targets, the high and low domains come from the prior "
                "prompt-only run rather than being selected from these masking outcomes. The "
                "contrast is high-domain minus low-domain mask delta NLL, with domains "
                "resampled independently. Controlled proxy extrema are shown as a replication "
                "check.",
                "",
                "| Layer/expert | Tested high vs low | Controlled proxy high vs low | "
                "Loss-delta contrast | 95% CI | Across-domain proxy/loss Spearman | "
                "Direction aligned |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for row in masking_contrasts:
            lines.append(
                f"| L{row['layer']}/E{row['expert_id']} | "
                f"{row['contrast_high_domain'].title()} vs "
                f"{row['contrast_low_domain'].title()} | "
                f"{row['proxy_high_domain'].title()} vs "
                f"{row['proxy_low_domain'].title()} | "
                f"{_fmt(row['high_minus_low_delta_nll'], 6)} | "
                f"[{_fmt(row['contrast_ci_low'], 6)}, "
                f"{_fmt(row['contrast_ci_high'], 6)}] | "
                f"{_fmt(row['proxy_loss_spearman'])} | "
                f"{'yes' if row['direction_aligned'] else 'no'} |"
            )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The functional metric remains a weighted-output norm. The masking experiment "
            "measures loss sensitivity only for the explicitly tested experts.",
            *(
                [
                    "- Sequence lengths, token budgets, answer inclusion, and the wrapper prefix "
                    "are controlled, but domain remains entangled with dataset choice and the "
                    "content's natural surface form."
                ]
                if controlled
                else [
                    "- Domain datasets differ in style, sequence length, and availability of "
                    "reference answers. Token normalization reduces but does not remove these "
                    "confounds."
                ]
            ),
            *(
                [
                    "- At least one domain reached the mean sequence-length cap in this run; "
                    "truncation can itself change the observed domain ranking."
                ]
                if capped_domains
                else []
            ),
            "- OLMoE's selected top-k weights are used exactly as implemented; when "
            "norm_topk_prob=false, selected weights do not sum to one.",
            "- Bootstrap intervals capture example-sampling uncertainty for these corpora, not "
            "model, checkpoint, prompt-format, or dataset-choice uncertainty.",
            "- This is one base checkpoint. Conclusions should be checked on another MoE model "
            "before guiding compression.",
            *(
                [
                    "- Selected-route zeroing does not reroute tokens to replacement experts and "
                    "is not identical to quantizing, deleting, or globally disabling an expert.",
                    "- Only a small pre-registered expert set was intervened on; it does not "
                    f"characterize causal sensitivity for all {tested_positions:,} "
                    "layer/expert positions.",
                    "- The three high-versus-low domain pairs were pre-registered from the prior "
                    "prompt-only run. Across-domain proxy/loss Spearman values use only four "
                    "domains and are descriptive; bootstrap intervals have no multiplicity "
                    "adjustment."
                ]
                if masking_rows
                else []
            ),
            "",
            "# Go / No-Go Assessment",
            "",
            f"**{decision}: {assessment} support.** {assessment_reason}",
            "",
            *(
                [
                    "The activation results are now accompanied by controlled selected-route "
                    "masking for the pre-registered experts. They still do not establish that a "
                    "specific mixed-precision allocation improves downstream quality."
                ]
                if masking_rows
                else [
                    "This result is evidence about distribution-conditioned routing utilization "
                    "and an activation-magnitude proxy. It is not yet evidence that a specific "
                    "mixed-precision allocation improves downstream quality. The next stage "
                    "should validate selected high-disagreement layers with expert ablation or "
                    "masking before implementing quantization."
                ]
            ),
        ]
    )
    if masking_contrasts:
        causal_decision, causal_reason = _causal_assessment(
            masking_contrasts, split_half
        )
        lines.extend(
            [
                "",
                "# Controlled Causal-Validation Assessment",
                "",
                f"**{causal_decision}.** {causal_reason}",
            ]
        )
    requested_examples = results["experiment"]["requested_examples_per_domain"]
    if requested_examples < 100:
        lines.extend(
            [
                "",
                f"> This was a reduced {requested_examples}-example/domain diagnostic, below "
                "the prescribed 100-example quick baseline. It validates the pipeline and "
                "provides a preliminary signal, not a go/no-go result.",
            ]
        )
    elif results["experiment"]["quick_mode"]:
        lines.extend(
            [
                "",
                "> This was a quick-mode run. Treat the assessment as preliminary until the "
                "500-example/domain run reproduces it.",
            ]
        )
    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def _causal_assessment(
    contrasts: list[dict[str, Any]], split_half: list[dict[str, Any]]
) -> tuple[str, str]:
    aligned = sum(bool(row["direction_aligned"]) for row in contrasts)
    robust = sum(bool(row["causal_specialization_supported"]) for row in contrasts)
    majority = len(contrasts) // 2 + 1
    reliability_values = []
    for row in split_half:
        if row["metric"] != "functional_contribution" or row["layer"] != "average":
            continue
        value = _sort_float(row.get("split_half_mean_spearman"))
        if np.isfinite(value):
            reliability_values.append(value)
    reliability = _finite_mean(reliability_values)
    reliability_text = (
        f"Mean functional split-half reliability was {_fmt(reliability)}. "
        if np.isfinite(reliability)
        else ""
    )
    if np.isfinite(reliability) and reliability < 0.5:
        return (
            "INCONCLUSIVE—IMPROVE RANKING RELIABILITY",
            reliability_text
            + "Within-domain rankings are too unstable to interpret the proxy/loss alignment "
            "confidently, even if individual masking effects are nonzero.",
        )
    if robust >= majority:
        return (
            "GO FOR A LIMITED, REVERSIBLE COMPRESSION PILOT",
            reliability_text
            + f"{robust}/{len(contrasts)} pre-registered experts showed both positive "
            "high-domain masking harm and a positive high-versus-low loss contrast, with both "
            "bootstrap intervals excluding zero. This supports testing a small reversible "
            "allocation pilot, not deployment-scale quantization.",
        )
    if aligned >= majority:
        return (
            "CONDITIONAL GO—EXPAND CAUSAL VALIDATION",
            reliability_text
            + f"{aligned}/{len(contrasts)} expert contrasts were directionally aligned with the "
            "functional proxy, but fewer than a majority had positive intervals excluding zero. "
            "Test more pre-registered experts and examples before bit allocation.",
        )
    return (
        "NO-GO FOR DOMAIN-AWARE BIT ALLOCATION AT THIS STAGE",
        reliability_text
        + f"Only {aligned}/{len(contrasts)} pre-registered expert contrasts aligned with the "
        "functional proxy. The activation ranking is not yet a reliable basis for domain-aware "
        "precision decisions.",
    )


def _inline_code(value: Any) -> str:
    return str(value).replace("`", "\\`").replace("\n", "\\n")


def _assessment(
    functional_layers: list[dict[str, Any]],
    top25_aggregate: list[dict[str, Any]],
    minimum_examples: int,
) -> tuple[str, str, str]:
    correlations = np.asarray(
        [row["spearman"] for row in functional_layers], dtype=float
    )
    correlations = correlations[np.isfinite(correlations)]
    jaccards = np.asarray(
        [row["jaccard_similarity"] for row in top25_aggregate], dtype=float
    )
    jaccards = jaccards[np.isfinite(jaccards)]
    if correlations.size == 0 or jaccards.size == 0:
        return (
            "inconclusive",
            "NO DECISION",
            "There were not enough finite correlations to apply the diagnostic heuristic.",
        )
    below_half = float(np.mean(correlations < 0.5))
    above_point_eight = float(np.mean(correlations > 0.8))
    mean_corr = float(correlations.mean())
    mean_jaccard = float(jaccards.mean())
    counts = (
        f"{int((correlations < 0.5).sum())}/{len(correlations)} layer-pair "
        f"functional correlations were below 0.5; mean functional Spearman was "
        f"{mean_corr:.3f}, and mean top-25% Jaccard was {mean_jaccard:.3f}."
    )
    if below_half >= 0.35 or (below_half >= 0.20 and mean_jaccard < 0.45):
        result = (
            "strong",
            "GO",
            counts
            + " The observed disagreement is broad enough to justify testing "
            "distributionally robust allocation.",
        )
    elif above_point_eight >= 0.80 and mean_jaccard >= 0.70:
        result = (
            "weak",
            "NO-GO",
            counts
            + " Expert ordering is mostly stable, so this checkpoint does not currently "
            "support domain-robust allocation as a primary research premise.",
        )
    else:
        result = (
            "moderate",
            "CONDITIONAL GO",
            counts
            + " Domain effects are present but not uniformly strong; validate the most "
            "disagreeing layers interventionally before building a compression method.",
        )
    if minimum_examples < 100:
        support, _, reason = result
        return (
            f"preliminary {support}",
            "NO DECISION",
            reason
            + f" However, the smallest domain has only {minimum_examples} examples, below "
            "the prescribed 100-example quick baseline; the heuristic decision is withheld.",
        )
    return result


def _strongest_layers(functional_layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[Any, list[dict[str, Any]]] = {}
    for row in functional_layers:
        by_layer.setdefault(row["layer"], []).append(row)
    output = []
    for layer, rows in by_layer.items():
        lowest = min(rows, key=lambda row: _sort_float(row["spearman"]))
        output.append(
            {
                "layer": layer,
                "mean": _finite_mean(row["spearman"] for row in rows),
                "lowest": lowest["spearman"],
                "lowest_pair": (
                    f"{lowest['domain_a'].title()} vs {lowest['domain_b'].title()}"
                ),
            }
        )
    output.sort(key=lambda row: _sort_float(row["mean"]))
    return output


def _finite_mean(values: Any) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _sort_float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return converted if np.isfinite(converted) else float("inf")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{converted:.{digits}f}" if np.isfinite(converted) else "NA"


def _fmt_specialization(row: dict[str, Any]) -> str:
    if row.get("minimum_was_zero"):
        return "∞ (zero minimum)"
    value = _sort_float(row.get("specialization_score"))
    if value > 1_000_000:
        return ">1e6"
    return _fmt(value, 2)
