from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import numpy as np


METRIC_LABELS = {
    "routing_frequency": "Routing frequency",
    "gate_mass": "Gate mass",
    "functional_contribution": "Functional contribution",
    "gradient_attribution": "Gradient × gate attribution",
}

DOMAIN_COLORS = {
    "general": "#4C78A8",
    "math": "#F58518",
    "coding": "#54A24B",
    "reasoning": "#B279A2",
}


def create_all_figures(results: dict[str, Any], output_dir: Path) -> list[Path]:
    matplotlib_cache = output_dir / ".matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for figures; install requirements.txt"
        ) from exc
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted(
        {
            row["metric"]
            for row in results["cross_domain_correlations"]
        },
        key=lambda item: list(METRIC_LABELS).index(item),
    )
    paths: list[Path] = []
    for metric in metrics:
        paths.extend(_spearman_heatmap(results, figure_dir, metric, plt))
        paths.extend(_correlation_by_layer(results, figure_dir, metric, plt))
    paths.extend(_expert_rank_change(results, figure_dir, plt))
    paths.extend(_routing_functional_scatter(results, figure_dir, plt))
    paths.extend(_top25_by_layer(results, figure_dir, plt))
    return paths


def _spearman_heatmap(
    results: dict[str, Any], output_dir: Path, metric: str, plt: Any
) -> list[Path]:
    domains = results["experiment"]["domains"]
    matrix = np.eye(len(domains), dtype=float)
    rows = [
        row
        for row in results["cross_domain_correlations"]
        if row["metric"] == metric and row["layer"] == "average"
    ]
    for row in rows:
        first = domains.index(row["domain_a"])
        second = domains.index(row["domain_b"])
        value = _float_or_nan(row["spearman"])
        matrix[first, second] = matrix[second, first] = value
    fig, axis = plt.subplots(figsize=(5.1, 4.3))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(domains)), [item.title() for item in domains], rotation=30)
    axis.set_yticks(range(len(domains)), [item.title() for item in domains])
    axis.set_title(
        f"Cross-domain Spearman: {METRIC_LABELS.get(metric, metric)}\n"
        "Mean across MoE layers"
    )
    for row in range(len(domains)):
        for column in range(len(domains)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "NA" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and abs(value) > 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    return _save_both(fig, output_dir / f"spearman_heatmap_{metric}", plt)


def _correlation_by_layer(
    results: dict[str, Any], output_dir: Path, metric: str, plt: Any
) -> list[Path]:
    domains = results["experiment"]["domains"]
    preferred = [
        ("general", "math"),
        ("general", "coding"),
        ("general", "reasoning"),
    ]
    pairs = [pair for pair in preferred if pair[0] in domains and pair[1] in domains]
    if not pairs:
        pairs = list(itertools.combinations(domains, 2))[:3]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for index, (domain_a, domain_b) in enumerate(pairs):
        rows = [
            row
            for row in results["cross_domain_correlations"]
            if row["metric"] == metric
            and row["domain_a"] == domain_a
            and row["domain_b"] == domain_b
            and row["layer"] != "average"
        ]
        rows.sort(key=lambda row: int(row["layer_ordinal"]))
        axis.plot(
            [row["layer"] for row in rows],
            [_float_or_nan(row["spearman"]) for row in rows],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=f"{domain_a.title()} vs {domain_b.title()}",
            color=list(DOMAIN_COLORS.values())[index],
        )
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, label="0.5 heuristic")
    axis.axhline(0.8, color="#AAAAAA", linestyle=":", linewidth=0.9, label="0.8 heuristic")
    axis.set_ylim(-1.02, 1.02)
    axis.set_xlabel("MoE layer")
    axis.set_ylabel("Spearman correlation")
    axis.set_title(f"Cross-domain stability by layer: {METRIC_LABELS.get(metric, metric)}")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2)
    return _save_both(fig, output_dir / f"correlation_by_layer_{metric}", plt)


def _expert_rank_change(
    results: dict[str, Any], output_dir: Path, plt: Any
) -> list[Path]:
    domains = results["experiment"]["domains"]
    strongest = _strongest_functional_layers(results)[:3]
    if not strongest:
        return []
    specialized = results["domain_specialized_experts"]
    figure, axes = plt.subplots(
        1,
        len(strongest),
        figsize=(5.0 * len(strongest), 4.7),
        squeeze=False,
        sharey=True,
    )
    for panel, layer in enumerate(strongest):
        axis = axes[0, panel]
        candidates = [row for row in specialized if row["layer"] == layer]
        candidates.sort(
            key=lambda row: (row["rank_range"], row["absolute_normalized_range"]),
            reverse=True,
        )
        for index, row in enumerate(candidates[:6]):
            y = [_float_or_nan(row[f"{domain}_rank"]) for domain in domains]
            axis.plot(
                range(len(domains)),
                y,
                marker="o",
                linewidth=1.4,
                color=plt.cm.tab10(index),
                label=f"E{row['expert_id']}",
            )
        axis.set_xticks(range(len(domains)), [item.title() for item in domains], rotation=25)
        axis.invert_yaxis()
        axis.set_title(f"MoE layer {layer}")
        axis.grid(alpha=0.2)
        axis.legend(title="Expert", ncol=2)
        if panel == 0:
            axis.set_ylabel("Functional-contribution rank (1 = highest)")
    figure.suptitle("Rank changes of the most domain-specialized experts", y=1.02)
    return _save_both(figure, output_dir / "expert_rank_change", plt)


def _routing_functional_scatter(
    results: dict[str, Any], output_dir: Path, plt: Any
) -> list[Path]:
    domains = results["experiment"]["domains"]
    strongest = _strongest_functional_layers(results)
    if not strongest:
        return []
    representative = [strongest[0]]
    all_layers = sorted(
        {
            row["layer"]
            for row in results["expert_importance"]
            if isinstance(row["layer"], int)
        }
    )
    median_layer = all_layers[len(all_layers) // 2]
    if median_layer not in representative:
        representative.append(median_layer)
    figure, axes = plt.subplots(
        len(representative),
        len(domains),
        figsize=(3.5 * len(domains), 3.7 * len(representative)),
        squeeze=False,
        constrained_layout=True,
    )
    correlation_lookup = {
        (row["domain"], row["layer"]): row["spearman"]
        for row in results["routing_vs_functional_correlation"]
        if row["layer"] != "average"
    }
    for row_index, layer in enumerate(representative):
        for column, domain in enumerate(domains):
            axis = axes[row_index, column]
            rows = [
                row
                for row in results["expert_importance"]
                if row["domain"] == domain and row["layer"] == layer
            ]
            axis.scatter(
                [row["routing_frequency"] for row in rows],
                [row["functional_contribution"] for row in rows],
                s=17,
                alpha=0.7,
                color=DOMAIN_COLORS.get(domain, "#4C78A8"),
                edgecolors="none",
            )
            correlation = _float_or_nan(correlation_lookup.get((domain, layer)))
            axis.set_title(
                f"{domain.title()}, layer {layer}\nSpearman = {correlation:.2f}",
                pad=9,
            )
            axis.grid(alpha=0.18)
            if row_index == len(representative) - 1:
                axis.set_xlabel("Routing frequency")
            if column == 0:
                axis.set_ylabel("Functional contribution")
    figure.suptitle("Routing utilization versus weighted expert-output magnitude")
    return _save_both(figure, output_dir / "routing_vs_functional", plt)


def _top25_by_layer(
    results: dict[str, Any], output_dir: Path, plt: Any
) -> list[Path]:
    rows = [
        row
        for row in results["topk_overlap"]
        if row["metric"] == "functional_contribution"
        and row["threshold"] == 0.25
        and row["layer"] != "average"
    ]
    if not rows:
        return []
    pairs = sorted({(row["domain_a"], row["domain_b"]) for row in rows})
    figure, axis = plt.subplots(figsize=(7.7, 4.5))
    for index, pair in enumerate(pairs):
        selected = [
            row
            for row in rows
            if (row["domain_a"], row["domain_b"]) == pair
        ]
        selected.sort(key=lambda row: int(row["layer_ordinal"]))
        axis.plot(
            [row["layer"] for row in selected],
            [row["jaccard_similarity"] for row in selected],
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=f"{pair[0].title()} vs {pair[1].title()}",
            color=plt.cm.tab10(index),
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("MoE layer")
    axis.set_ylabel("Top-25% Jaccard similarity")
    axis.set_title("Functional-contribution top-expert overlap by layer")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2)
    return _save_both(figure, output_dir / "top25_overlap_by_layer", plt)


def _strongest_functional_layers(results: dict[str, Any]) -> list[int]:
    rows = [
        row
        for row in results["cross_domain_correlations"]
        if row["metric"] == "functional_contribution" and row["layer"] != "average"
    ]
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["layer"]), []).append(_float_or_nan(row["spearman"]))
    scored = []
    for layer, values in grouped.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            scored.append((float(finite.mean()), layer))
    scored.sort()
    return [layer for _, layer in scored]


def _save_both(figure: Any, base_path: Path, plt: Any) -> list[Path]:
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
