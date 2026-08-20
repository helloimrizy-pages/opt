#!/usr/bin/env python3
"""Section 23 figures.  Aggregates first, representative examples clearly labelled."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optstate.adam_state import INTERVENTIONS   # noqa: E402
from optstate.metrics import PRIMARY_WINDOW     # noqa: E402

REF = "CARRY_ALL"
COLORS = {
    "CARRY_ALL": "#222222",
    "RESET_M_KEEP_V_STEP": "#0072B2",
    "RESET_V_KEEP_M_STEP": "#D55E00",
    "RESET_MV_KEEP_STEP": "#009E73",
    "RESET_STEP_ONLY": "#CC79A7",
    "FRESH_ADAM": "#E69F00",
}
SHORT = {
    "CARRY_ALL": "CARRY_ALL",
    "RESET_M_KEEP_V_STEP": "RESET_M",
    "RESET_V_KEEP_M_STEP": "RESET_V",
    "RESET_MV_KEEP_STEP": "RESET_MV",
    "RESET_STEP_ONLY": "RESET_STEP",
    "FRESH_ADAM": "FRESH_ADAM",
}
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def savefig(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig1_curves(curves: pd.DataFrame, out: Path):
    sub = curves[(curves["mode"] == "primary") & (curves.condition == "shift")
                 & (curves.beta1 == 0.9) & (curves.lr == 1e-3)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
    for name in INTERVENTIONS:
        s = sub[sub.intervention == name].sort_values("batch_since_boundary")
        if s.empty:
            continue
        axes[0].plot(s.batch_since_boundary + 1, 100 * s.accuracy, label=SHORT[name],
                     color=COLORS[name], lw=1.6 if name == REF else 1.2,
                     ls="-" if name == REF else "--")
    carry = sub[sub.intervention == REF].sort_values("batch_since_boundary")
    for name in INTERVENTIONS:
        if name == REF:
            continue
        s = sub[sub.intervention == name].sort_values("batch_since_boundary")
        if s.empty:
            continue
        d = 100 * (s.accuracy.values - carry.accuracy.values)
        axes[1].plot(s.batch_since_boundary + 1, d, label=SHORT[name], color=COLORS[name], lw=1.2)
    # third panel: same differences with the RESET_V pathology removed, so the
    # non-pathological variants are readable on their own scale
    for name in INTERVENTIONS:
        if name in (REF, "RESET_V_KEEP_M_STEP"):
            continue
        s = sub[sub.intervention == name].sort_values("batch_since_boundary")
        if s.empty:
            continue
        d = 100 * (s.accuracy.values - carry.accuracy.values)
        axes[2].plot(s.batch_since_boundary + 1, d, label=SHORT[name],
                     color=COLORS[name], lw=1.2)
    axes[1].axhline(0, color="#222222", lw=1.2)
    axes[2].axhline(0, color="#222222", lw=1.2)
    for ax in axes:
        ax.axvspan(0.5, PRIMARY_WINDOW + 0.5, color="#cccccc", alpha=0.35, zorder=0)
        ax.set_xlabel("batches since boundary")
    axes[0].set_ylabel("online top-1 accuracy (%)")
    axes[1].set_ylabel("accuracy - CARRY_ALL (pp)")
    axes[2].set_ylabel("accuracy - CARRY_ALL (pp)")
    axes[0].set_title("Adaptation after an abrupt corruption shift")
    axes[1].set_title("Difference from CARRY_ALL")
    axes[2].set_title("Difference, excluding the RESET_V pathology")
    axes[0].legend(ncol=2, fontsize=7.5)
    fig.suptitle("Shaded band = preregistered early10 window", y=1.02, fontsize=8, color="#555")
    savefig(fig, out, "fig1_adaptation_curves")


def fig2_benefit_distribution(summ: pd.DataFrame, out: Path):
    sub = summ[(summ["mode"] == "primary") & (summ.condition == "shift")
               & (summ.beta1 == 0.9) & (summ.lr == 1e-3) & (summ.intervention != REF)]
    if sub.empty:
        return
    per = (sub.groupby(["intervention", "order_name", "boundary_id", "transition"])
              ["reset_benefit"].mean().reset_index())
    names = [n for n in INTERVENTIONS if n != REF]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    data = [per[per.intervention == n]["reset_benefit"].values for n in names]
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.85)
    for pc, n in zip(parts["bodies"], names):
        pc.set_facecolor(COLORS[n]); pc.set_alpha(0.28)
    for i, (n, d) in enumerate(zip(names, data), start=1):
        jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.22
        ax.scatter(i + jitter, d, s=9, color=COLORS[n], alpha=0.75, lw=0)
        ax.hlines(d.mean(), i - 0.3, i + 0.3, color="black", lw=2)
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([SHORT[n] for n in names], rotation=12)
    ax.set_ylabel(f"early{PRIMARY_WINDOW} reset benefit (pp)")
    ax.set_title(f"Per-transition benefit over CARRY_ALL  (n={per.transition.nunique()} "
                 f"transitions x seeds, all shown)")
    savefig(fig, out, "fig2_benefit_distribution")


def fig3_shift_vs_stationary(summ: pd.DataFrame, out: Path):
    ctrl = summ[(summ["mode"] == "control") & (summ.beta1 == 0.9) & (summ.lr == 1e-3)
                & (summ.intervention != REF)]
    if ctrl.empty:
        return
    names = [n for n in INTERVENTIONS if n != REF]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    width = 0.36
    for k, (cond, hatch) in enumerate((("shifted", ""), ("stationary", "//"))):
        means, los, his = [], [], []
        for n in names:
            v = (ctrl[(ctrl.condition == cond) & (ctrl.intervention == n)]
                 .groupby(["order_name", "boundary_id"])["reset_benefit"].mean().values)
            if len(v) == 0:
                means.append(np.nan); los.append(0); his.append(0); continue
            rng = np.random.default_rng(7)
            boot = v[rng.integers(0, len(v), size=(5000, len(v)))].mean(axis=1)
            means.append(v.mean())
            los.append(v.mean() - np.percentile(boot, 2.5))
            his.append(np.percentile(boot, 97.5) - v.mean())
        xs = np.arange(len(names)) + (k - 0.5) * width
        ax.bar(xs, means, width, yerr=[los, his], capsize=3, hatch=hatch,
               color=["#4C72B0" if cond == "shifted" else "#BBBBBB"] * len(names),
               edgecolor="black", lw=0.6,
               label="true shift (A->B)" if cond == "shifted" else "stationary (A->A)")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SHORT[n] for n in names], rotation=12)
    ax.set_ylabel(f"early{PRIMARY_WINDOW} reset benefit (pp)")
    ax.set_title("Matched pseudo-boundary control: identical checkpoint, only the "
                 "incoming distribution differs")
    ax.legend()
    savefig(fig, out, "fig3_shift_vs_stationary")


def fig4_cos_vs_benefit(summ: pd.DataFrame, diag: pd.DataFrame, out: Path):
    if diag.empty:
        return
    sub = summ[(summ["mode"] == "primary") & (summ.condition == "shift")
               & (summ.beta1 == 0.9) & (summ.lr == 1e-3)]
    if sub.empty:
        return
    dk = diag.set_index(["_file", "boundary_id", "condition"])
    names = ["RESET_M_KEEP_V_STEP", "RESET_MV_KEEP_STEP", "FRESH_ADAM"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2), sharey=True)
    from scipy import stats as sps
    for ax, n in zip(np.atleast_1d(axes), names):
        s = sub[sub.intervention == n]
        xs, ys = [], []
        for _, r in s.iterrows():
            key = (r["_file"], r["boundary_id"], r["condition"])
            if key in dk.index:
                d = dk.loc[key]
                if isinstance(d, pd.DataFrame):
                    d = d.iloc[0]
                xs.append(float(d["cos_m_g"])); ys.append(float(r["reset_benefit"]))
        if len(xs) < 4:
            continue
        ax.scatter(xs, ys, s=12, color=COLORS[n], alpha=0.7, lw=0)
        rho = sps.spearmanr(xs, ys)
        ax.axhline(0, color="#888", lw=0.8, ls=":")
        ax.axvline(0, color="#888", lw=0.8, ls=":")
        ax.set_title(f"{SHORT[n]}\nSpearman rho={rho.statistic:+.2f} (p={rho.pvalue:.3g})",
                     fontsize=8.5)
        ax.set_xlabel(r"boundary $\cos(m_{prev},\,g_{new})$")
    np.atleast_1d(axes)[0].set_ylabel(f"early{PRIMARY_WINDOW} reset benefit (pp)")
    savefig(fig, out, "fig4_cos_mg_vs_benefit")


def fig5_beta1(summ: pd.DataFrame, out: Path):
    sub = summ[(summ["mode"] == "primary") & (summ.condition == "shift") & (summ.lr == 1e-3)]
    if sub.beta1.nunique() < 2:
        return
    common = set.intersection(*[set(sub[sub.beta1 == v]["cluster"])
                                for v in sorted(sub.beta1.unique())])
    sub = sub[sub["cluster"].isin(common)]
    betas = sorted(sub.beta1.unique())
    names = [n for n in INTERVENTIONS if n != REF]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for n in names:
        means, los, his = [], [], []
        for v in betas:
            vals = (sub[(sub.beta1 == v) & (sub.intervention == n)]
                    .groupby(["order_name", "boundary_id"])["reset_benefit"].mean().values)
            rng = np.random.default_rng(11)
            boot = vals[rng.integers(0, len(vals), size=(5000, len(vals)))].mean(axis=1)
            means.append(vals.mean())
            los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
        axes[0].errorbar(betas, means,
                         yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                         marker="o", ms=4, lw=1.3, capsize=3, color=COLORS[n], label=SHORT[n])
    axes[0].axhline(0, color="#888", lw=0.9, ls=":")
    axes[0].set_xlabel(r"Adam $\beta_1$"); axes[0].set_ylabel(f"early{PRIMARY_WINDOW} benefit (pp)")
    axes[0].set_title(r"Reset benefit vs first-moment memory ($\beta_1$)")
    axes[0].legend(fontsize=7.5, ncol=2)

    carry = (sub[sub.intervention == REF].groupby("beta1")[f"acc_first{PRIMARY_WINDOW}"]
             .mean() * 100)
    axes[1].plot(carry.index, carry.values, marker="s", color="#222", lw=1.4)
    axes[1].set_xlabel(r"Adam $\beta_1$")
    axes[1].set_ylabel(f"CARRY_ALL early{PRIMARY_WINDOW} accuracy (%)")
    axes[1].set_title("Absolute CARRY_ALL level (context)")
    savefig(fig, out, "fig5_benefit_vs_beta1")


def fig6_state_behaviour(curves: pd.DataFrame, batches: pd.DataFrame, out: Path):
    sub = curves[(curves["mode"] == "primary") & (curves.condition == "shift")
                 & (curves.beta1 == 0.9) & (curves.lr == 1e-3)]
    if sub.empty or "m_norm" not in sub.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for n in INTERVENTIONS:
        s = sub[sub.intervention == n].sort_values("batch_since_boundary")
        s = s[s.batch_since_boundary < 10]
        if s.empty:
            continue
        axes[0].plot(s.batch_since_boundary + 1, s.m_norm, marker="o", ms=3,
                     color=COLORS[n], label=SHORT[n], lw=1.2)
        axes[1].plot(s.batch_since_boundary + 1, s.sqrt_v_mean, marker="o", ms=3,
                     color=COLORS[n], lw=1.2)
        axes[2].plot(s.batch_since_boundary + 1, s.cos_m_g, marker="o", ms=3,
                     color=COLORS[n], lw=1.2)
    axes[2].axhline(0, color="#888", lw=0.8, ls=":")
    axes[0].set_ylabel(r"$\|m\|$"); axes[1].set_ylabel(r"mean $\sqrt{v}$")
    axes[2].set_ylabel(r"$\cos(m_t,\,g_t)$")
    for ax in axes:
        ax.set_xlabel("batches since boundary")
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Optimizer-state behaviour after abrupt shifts "
                 "(mean over all transitions, seeds and orders)", fontsize=9)
    savefig(fig, out, "fig6_state_behaviour")


def fig7_baseline(baseline_json: Path, out: Path):
    if not baseline_json.exists():
        return
    data = json.loads(baseline_json.read_text())["results"]
    fig, axes = plt.subplots(1, len(data), figsize=(6.0 * len(data), 3.2), squeeze=False)
    for ax, (arch, entry) in zip(axes[0], data.items()):
        methods = ["source", "norm", "tent"]
        corr = list(entry["per_corruption_error_pct"]["source"].keys())
        xs = np.arange(len(corr))
        for i, m in enumerate(methods):
            ax.bar(xs + (i - 1) * 0.27,
                   [entry["per_corruption_error_pct"][m][c] for c in corr],
                   0.27, label=f"{m} (mean {entry['mean_error_pct'][m]:.1f}%)")
        ref = entry.get("official_reference_error_pct", {})
        for m, style in (("source", ":"), ("tent", "--")):
            if m in ref:
                ax.axhline(ref[m], ls=style, lw=1, color="#444")
        ax.set_xticks(xs); ax.set_xticklabels(corr, rotation=75, fontsize=6.5)
        ax.set_ylabel("error (%)")
        ax.set_title(f"{arch} ({entry.get('arch_description')})\n"
                     f"dashed/dotted = official reference means", fontsize=8.5)
        ax.legend(fontsize=7)
    savefig(fig, out, "fig0_baseline_reproduction")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/optimizer_state_stage1"))
    ap.add_argument("--out", default=str(ROOT / "figures/optimizer_state_stage1"))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)

    summ = pd.read_csv(res / "summary.csv")
    curves_path = res / "tables/adaptation_curves.csv"
    curves = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()
    diag_path = res / "tables/boundary_diagnostics.csv"
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()

    fig7_baseline(res / "raw/baseline/baseline_summary.json", out)
    if not curves.empty:
        fig1_curves(curves, out)
        fig6_state_behaviour(curves, pd.DataFrame(), out)
    fig2_benefit_distribution(summ, out)
    fig3_shift_vs_stationary(summ, out)
    fig4_cos_vs_benefit(summ, diag, out)
    fig5_beta1(summ, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
