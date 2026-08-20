#!/usr/bin/env python3
"""Section 14-19 analysis: effect sizes, cluster bootstrap CIs, mechanism tests.

Reads every ``*.jsonl`` produced by ``run_boundary_experiment.py`` and writes
machine-readable tables plus one aggregate JSON.  Nothing is filtered: every
tested transition, including negative results, is carried through.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optstate import stats as S                      # noqa: E402
from optstate.adam_state import INTERVENTIONS        # noqa: E402
from optstate.env import environment_record, write_json  # noqa: E402
from optstate.metrics import PRIMARY_WINDOW, WINDOWS  # noqa: E402

REF = "CARRY_ALL"
KEY_COLS = ["mode", "condition", "order_name", "seed", "beta1", "lr",
            "boundary_id", "transition", "domain_from", "domain_to"]


def load_rows(raw_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    buckets: Dict[str, list] = defaultdict(list)
    files = []
    for d in raw_dirs:
        files.extend(sorted(Path(d).glob("*.jsonl")))
    skipped = 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # tolerate a torn final line while a run is still writing
                    skipped += 1
                    continue
                obj["_file"] = path.name
                buckets[obj.get("type", "unknown")].append(obj)
    if skipped:
        print(f"warning: skipped {skipped} unparseable JSONL line(s)", file=sys.stderr)
    out = {}
    for k, v in buckets.items():
        df = pd.DataFrame(v)
        if "adapted_param_names" in df.columns:
            df = df.drop(columns=["adapted_param_names"])
        out[k] = df
    return out


def add_benefits(summ: pd.DataFrame) -> pd.DataFrame:
    """reset_benefit (percentage points) against CARRY_ALL at each boundary."""
    metric_cols = [f"acc_first{k}" for k in WINDOWS] + [
        "acc_full", "cumulative_errors_first10", "mean_entropy_loss_first10",
        "mean_pred_entropy_first10", "mean_grad_norm_first10", "recovery_batch",
        "plateau_last10", "tail_accuracy", "min_batch_accuracy", "collapsed"]
    metric_cols = [c for c in metric_cols if c in summ.columns]
    ref = summ[summ.intervention == REF].set_index(KEY_COLS)[metric_cols]
    ref = ref[~ref.index.duplicated()]
    joined = summ.set_index(KEY_COLS)
    for c in metric_cols:
        base = ref[c].reindex(joined.index)
        scale = 100.0 if c.startswith("acc_") or c in ("plateau_last10", "tail_accuracy",
                                                       "min_batch_accuracy") else 1.0
        joined[f"benefit_{c}"] = (joined[c] - base) * scale
    joined["reset_benefit"] = joined[f"benefit_acc_first{PRIMARY_WINDOW}"]
    return joined.reset_index()


def cluster_of(row) -> str:
    return f"{row['order_name']}|{row['boundary_id']}|{row['transition']}"


def aggregate(df: pd.DataFrame, metric: str = "reset_benefit") -> Dict[str, dict]:
    out = {}
    for name in INTERVENTIONS:
        if name == REF:
            continue
        sub = df[df.intervention == name]
        if sub.empty:
            continue
        by_cluster: Dict[str, list] = defaultdict(list)
        for _, r in sub.iterrows():
            by_cluster[cluster_of(r)].append(r[metric])
        res = S.cluster_bootstrap_mean(by_cluster)
        res.update(S.paired_wilcoxon(by_cluster))
        res["n_runs"] = int(len(sub))
        res["frac_runs_positive"] = float((sub[metric] > 0).mean())
        out[name] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+",
                    default=[str(ROOT / "results/optimizer_state_stage1/raw/boundary")])
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1"))
    args = ap.parse_args()

    out_dir = Path(args.out); (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    data = load_rows([Path(p) for p in args.raw])
    if "branch_summary" not in data:
        raise SystemExit("no branch_summary rows found; run the experiment first")

    summ = add_benefits(data["branch_summary"])
    summ["cluster"] = summ.apply(cluster_of, axis=1)
    summ.to_csv(out_dir / "summary.csv", index=False)

    diag = data.get("boundary_diagnostic", pd.DataFrame())
    if not diag.empty:
        diag.to_csv(out_dir / "tables" / "boundary_diagnostics.csv", index=False)

    batches = data.get("batch", pd.DataFrame())
    curves = pd.DataFrame()
    if not batches.empty:
        b = batches[batches.condition.isin(["shift", "shifted", "stationary"])]
        curves = (b.groupby(["mode", "condition", "beta1", "lr", "intervention",
                             "batch_since_boundary"])
                    .agg(accuracy=("accuracy", "mean"),
                         entropy_loss=("entropy_loss", "mean"),
                         mean_pred_entropy=("mean_pred_entropy", "mean"),
                         grad_norm=("grad_norm", "mean"),
                         cos_m_g=("cos_m_g", "mean") if "cos_m_g" in b.columns else ("accuracy", "size"),
                         m_norm=("m_norm", "mean") if "m_norm" in b.columns else ("accuracy", "size"),
                         sqrt_v_mean=("sqrt_v_mean", "mean") if "sqrt_v_mean" in b.columns else ("accuracy", "size"),
                         n=("accuracy", "size"))
                    .reset_index())
        curves.to_csv(out_dir / "tables" / "adaptation_curves.csv", index=False)

    base_mask = (summ["mode"] == "primary") & (summ.beta1 == 0.9) & (summ.lr == 1e-3)
    primary = summ[base_mask & (summ.condition == "shift")]

    report: Dict[str, object] = {
        "n_raw_files": int(summ["_file"].nunique()),
        "windows": list(WINDOWS),
        "primary_window": PRIMARY_WINDOW,
        "reference_intervention": REF,
    }

    if not primary.empty:
        carry = primary[primary.intervention == REF]
        report["primary"] = {
            "n_transition_runs": int(carry.shape[0]),
            "n_unique_transitions": int(carry["cluster"].nunique()),
            "orders": sorted(carry.order_name.unique().tolist()),
            "seeds": sorted(int(s) for s in carry.seed.unique().tolist()),
            "carry_early10_accuracy_mean": float(carry[f"acc_first{PRIMARY_WINDOW}"].mean()),
            "carry_early10_accuracy_sd": float(carry[f"acc_first{PRIMARY_WINDOW}"].std(ddof=1)),
            "carry_full_domain_accuracy_mean": float(carry["acc_full"].mean()),
            "effects_early10": aggregate(primary, "reset_benefit"),
        }
        for k in WINDOWS:
            if k == PRIMARY_WINDOW:
                continue
            report["primary"][f"effects_first{k}"] = aggregate(primary, f"benefit_acc_first{k}")
        report["primary"]["effects_full_domain"] = aggregate(primary, "benefit_acc_full")
        report["primary"]["effects_recovery_batch"] = aggregate(primary, "benefit_recovery_batch")

        per_tr = (primary.pivot_table(index=["order_name", "boundary_id", "transition"],
                                      columns="intervention", values="reset_benefit",
                                      aggfunc="mean"))
        per_tr.to_csv(out_dir / "tables" / "per_transition_early10_benefit.csv")

    # ---- stationary control (matched pseudo-boundary) ----
    ctrl = summ[(summ["mode"] == "control") & (summ.beta1 == 0.9) & (summ.lr == 1e-3)]
    if not ctrl.empty:
        cc: Dict[str, object] = {}
        for cond in ("shifted", "stationary"):
            sub = ctrl[ctrl.condition == cond]
            if sub.empty:
                continue
            cc[cond] = {
                "n_transition_runs": int(sub[sub.intervention == REF].shape[0]),
                "carry_early10_accuracy_mean": float(
                    sub[sub.intervention == REF][f"acc_first{PRIMARY_WINDOW}"].mean()),
                "effects_early10": aggregate(sub, "reset_benefit"),
            }
        contrasts = {}
        for name in INTERVENTIONS:
            if name == REF:
                continue
            a = ctrl[(ctrl.condition == "shifted") & (ctrl.intervention == name)]
            b = ctrl[(ctrl.condition == "stationary") & (ctrl.intervention == name)]
            if a.empty or b.empty:
                continue
            ga: Dict[str, list] = defaultdict(list)
            gb: Dict[str, list] = defaultdict(list)
            for _, r in a.iterrows():
                ga[cluster_of(r)].append(r["reset_benefit"])
            for _, r in b.iterrows():
                gb[cluster_of(r)].append(r["reset_benefit"])
            contrasts[name] = S.diff_of_means_ci(ga, gb)
        cc["shifted_minus_stationary"] = contrasts
        report["stationary_control"] = cc

    # ---- mechanism: boundary cos(m,g) vs reset benefit ----
    if not diag.empty and not primary.empty:
        dk = diag.set_index(["_file", "boundary_id", "condition"])
        assoc = {}
        for name in INTERVENTIONS:
            if name == REF:
                continue
            sub = primary[primary.intervention == name]
            xs, ys, cls = [], [], []
            for _, r in sub.iterrows():
                key = (r["_file"], r["boundary_id"], r["condition"])
                if key in dk.index:
                    d = dk.loc[key]
                    if isinstance(d, pd.DataFrame):
                        d = d.iloc[0]
                    xs.append(float(d["cos_m_g"]))
                    ys.append(float(r["reset_benefit"]))
                    cls.append(r["cluster"])
            if len(xs) >= 8:
                assoc[name] = S.spearman_with_ci(xs, ys, cls)
                assoc[name]["directional_prediction"] = (
                    "negative rho: lower cos(m,g) -> larger benefit from removing m")
        report["mechanism_cos_m_g_vs_benefit"] = assoc

        extra = {}
        for col in ("update_cos_neg_g[CARRY_ALL]", "update_cos_neg_g[FRESH_ADAM]",
                    "m_over_g_norm", "grad_norm", "m_norm", "sqrt_v_mean"):
            if col not in diag.columns:
                continue
            sub = primary[primary.intervention == "RESET_M_KEEP_V_STEP"]
            xs, ys, cls = [], [], []
            for _, r in sub.iterrows():
                key = (r["_file"], r["boundary_id"], r["condition"])
                if key in dk.index:
                    d = dk.loc[key]
                    if isinstance(d, pd.DataFrame):
                        d = d.iloc[0]
                    xs.append(float(d[col])); ys.append(float(r["reset_benefit"]))
                    cls.append(r["cluster"])
            if len(xs) >= 8:
                extra[col] = S.spearman_with_ci(xs, ys, cls)
        report["mechanism_other_predictors_reset_m"] = extra

        # Section 13: effective per-parameter update-scale summaries.  Every
        # intervention changes the bias-correction ratio as well as the memory
        # content, so the implied first-step size is reported explicitly.
        scale = {}
        for cond, sub_d in (("shift", diag[diag.condition.isin(["shift", "shifted"])]),
                            ("stationary", diag[diag.condition == "stationary"])):
            if sub_d.empty or "update_norm[CARRY_ALL]" not in sub_d.columns:
                continue
            ref_norm = sub_d["update_norm[CARRY_ALL]"].astype(float)
            entry = {}
            for name in INTERVENTIONS:
                col = f"update_norm[{name}]"
                if col not in sub_d.columns:
                    continue
                ratio = sub_d[col].astype(float) / ref_norm
                entry[name] = {
                    "mean_update_norm": float(sub_d[col].astype(float).mean()),
                    "median_ratio_vs_carry": float(np.nanmedian(ratio)),
                    "mean_update_cos_neg_g": float(
                        sub_d.get(f"update_cos_neg_g[{name}]",
                                  pd.Series(dtype=float)).astype(float).mean()),
                    "mean_update_cos_vs_carry": (
                        float(sub_d[f"update_cos_vs_carry[{name}]"].astype(float).mean())
                        if f"update_cos_vs_carry[{name}]" in sub_d.columns else None),
                }
            scale[cond] = entry
        report["boundary_effective_step_scale"] = scale

        shift_diag = diag[diag.condition.isin(["shift", "shifted"])]
        stat_diag = diag[diag.condition == "stationary"]
        report["boundary_diagnostic_summary"] = {
            "shift": {c: float(shift_diag[c].mean()) for c in
                      ("cos_m_g", "grad_norm", "m_norm", "sqrt_v_mean", "m_over_g_norm",
                       "step_prev", "boundary_pre_update_accuracy")
                      if c in shift_diag.columns},
            "stationary": {c: float(stat_diag[c].mean()) for c in
                           ("cos_m_g", "grad_norm", "m_norm", "sqrt_v_mean", "m_over_g_norm",
                            "step_prev", "boundary_pre_update_accuracy")
                           if c in stat_diag.columns} if not stat_diag.empty else None,
        }

    # ---- beta1 sweep ----
    b1 = summ[(summ["mode"] == "primary") & (summ.condition == "shift") & (summ.lr == 1e-3)]
    if b1.beta1.nunique() > 1:
        common = sorted(set.intersection(*[
            set(b1[b1.beta1 == v]["cluster"]) for v in sorted(b1.beta1.unique())]))
        sweep = {}
        for v in sorted(b1.beta1.unique()):
            sub = b1[(b1.beta1 == v) & (b1["cluster"].isin(common))]
            carry = sub[sub.intervention == REF]
            sweep[str(v)] = {
                "n_transition_runs": int(carry.shape[0]),
                "n_clusters": int(carry["cluster"].nunique()),
                "carry_early10_accuracy_mean": float(carry[f"acc_first{PRIMARY_WINDOW}"].mean()),
                "effects_early10": aggregate(sub, "reset_benefit"),
            }
        report["beta1_sweep"] = {"matched_clusters": len(common), "by_beta1": sweep}

    # ---- learning-rate robustness ----
    lrdf = summ[(summ["mode"] == "primary") & (summ.condition == "shift") & (summ.beta1 == 0.9)]
    if lrdf.lr.nunique() > 1:
        common = sorted(set.intersection(*[
            set(lrdf[lrdf.lr == v]["cluster"]) for v in sorted(lrdf.lr.unique())]))
        sweep = {}
        for v in sorted(lrdf.lr.unique()):
            sub = lrdf[(lrdf.lr == v) & (lrdf["cluster"].isin(common))]
            carry = sub[sub.intervention == REF]
            sweep[str(v)] = {
                "n_transition_runs": int(carry.shape[0]),
                "carry_early10_accuracy_mean": float(carry[f"acc_first{PRIMARY_WINDOW}"].mean()),
                "carry_collapsed_fraction": float(carry["collapsed"].mean()),
                "effects_early10": aggregate(sub, "reset_benefit"),
            }
        report["lr_sweep"] = {"matched_clusters": len(common), "by_lr": sweep}

    # ---- gradual control ----
    grad = summ[(summ["mode"] == "gradual")]
    if not grad.empty:
        carry = grad[grad.intervention == REF]
        report["gradual_control"] = {
            "n_transition_runs": int(carry.shape[0]),
            "carry_early10_accuracy_mean": float(carry[f"acc_first{PRIMARY_WINDOW}"].mean()),
            "effects_early10": aggregate(grad, "reset_benefit"),
        }

    # ---- failure cases, per-order breakdown, crossover analysis ----
    if not primary.empty:
        fails = {}
        per_order = {}
        for name in INTERVENTIONS:
            if name == REF:
                continue
            sub = primary[primary.intervention == name]
            if sub.empty:
                continue
            per_cluster = (sub.groupby(["order_name", "boundary_id", "transition"])
                              ["reset_benefit"].mean().reset_index())
            neg = per_cluster[per_cluster.reset_benefit <= 0].sort_values("reset_benefit")
            fails[name] = {
                "n_transitions": int(len(per_cluster)),
                "n_non_positive": int(len(neg)),
                "worst": [
                    {"order": r.order_name, "boundary_id": int(r.boundary_id),
                     "transition": r.transition, "benefit_pp": float(r.reset_benefit)}
                    for r in neg.head(12).itertuples()],
            }
            per_order[name] = {
                o: float(g.reset_benefit.mean())
                for o, g in per_cluster.groupby("order_name")}
        report["failure_cases"] = fails
        report["per_order_mean_benefit_pp"] = per_order

        # Continual Tent degrades along the chain, so boundary position matters.
        by_pos = {}
        for name in INTERVENTIONS:
            sub = primary[primary.intervention == name]
            if sub.empty:
                continue
            col = "reset_benefit" if name != REF else f"acc_first{PRIMARY_WINDOW}"
            by_pos[name] = {int(b): float(g[col].mean())
                            for b, g in sub.groupby("boundary_id")}
        report["benefit_by_boundary_index"] = by_pos

        if not diag.empty:
            dk2 = diag.set_index(["_file", "boundary_id", "condition"])
            cross = {}
            for name in ("RESET_M_KEEP_V_STEP", "FRESH_ADAM"):
                sub = primary[primary.intervention == name]
                pos, negg = defaultdict(list), defaultdict(list)
                for _, r in sub.iterrows():
                    key = (r["_file"], r["boundary_id"], r["condition"])
                    if key not in dk2.index:
                        continue
                    d = dk2.loc[key]
                    if isinstance(d, pd.DataFrame):
                        d = d.iloc[0]
                    (pos if r["reset_benefit"] > 0 else negg)[r["cluster"]].append(
                        float(d["cos_m_g"]))
                if pos and negg:
                    cross[name] = {
                        "n_transitions_reset_helps": len(pos),
                        "n_transitions_carry_helps_or_ties": len(negg),
                        "cos_m_g_when_reset_helps_minus_when_it_does_not":
                            S.diff_of_means_ci(pos, negg),
                    }
            report["crossover_analysis"] = cross

    # ---- SECONDARY whole-sequence strategy comparison ----
    seq = summ[summ["mode"] == "sequence"]
    if not seq.empty and not batches.empty:
        sb = batches[batches["mode"] == "sequence"]
        tot = (sb.groupby(["intervention", "seed"])
                 .apply(lambda g: g.n_correct.sum() / g.n.sum(), include_groups=False)
                 .rename("stream_accuracy").reset_index())
        tot.to_csv(out_dir / "tables" / "whole_sequence_strategies.csv", index=False)
        base_acc = tot[tot.intervention == REF].set_index("seed")["stream_accuracy"]
        rows = {}
        for name, g in tot.groupby("intervention"):
            g = g.set_index("seed")
            rows[name] = {
                "mean_stream_accuracy": float(g["stream_accuracy"].mean()),
                "mean_stream_error_pct": float(100 * (1 - g["stream_accuracy"].mean())),
                "delta_vs_carry_pp": float(
                    100 * (g["stream_accuracy"] - base_acc.reindex(g.index)).mean()),
                "per_seed": {int(k): float(v) for k, v in g["stream_accuracy"].items()},
            }
        report["whole_sequence_strategies"] = {
            "note": "SECONDARY, descriptive. A fixed always-on policy applied at every "
                    "boundary of one 15-domain chain. Not a method and not tuned.",
            "by_policy": rows,
        }

    report["environment"] = environment_record([0, 1, 2])
    write_json(out_dir / "aggregate.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("primary", "stationary_control", "beta1_sweep")},
                     indent=2, default=str)[:6000])
    print(f"\nwrote {out_dir/'summary.csv'} ({len(summ)} rows)")
    print(f"wrote {out_dir/'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
