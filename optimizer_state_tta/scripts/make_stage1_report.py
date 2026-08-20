#!/usr/bin/env python3
"""Generate the data-driven tables for reports/optimizer_state_stage1.md.

Writes ``reports/_stage1_tables.md``.  The narrative report includes these
tables verbatim, so every number in the report comes from ``aggregate.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/optimizer_state_stage1"
OUT = ROOT / "reports/_stage1_tables.md"

SHORT = {
    "CARRY_ALL": "CARRY_ALL", "RESET_M_KEEP_V_STEP": "RESET_M_KEEP_V_STEP",
    "RESET_V_KEEP_M_STEP": "RESET_V_KEEP_M_STEP",
    "RESET_MV_KEEP_STEP": "RESET_MV_KEEP_STEP",
    "RESET_STEP_ONLY": "RESET_STEP_ONLY", "FRESH_ADAM": "FRESH_ADAM",
}
ORDER = list(SHORT)


def effect_table(effects: dict, title: str) -> str:
    lines = [f"**{title}**", "",
             "| intervention | mean Δ (pp) | 95% CI (cluster bootstrap) | median Δ | "
             "transitions positive | n transitions | Wilcoxon p |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for k in ORDER:
        if k not in effects:
            continue
        e = effects[k]
        p = e.get("p_value")
        ptxt = "—" if p is None or p != p else f"{p:.3g}"
        lines.append(
            f"| `{k}` | {e['mean']:+.3f} | [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] | "
            f"{e['median']:+.3f} | {100*e['frac_positive']:.0f}% | {e['n_clusters']} | {ptxt} |")
    return "\n".join(lines)


def main() -> int:
    agg = json.loads((RES / "aggregate.json").read_text())
    verdict = json.loads((RES / "verdict.json").read_text()) if (RES / "verdict.json").exists() else {}
    parts: list[str] = []

    p = agg.get("primary")
    if p:
        parts += [
            "### Primary experiment — abrupt corruption-type boundaries",
            "",
            f"- transition-runs: **{p['n_transition_runs']}** "
            f"({p['n_unique_transitions']} unique transitions × seeds {p['seeds']}, "
            f"orders {p['orders']})",
            f"- `CARRY_ALL` early10 accuracy: **{100*p['carry_early10_accuracy_mean']:.2f}%** "
            f"(sd across transition-runs {100*p['carry_early10_accuracy_sd']:.2f} pp)",
            f"- `CARRY_ALL` full-domain (50-batch) accuracy: "
            f"**{100*p['carry_full_domain_accuracy_mean']:.2f}%**",
            "",
            effect_table(p["effects_early10"], "early10_accuracy benefit over CARRY_ALL"),
            "",
        ]
        for k in (1, 5, 25, 50):
            key = f"effects_first{k}"
            if key in p:
                parts += [effect_table(p[key], f"first-{k}-batch window"), ""]
        if "effects_full_domain" in p:
            parts += [effect_table(p["effects_full_domain"], "full 50-batch domain"), ""]
        if "effects_recovery_batch" in p:
            parts += [effect_table(p["effects_recovery_batch"],
                                   "recovery batch (negative = recovers earlier)"), ""]

    c = agg.get("stationary_control")
    if c:
        parts += ["### Stationary pseudo-boundary control", ""]
        for cond, label in (("shifted", "true shift (A→B)"),
                            ("stationary", "stationary (A→A, held-out samples)")):
            if cond in c:
                parts += [
                    f"{label}: CARRY_ALL early10 = "
                    f"{100*c[cond]['carry_early10_accuracy_mean']:.2f}% over "
                    f"{c[cond]['n_transition_runs']} transition-runs", "",
                    effect_table(c[cond]["effects_early10"], f"benefit — {label}"), ""]
        sm = c.get("shifted_minus_stationary", {})
        if sm:
            parts += ["**Shift specificity: (true-shift benefit) − (stationary benefit)**", "",
                      "| intervention | difference (pp) | 95% CI | shift mean | stationary mean |",
                      "|---|---:|---|---:|---:|"]
            for k in ORDER:
                if k in sm:
                    d = sm[k]
                    parts.append(f"| `{k}` | {d['diff']:+.3f} | "
                                 f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | "
                                 f"{d['mean_a']:+.3f} | {d['mean_b']:+.3f} |")
            parts.append("")

    m = agg.get("mechanism_cos_m_g_vs_benefit")
    if m:
        parts += ["### Mechanism — boundary cos(m_prev, g_new) vs reset benefit", "",
                  "| intervention | Spearman ρ | 95% CI (cluster bootstrap) | p | n |",
                  "|---|---:|---|---:|---:|"]
        for k in ORDER:
            if k in m:
                e = m[k]
                parts.append(f"| `{k}` | {e['rho']:+.3f} | "
                             f"[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] | "
                             f"{e['p_value']:.3g} | {e['n']} |")
        parts.append("")

    o = agg.get("mechanism_other_predictors_reset_m")
    if o:
        parts += ["**Other boundary predictors of `RESET_M_KEEP_V_STEP` benefit**", "",
                  "| predictor | Spearman ρ | 95% CI | p |", "|---|---:|---|---:|"]
        for k, e in o.items():
            parts.append(f"| `{k}` | {e['rho']:+.3f} | "
                         f"[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] | {e['p_value']:.3g} |")
        parts.append("")

    bd = agg.get("boundary_diagnostic_summary")
    if bd:
        parts += ["**Boundary state summary (means)**", "",
                  "| quantity | at shift boundaries | at stationary pseudo-boundaries |",
                  "|---|---:|---:|"]
        keys = list(bd.get("shift", {}))
        st = bd.get("stationary") or {}
        for k in keys:
            parts.append(f"| `{k}` | {bd['shift'][k]:.4g} | "
                         f"{st.get(k, float('nan')):.4g} |" if k in st else
                         f"| `{k}` | {bd['shift'][k]:.4g} | — |")
        parts.append("")

    es = agg.get("boundary_effective_step_scale", {}).get("shift")
    if es:
        parts += ["**Effective first-step scale at shift boundaries** "
                  "(implied Adam update, computed without mutating state)", "",
                  "| intervention | mean ‖Δθ‖ | median ‖Δθ‖ / ‖Δθ_CARRY‖ | cos(Δθ, −g) | cos(Δθ, Δθ_CARRY) |",
                  "|---|---:|---:|---:|---:|"]
        for k in ORDER:
            if k in es:
                e = es[k]
                cv = e.get("mean_update_cos_vs_carry")
                parts.append(f"| `{k}` | {e['mean_update_norm']:.4g} | "
                             f"{e['median_ratio_vs_carry']:.3f} | "
                             f"{e['mean_update_cos_neg_g']:+.3f} | "
                             + (f"{cv:+.3f} |" if cv is not None else "1.000 |"))
        parts.append("")

    bp = agg.get("benefit_by_boundary_index")
    if bp:
        idxs = sorted({int(i) for v in bp.values() for i in v})
        parts += ["### Benefit by boundary position along the chain", "",
                  "(continual Tent degrades as the chain proceeds, so position matters)", "",
                  "| intervention | " + " | ".join(str(i) for i in idxs) + " |",
                  "|---|" + "---:|" * len(idxs)]
        for k in ORDER:
            if k in bp:
                unit = "%" if k == REF else ""
                vals = []
                for i in idxs:
                    v = bp[k].get(str(i), bp[k].get(i))
                    if v is None:
                        vals.append("—")
                    elif k == REF:
                        vals.append(f"{100*v:.1f}")
                    else:
                        vals.append(f"{v:+.2f}")
                label = f"`{k}` (early10 %)" if k == REF else f"`{k}` (Δ pp)"
                parts.append(f"| {label} | " + " | ".join(vals) + " |")
        parts.append("")

    ws = agg.get("whole_sequence_strategies")
    if ws:
        parts += ["### SECONDARY — whole-sequence fixed-policy comparison", "",
                  "*Descriptive only. A single fixed policy applied at every boundary of "
                  "one 15-domain chain; weights diverge, so this is not a matched causal "
                  "comparison and nothing was tuned on it.*", "",
                  "| policy | mean stream error (%) | Δ accuracy vs CARRY_ALL (pp) |",
                  "|---|---:|---:|"]
        for k in ORDER:
            if k in ws["by_policy"]:
                d = ws["by_policy"][k]
                parts.append(f"| `{k}` | {d['mean_stream_error_pct']:.2f} | "
                             f"{d['delta_vs_carry_pp']:+.2f} |")
        parts.append("")

    b1 = agg.get("beta1_sweep")
    if b1:
        parts += [f"### β₁ sweep ({b1['matched_clusters']} matched transitions per arm)", "",
                  "| β₁ | CARRY_ALL early10 | " +
                  " | ".join(f"Δ `{k}`" for k in ORDER if k != "CARRY_ALL") + " |",
                  "|---|---:|" + "---:|" * (len(ORDER) - 1)]
        for v, d in sorted(b1["by_beta1"].items(), key=lambda kv: float(kv[0])):
            row = [f"**{v}**", f"{100*d['carry_early10_accuracy_mean']:.2f}%"]
            for k in ORDER:
                if k == "CARRY_ALL":
                    continue
                e = d["effects_early10"].get(k)
                row.append(f"{e['mean']:+.2f}" if e else "—")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

    lr = agg.get("lr_sweep")
    if lr:
        parts += [f"### Learning-rate robustness ({lr['matched_clusters']} matched transitions per arm)",
                  "",
                  "| lr | CARRY_ALL early10 | collapsed frac | " +
                  " | ".join(f"Δ `{k}`" for k in ORDER if k != "CARRY_ALL") + " |",
                  "|---|---:|---:|" + "---:|" * (len(ORDER) - 1)]
        for v, d in sorted(lr["by_lr"].items(), key=lambda kv: float(kv[0])):
            row = [f"**{v}**", f"{100*d['carry_early10_accuracy_mean']:.2f}%",
                   f"{d['carry_collapsed_fraction']:.2f}"]
            for k in ORDER:
                if k == "CARRY_ALL":
                    continue
                e = d["effects_early10"].get(k)
                row.append(f"{e['mean']:+.2f}" if e else "—")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

    g = agg.get("gradual_control")
    if g:
        parts += ["### Gradual-shift control (severity 1→2→3→4→5, same corruption)", "",
                  f"CARRY_ALL early10 = {100*g['carry_early10_accuracy_mean']:.2f}% over "
                  f"{g['n_transition_runs']} transition-runs", "",
                  effect_table(g["effects_early10"], "benefit at gradual severity boundaries"), ""]

    po = agg.get("per_order_mean_benefit_pp")
    if po:
        orders = sorted({o for v in po.values() for o in v})
        parts += ["### Per-corruption-order breakdown (mean early10 benefit, pp)", "",
                  "| intervention | " + " | ".join(orders) + " |",
                  "|---|" + "---:|" * len(orders)]
        for k in ORDER:
            if k in po:
                parts.append(f"| `{k}` | " +
                             " | ".join(f"{po[k].get(o, float('nan')):+.2f}" for o in orders) + " |")
        parts.append("")

    fc = agg.get("failure_cases")
    if fc:
        parts += ["### Failure cases", "",
                  "| intervention | transitions with benefit ≤ 0 | worst transitions (pp) |",
                  "|---|---:|---|"]
        for k in ORDER:
            if k in fc:
                f = fc[k]
                worst = "; ".join(f"{w['transition']} ({w['benefit_pp']:+.2f})"
                                  for w in f["worst"][:4]) or "—"
                parts.append(f"| `{k}` | {f['n_non_positive']}/{f['n_transitions']} | {worst} |")
        parts.append("")

    cx = agg.get("crossover_analysis")
    if cx:
        parts += ["### Crossover analysis", "",
                  "| intervention | reset helps | carry helps/ties | Δ mean cos(m,g) | 95% CI |",
                  "|---|---:|---:|---:|---|"]
        for k, d in cx.items():
            c2 = d["cos_m_g_when_reset_helps_minus_when_it_does_not"]
            parts.append(f"| `{k}` | {d['n_transitions_reset_helps']} | "
                         f"{d['n_transitions_carry_helps_or_ties']} | {c2['diff']:+.4f} | "
                         f"[{c2['ci_low']:+.4f}, {c2['ci_high']:+.4f}] |")
        parts.append("")

    if verdict:
        parts += ["### Preregistered criteria evaluation", "", "| criterion | result |", "|---|---|"]
        for k, v in verdict["criteria"].items():
            if k.endswith("_detail") or k.startswith("largest"):
                continue
            parts.append(f"| `{k}` | {v} |")
        parts += ["", f"Mechanical verdict: **{verdict['verdict']}**", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
