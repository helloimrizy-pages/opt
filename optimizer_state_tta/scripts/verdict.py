#!/usr/bin/env python3
"""Apply the preregistered GO / NO-GO rules to ``aggregate.json``.

The rules were frozen in ``configs/optimizer_state_stage1/prereg_stage1.json``
before any experiment ran.  This script only evaluates them; it does not choose
thresholds.  The verdict namespace is ``OSM_STAGE1_*`` because ``RACE_STAGE1_*``
is already used by the unrelated ``stage3_residency/stage1_prediction`` study in
this repository; the equivalent RACE token is printed alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optstate.env import write_json     # noqa: E402

MOMENT_INTERVENTIONS = ["RESET_M_KEEP_V_STEP", "RESET_V_KEEP_M_STEP",
                        "RESET_MV_KEEP_STEP", "FRESH_ADAM"]
STRONG_PP, WEAK_PP = 1.0, 0.5
MAJORITY = 0.70


def fmt_ci(d):
    return f"{d['mean']:+.2f} pp [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", default=str(ROOT / "results/optimizer_state_stage1/aggregate.json"))
    ap.add_argument("--baseline", default=str(ROOT / "results/optimizer_state_stage1/raw/baseline/baseline_summary.json"))
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1/verdict.json"))
    args = ap.parse_args()

    agg = json.loads(Path(args.aggregate).read_text())
    base = json.loads(Path(args.baseline).read_text()) if Path(args.baseline).exists() else {}

    checks: dict = {}
    primary = agg.get("primary", {})
    eff = primary.get("effects_early10", {})

    baseline_valid = bool(base.get("baseline_valid_primary_arch", False))
    checks["baseline_valid"] = baseline_valid

    # ranked by absolute effect so a "carry helps" reverse result is not hidden
    ranked = sorted(((n, eff[n]) for n in MOMENT_INTERVENTIONS if n in eff),
                    key=lambda kv: -abs(kv[1]["mean"]))
    best_name, best = (ranked[0] if ranked else (None, None))
    best_pos = max(((n, eff[n]) for n in MOMENT_INTERVENTIONS if n in eff),
                   key=lambda kv: kv[1]["mean"], default=(None, None))

    checks["largest_absolute_moment_effect"] = (
        {"intervention": best_name, **best} if best else None)
    checks["largest_positive_moment_effect"] = (
        {"intervention": best_pos[0], **best_pos[1]} if best_pos[0] else None)

    if best_pos[0]:
        b = best_pos[1]
        checks["c1_effect_ge_1pp"] = bool(b["mean"] >= STRONG_PP)
        checks["c1b_effect_ge_0p5pp"] = bool(b["mean"] >= WEAK_PP)
        checks["c2_ci_excludes_zero"] = bool(b["ci_low"] > 0 or b["ci_high"] < 0)
        checks["c3_majority_positive"] = bool(b["frac_positive"] >= MAJORITY)
    else:
        checks.update({"c1_effect_ge_1pp": False, "c1b_effect_ge_0p5pp": False,
                       "c2_ci_excludes_zero": False, "c3_majority_positive": False})

    # C4: shift specificity
    ctrl = agg.get("stationary_control", {})
    contrasts = ctrl.get("shifted_minus_stationary", {})
    ref_name = best_pos[0] or "RESET_M_KEEP_V_STEP"
    c = contrasts.get(ref_name)
    if c:
        checks["c4_shift_specific"] = bool(
            (c["ci_low"] > 0) and (c["mean_a"] > c["mean_b"]))
        checks["c4_detail"] = c
    else:
        checks["c4_shift_specific"] = None

    # C5: step-counter confound
    step = eff.get("RESET_STEP_ONLY")
    if step and best_pos[0] and abs(best_pos[1]["mean"]) > 1e-9:
        ratio = step["mean"] / best_pos[1]["mean"]
        checks["c5_step_only_does_not_explain"] = bool(ratio < 0.5)
        checks["c5_detail"] = {"reset_step_only": step["mean"],
                               "best_moment": best_pos[1]["mean"], "ratio": ratio}
    else:
        checks["c5_step_only_does_not_explain"] = None

    # C6: mechanism
    mech = agg.get("mechanism_cos_m_g_vs_benefit", {}).get("RESET_M_KEEP_V_STEP")
    mech_ok = bool(mech and (mech["ci_high"] < 0 or mech["ci_low"] > 0))
    checks["c6a_cos_mg_association_significant"] = mech_ok
    checks["c6a_detail"] = mech

    b1s = agg.get("beta1_sweep", {}).get("by_beta1", {})
    b1_ok = None
    if b1s:
        def m(v, name="RESET_M_KEEP_V_STEP"):
            e = b1s.get(v, {}).get("effects_early10", {}).get(name)
            return e["mean"] if e else float("nan")
        vals = {v: m(v) for v in b1s}
        checks["c6b_detail"] = vals
        try:
            low = abs(vals.get("0.0", vals.get("0", float("nan"))))
            high = max(abs(vals.get("0.9", float("nan"))), abs(vals.get("0.99", float("nan"))))
            b1_ok = bool(high > 0 and low < 0.5 * high)
        except Exception:
            b1_ok = None
    checks["c6b_beta1_memory_dependence"] = b1_ok
    checks["c6_any_mechanistic_signature"] = bool(mech_ok or b1_ok)

    # sign stability across seeds is reported by the analysis; treat a
    # cluster-level positive fraction below 50% as unstable
    if best_pos[0]:
        checks["c7_sign_stable"] = bool(best_pos[1]["frac_positive"] >= 0.5)

    strong = all([
        baseline_valid,
        checks.get("c1_effect_ge_1pp"),
        checks.get("c2_ci_excludes_zero"),
        checks.get("c3_majority_positive"),
        checks.get("c4_shift_specific") is True,
        checks.get("c5_step_only_does_not_explain") is not False,
        checks.get("c6_any_mechanistic_signature"),
    ])
    weak = (not strong) and all([
        baseline_valid,
        checks.get("c1b_effect_ge_0p5pp"),
        checks.get("c2_ci_excludes_zero") or checks.get("c3_majority_positive"),
    ])
    verdict = ("OSM_STAGE1_STRONG_GO" if strong else
               "OSM_STAGE1_WEAK_GO" if weak else "OSM_STAGE1_NO_GO")
    if not baseline_valid:
        verdict = "OSM_STAGE1_NO_GO"
        checks["forced_no_go_reason"] = "BASELINE_INVALID"

    payload = {
        "verdict": verdict,
        "verdict_race_equivalent": verdict.replace("OSM_STAGE1", "RACE_STAGE1"),
        "criteria": checks,
        "thresholds": {"strong_pp": STRONG_PP, "weak_pp": WEAK_PP, "majority": MAJORITY},
        "primary_effects_early10": eff,
        "note": "Mechanical evaluation of the preregistered rules. The written "
                "report interprets these together with the reverse-result rule.",
    }
    write_json(Path(args.out), payload)

    print("\n" + "=" * 78)
    print("STAGE 1 PREREGISTERED CRITERIA")
    print("=" * 78)
    print(f"baseline valid                    : {baseline_valid}")
    if primary:
        print(f"transitions tested (runs/unique)  : "
              f"{primary.get('n_transition_runs')} / {primary.get('n_unique_transitions')}")
        print(f"seeds                             : {primary.get('seeds')}")
        print(f"CARRY_ALL early10 accuracy        : "
              f"{100*primary.get('carry_early10_accuracy_mean', float('nan')):.2f}%")
    for n, d in eff.items():
        print(f"  {n:<22s} {fmt_ci(d)}  positive in {100*d['frac_positive']:.0f}% "
              f"of {d['n_clusters']} transitions")
    if c:
        print(f"shift - stationary ({ref_name}) : {c['diff']:+.2f} pp "
              f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]")
    if mech:
        print(f"Spearman cos(m,g) vs RESET_M benefit: rho={mech['rho']:+.3f} "
              f"[{mech['ci_low']:+.3f}, {mech['ci_high']:+.3f}] (n={mech['n']})")
    if b1s:
        print("beta1 pattern (RESET_M mean pp)   : " +
              "  ".join(f"b1={k}:{v:+.2f}" for k, v in sorted(checks.get('c6b_detail', {}).items())))
    lrs = agg.get("lr_sweep", {}).get("by_lr", {})
    if lrs:
        print("lr robustness (RESET_M mean pp)   : " + "  ".join(
            f"lr={k}:{(v['effects_early10'].get('RESET_M_KEEP_V_STEP') or {}).get('mean', float('nan')):+.2f}"
            for k, v in sorted(lrs.items(), key=lambda kv: float(kv[0]))))
    grad = agg.get("gradual_control", {}).get("effects_early10", {})
    if grad:
        print("gradual severity control          : " + "  ".join(
            f"{k}:{v['mean']:+.2f}" for k, v in grad.items()))
    print("-" * 78)
    print(f"VERDICT: {verdict}   (equivalently {payload['verdict_race_equivalent']})")
    print("=" * 78)

    proj = ROOT
    print("\nlocations")
    print(f"  raw data : {proj/'results/optimizer_state_stage1/raw'}")
    print(f"  summary  : {proj/'results/optimizer_state_stage1/summary.csv'}")
    print(f"  aggregate: {proj/'results/optimizer_state_stage1/aggregate.json'}")
    print(f"  figures  : {proj/'figures/optimizer_state_stage1'}")
    print(f"  report   : {proj/'reports/optimizer_state_stage1.md'}")
    print(f"  verdict  : {Path(args.out)}")
    import subprocess
    try:
        st = subprocess.run(["git", "status", "--short"], cwd=str(proj.parent),
                            capture_output=True, text=True, check=True).stdout
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(proj.parent),
                              capture_output=True, text=True, check=True).stdout.strip()
        print(f"\ngit HEAD {head}; status:")
        print("\n".join("  " + l for l in st.strip().splitlines()[:40]) or "  (clean)")
    except Exception as exc:
        print(f"git status unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
