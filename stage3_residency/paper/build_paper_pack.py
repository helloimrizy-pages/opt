"""Build the cross-stage artifacts a write-up needs but no single stage produces.

Everything here is derived from the sealed Stage 0/1/2/3 archives. It computes nothing
new about policy behaviour and changes no frozen artifact; it joins, compares and
accounts for what the four stages already measured.

Run from the repository root:
    stage3_residency/paper/build_paper_pack.sh
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residency_headroom.common import atomic_write_json, atomic_write_text, read_jsonl, write_csv

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "stage3_residency/paper"
CAPACITIES = (8, 12, 16, 24, 32)
DECISION = (12, 16, 24, 32)
REGIMES = ("stationary", "abrupt", "repeated", "mixed")

STAGE0 = ROOT / "stage3_residency"
STAGE1 = ROOT / "stage3_residency/stage1_prediction"
STAGE2 = ROOT / "stage3_residency/stage2_race"
STAGE3 = ROOT / "stage3_residency/stage3_ranking"


# --------------------------------------------------------------------------- load

def stage0_reference() -> tuple[dict, dict, dict]:
    """Strongest-simple and oracle costs per (workload, capacity), plus regimes."""
    simple: dict = {}
    oracle: dict = {}
    regime: dict = {}
    simple_names = {"lru", "lfu", "lfu_decay", "static_hotset"}
    best: dict = {}
    with (STAGE0 / "results/full/evaluation/results.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cost_model") != "unit_miss" or float(row.get("lambda", -1)) != 0.0:
                continue
            key = (row["workload"], int(row["cache_capacity"]))
            regime[row["workload"]] = row["regime"]
            policy = row["policy"]
            if policy == "oracle":
                oracle[key] = int(row["misses"])
            elif policy in simple_names:
                if policy == "lfu_decay" and not row.get("selected_decay_alpha"):
                    continue
                rank = (int(row["misses"]), policy)
                if key not in best or rank < best[key]:
                    best[key] = rank
                    simple[key] = int(row["misses"])
    return simple, oracle, regime


def stage_costs(path: Path, key_field: str, wanted: str) -> dict:
    out: dict = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row[key_field]) != wanted:
                continue
            out[(row["workload"], int(row["capacity"]))] = int(row["misses"])
    return out


def per_sequence(path: Path, key_field: str, wanted: str) -> dict:
    """(workload, capacity) -> list of per-sequence miss counts in workload order."""
    grouped: dict = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get(key_field)) != wanted:
                continue
            grouped[(row["workload"], int(row["capacity"]))].append(
                (int(row["sequence_position"]), int(row["source_sequence_id"]),
                 str(row["domain"]), int(row["misses"]))
            )
    return {k: sorted(v) for k, v in grouped.items()}


def stage1_per_sequence(wanted: str) -> dict:
    """Stage 1 rows, rebuilding the gitignored aggregate from committed checkpoints."""
    aggregate = STAGE1 / "results/full/per_sequence_results.jsonl"
    if aggregate.exists():
        return per_sequence(aggregate, "method_id", wanted)
    frozen = json.loads((STAGE0 / "results/full/frozen/frozen_evaluation_config.json").read_text())
    grouped: dict = defaultdict(list)
    for name in [w["name"] for w in frozen["workloads"]]:
        path = STAGE1 / f"results/full/checkpoints/{name}/per_sequence_results.jsonl"
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("method_id")) != wanted:
                    continue
                grouped[(row["workload"], int(row["capacity"]))].append(
                    (int(row["sequence_position"]), int(row["source_sequence_id"]),
                     str(row["domain"]), int(row["misses"]))
                )
    return {k: sorted(v) for k, v in grouped.items()}


# ---------------------------------------------------------------- paired bootstrap

def paired_bootstrap(left: dict, right: dict, workloads, capacity: int,
                     replicates: int = 1000, seed: int = 20260823) -> dict:
    """Conditional paired bootstrap of (left - right) / left, clustered by sequence."""
    parts, domains = [], {}
    for name in workloads:
        a, b = left[(name, capacity)], right[(name, capacity)]
        if [(p, s, d) for p, s, d, _ in a] != [(p, s, d) for p, s, d, _ in b]:
            raise SystemExit(f"per-sequence rows misaligned for {name} at capacity {capacity}")
        for _p, sequence, domain, _m in a:
            domains.setdefault(sequence, domain)
        parts.append(([s for _p, s, _d, _m in a],
                      np.asarray([m for *_x, m in a], dtype=np.float64),
                      np.asarray([m for *_x, m in b], dtype=np.float64)))
    by_domain = defaultdict(list)
    for sequence, domain in domains.items():
        by_domain[domain].append(sequence)
    digest = hashlib.sha256(f"paper-pack\0{seed}\0{capacity}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    multiplicity: dict = {}
    index = np.arange(replicates, dtype=np.int64)[:, None]
    for domain in sorted(by_domain):
        ids = np.asarray(sorted(by_domain[domain]), dtype=np.int64)
        counts = np.zeros((replicates, len(ids)), dtype=np.int32)
        np.add.at(counts, (index, rng.integers(0, len(ids), size=counts.shape)), 1)
        for position, sequence in enumerate(ids):
            multiplicity[int(sequence)] = counts[:, position]
    left_boot = np.zeros(replicates)
    right_boot = np.zeros(replicates)
    paired = []
    for sequences, a, b in parts:
        weights = np.stack([multiplicity[s] for s in sequences], axis=1)
        left_boot += (weights * a[None, :]).sum(axis=1)
        right_boot += (weights * b[None, :]).sum(axis=1)
        paired.extend(a - b)
    improvement = np.divide(left_boot - right_boot, left_boot,
                            out=np.full(replicates, np.nan), where=left_boot != 0)
    values = np.asarray(paired, dtype=np.float64)
    finite = improvement[np.isfinite(improvement)]
    total_left = sum(float(a.sum()) for _s, a, _b in parts)
    total_right = sum(float(b.sum()) for _s, _a, b in parts)
    return {
        "capacity": capacity,
        "left_cost": total_left,
        "right_cost": total_right,
        "improvement": (total_left - total_right) / total_left,
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "sequences_improved": int((values > 0).sum()),
        "sequences_worsened": int((values < 0).sum()),
        "paired_units": int(values.size),
        "bootstrap_replicates": replicates,
    }


# ------------------------------------------------------------------ policy overhead

FEATURE_OPS = {
    "markov gather and row mean, 6 horizons x top-8 x 64 experts": 6 * 8 * 64 + 6 * 64,
    "survival band differences": 4 * 64,
    "noisy-OR at two horizons": 2 * (8 * 64 + 64),
    "previous-request context at two horizons": 2 * (8 * 64 + 64),
    "renewal features (elapsed, gaps, overdue; logs counted at 10 ops)": 7 * 64 + 3 * 64 * 10,
    "windowed count reads": 4 * 64,
    "decayed request and gate statistics": 6 * 64,
    "static popularity": 64,
    "fixed interactions": 4 * 64,
    "decode-request-scope features": 8 * 64,
    "state update (absorb): decays, ring, counters": 3 * 64 + 64 + 4 * 8 + 64,
    "linear score, 45 features x 64 experts": 45 * 64,
    "eviction sort over <=32 candidates (~n log n comparisons)": 32 * 5,
}


def overhead(stage1: dict, stage3: dict, events: int, expert_bytes: int) -> dict:
    per_event = sum(FEATURE_OPS.values())
    layers = 16
    per_token = per_event * layers
    # OLMoE-1B-7B-0924 activates ~1.3B parameters per decode token; a forward pass is
    # about two FLOPs per active parameter.
    active_params = 1.3e9
    forward_flops = 2 * active_params
    rows = {}
    for capacity in DECISION:
        saved = stage1[capacity] - stage3[capacity]
        rows[capacity] = {
            "transfers_avoided_vs_stage1": int(saved),
            "bytes_avoided_vs_stage1": int(saved) * expert_bytes,
            "policy_flops_over_the_run": int(per_event * events),
            "bytes_avoided_per_policy_flop": (int(saved) * expert_bytes) / (per_event * events),
        }
    return {
        "schema_version": "race_paper_policy_overhead_v1",
        "note": ("Analytic arithmetic accounting for the Stage 3 scorer. Counts elementwise "
                 "array operations in the feature block, the linear score and the eviction "
                 "sort; transcendentals are charged 10 operations each."),
        "operation_breakdown_per_same_layer_event": FEATURE_OPS,
        "arithmetic_ops_per_same_layer_event": per_event,
        "arithmetic_ops_per_generated_token": per_token,
        "moe_layers": layers,
        "assumed_active_params_per_token": active_params,
        "assumed_decode_forward_flops_per_token": forward_flops,
        "policy_fraction_of_decode_forward_pass": per_token / forward_flops,
        "expert_bytes": expert_bytes,
        "policy_state_bytes": {
            "markov_tables_float64": 6 * 16 * 64 * 64 * 8,
            "per_expert_feature_state": 15 * 16 * 64 * 8,
            "ring_buffer_bool": 16 * 32 * 64,
            "model_weights": 4 * 45 * 8,
        },
        "expert_weight_bytes_all_layers": 64 * 16 * expert_bytes,
        "evaluation_events_per_variant": events,
        "by_capacity": rows,
        "caveat": ("The measured Python replay throughput of roughly 145 microseconds per "
                   "same-layer event is interpreter-dominated and is NOT the deployable cost. "
                   "It is reported only as an upper bound for a research implementation."),
    }


# ------------------------------------------------------------------------------ main

def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    simple, oracle, regime = stage0_reference()
    workloads = sorted({w for w, _c in simple})
    stage1 = stage_costs(STAGE1 / "results/full/results.jsonl", "method_id",
                         "markov_plus_ewma_h2_beta0.5_alpha0.95")
    stage2 = stage_costs(STAGE2 / "results/full/results.jsonl", "variant", "RACE_ONLINE")
    stage3 = stage_costs(STAGE3 / "results/full/results.jsonl", "variant", "STAGE3_RANKER")
    stage3_lfl = stage_costs(STAGE3 / "results/full/results.jsonl", "variant",
                             "STAGE3_RANKER_NO_REQUEST_SCOPE")

    def total(table, names, capacity):
        return float(sum(table[(n, capacity)] for n in names))

    rows = []
    for scope, names in [("all_frozen_workloads", workloads)] + [
        (r, [w for w in workloads if regime[w] == r]) for r in REGIMES
    ]:
        for capacity in CAPACITIES:
            base = total(simple, names, capacity)
            orc = total(oracle, names, capacity)
            gap = base - orc
            entry = {"scope": scope, "capacity": capacity, "spare_residency": capacity - 8,
                     "stage0_simple": base, "oracle": orc}
            for label, table in (("stage1", stage1), ("stage2", stage2),
                                 ("stage3", stage3), ("stage3_like_for_like", stage3_lfl)):
                cost = total(table, names, capacity)
                entry[f"{label}_cost"] = cost
                entry[f"{label}_gap_closed"] = None if gap == 0 else (base - cost) / gap
                entry[f"{label}_normalized"] = cost / base
            entry["stage3_vs_stage1"] = (entry["stage1_cost"] - entry["stage3_cost"]) / entry["stage1_cost"]
            entry["stage3_vs_stage2"] = (entry["stage2_cost"] - entry["stage3_cost"]) / entry["stage2_cost"]
            rows.append(entry)
    write_csv(PAPER / "results_all_stages.csv", rows, list(rows[0]))

    workload_rows = []
    for name in workloads:
        for capacity in CAPACITIES:
            base = float(simple[(name, capacity)])
            gap = base - oracle[(name, capacity)]
            entry = {"workload": name, "regime": regime[name], "capacity": capacity,
                     "stage0_simple": base, "oracle": float(oracle[(name, capacity)])}
            for label, table in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3)):
                entry[f"{label}_cost"] = float(table[(name, capacity)])
                entry[f"{label}_gap_closed"] = None if gap == 0 else (base - entry[f"{label}_cost"]) / gap
            workload_rows.append(entry)
    write_csv(PAPER / "results_by_workload.csv", workload_rows, list(workload_rows[0]))

    # head-to-head paired comparisons the individual stages never ran
    s1_seq = stage1_per_sequence("markov_plus_ewma_h2_beta0.5_alpha0.95")
    s2_seq = per_sequence(STAGE2 / "results/full/per_sequence_results.jsonl", "variant", "RACE_ONLINE")
    s3_seq = per_sequence(STAGE3 / "results/full/per_sequence_results.jsonl", "variant", "STAGE3_RANKER")
    comparisons = {
        "stage3_vs_stage1": [paired_bootstrap(s1_seq, s3_seq, workloads, c) for c in DECISION],
        "stage3_vs_stage2": [paired_bootstrap(s2_seq, s3_seq, workloads, c) for c in DECISION],
        "stage2_vs_stage1": [paired_bootstrap(s1_seq, s2_seq, workloads, c) for c in DECISION],
    }
    atomic_write_json(PAPER / "stage_comparisons.json", {
        "schema_version": "race_paper_stage_comparisons_v1",
        "note": ("Conditional paired bootstrap over source decode sequences, stratified by "
                 "domain, on the frozen ten-workload suite. Positive improvement means the "
                 "second stage costs less. Intervals are conditional on the frozen workload "
                 "ordering; stateful trajectories are not regenerated."),
        "comparisons": comparisons,
    })

    # per-layer behaviour, recorded by Stage 3 but never analyzed
    layer_rows = []
    for row in read_jsonl(STAGE3 / "results/full/diagnostics.jsonl"):
        if row["variant"] != "STAGE3_RANKER" or not row.get("layer_misses"):
            continue
        for layer, misses in enumerate(row["layer_misses"]):
            layer_rows.append({"workload": row["workload"], "regime": row["regime"],
                               "capacity": int(row["capacity"]), "layer": layer,
                               "stage3_misses": int(misses)})
    write_csv(PAPER / "stage3_per_layer_misses.csv", layer_rows, list(layer_rows[0]))

    events = int(json.loads(next(iter(
        (STAGE3 / "results/full/results.jsonl").open()))) ["events"])
    total_events = sum(
        int(r["events"]) for r in read_jsonl(STAGE3 / "results/full/results.jsonl")
        if r["variant"] == "STAGE3_RANKER" and int(r["capacity"]) == 32)
    expert_bytes = 12582912
    suite = {c: {"stage1": total(stage1, workloads, c), "stage3": total(stage3, workloads, c)}
             for c in DECISION}
    cost_overhead = overhead({c: suite[c]["stage1"] for c in DECISION},
                             {c: suite[c]["stage3"] for c in DECISION},
                             total_events, expert_bytes)
    atomic_write_json(PAPER / "policy_overhead.json", cost_overhead)

    # the paper's main figure: the whole arc in one panel
    suite_rows = [r for r in rows if r["scope"] == "all_frozen_workloads"]
    caps = [r["capacity"] for r in suite_rows]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axis = axes[0]
    axis.plot(caps, [1.0] * len(caps), marker="o", color="0.45", label="Stage 0 strongest simple")
    axis.plot(caps, [r["stage1_normalized"] for r in suite_rows], marker="s", label="Stage 1 fixed predictor")
    axis.plot(caps, [r["stage2_normalized"] for r in suite_rows], marker="^", label="Stage 2 adaptive rank mixing")
    axis.plot(caps, [r["stage3_normalized"] for r in suite_rows], marker="D", linewidth=2.4,
              label="Stage 3 learned ranking")
    axis.plot(caps, [r["oracle"] / r["stage0_simple"] for r in suite_rows], marker="o",
              color="crimson", label="Offline oracle (Belady)")
    axis.set_xlabel("Cache capacity $B$ (experts per layer, top-$k$=8)")
    axis.set_ylabel("Expert transfers / Stage 0 strongest simple")
    axis.set_xticks(caps)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="lower left")
    axis.set_title("(a) Normalized transfer cost")

    axis = axes[1]
    decision_rows = [r for r in suite_rows if r["capacity"] != 8]
    spare = [r["spare_residency"] for r in decision_rows]
    for label, marker, style in (("stage1", "s", "-"), ("stage2", "^", "-"), ("stage3", "D", "-")):
        axis.plot(spare, [100 * r[f"{label}_gap_closed"] for r in decision_rows],
                  marker=marker, linestyle=style, linewidth=2.4 if label == "stage3" else 1.4,
                  label={"stage1": "Stage 1 fixed predictor",
                         "stage2": "Stage 2 adaptive rank mixing",
                         "stage3": "Stage 3 learned ranking"}[label])
    axis.plot(spare, [100 * r["stage3_like_for_like_gap_closed"] for r in decision_rows],
              marker="D", linestyle="--", linewidth=1.2, color="0.35",
              label="Stage 3, Stage-1 information only")
    axis.set_xlabel("Spare residency slots $S = B - 8$")
    axis.set_ylabel("Stage 0 oracle gap closed (%)")
    axis.set_xticks(spare)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("(b) Fraction of the offline-oracle advantage recovered")
    figure.tight_layout()
    figure.savefig(PAPER / "figure_cross_stage.png", dpi=200, bbox_inches="tight")
    figure.savefig(PAPER / "figure_cross_stage.pdf", bbox_inches="tight")
    plt.close(figure)

    headline = {
        "schema_version": "race_paper_headline_v1",
        "suite": "ten frozen Stage 0 evaluation workload paths, 320 disjoint decode sequences",
        "cost_model": "unit expert transfers (misses), lambda = 0",
        "by_capacity": {
            str(r["capacity"]): {
                "stage0_simple": r["stage0_simple"], "stage1": r["stage1_cost"],
                "stage2": r["stage2_cost"], "stage3": r["stage3_cost"], "oracle": r["oracle"],
                "stage1_gap_closed": r["stage1_gap_closed"],
                "stage2_gap_closed": r["stage2_gap_closed"],
                "stage3_gap_closed": r["stage3_gap_closed"],
                "stage3_like_for_like_gap_closed": r["stage3_like_for_like_gap_closed"],
            }
            for r in suite_rows
        },
        "verdicts": {
            "stage0": "RACE_STAGE0_STRONG_GO", "stage1": "RACE_STAGE1_STRONG_GO",
            "stage2": "RACE_STAGE2_NO_GO", "stage3": "RACE_STAGE3_PARTIAL_SUCCESS",
        },
        "archive_hashes": {
            "stage0": "9af9a053502709bff9a33017c4f5b80bc0faa2306bc41d980e7bd2d7274346d2",
            "stage1": "4539cab504052010c32f0b87571adc90884283ac060784ddef822c01768e5d1b",
            "stage2": "a0ec2f31263daa6fb304cdfc0f03247db953ec7f676fe101d680e1785aff4cd3",
            "stage3": "fc8307220b54f4b9b3336fc42abb4a5f8432930133dd54899c356f077c59ace7",
        },
        "trace_logical_hash": "ccec01b2ae5059655e23d7f791427fac75b5fac21e967b9e157bb6087c639dea",
    }
    atomic_write_json(PAPER / "headline_results.json", headline)

    print("wrote:")
    for name in sorted(p.name for p in PAPER.iterdir() if p.is_file()):
        print("  ", name)
    print()
    print("Stage 3 vs Stage 2, paired, suite aggregate:")
    for row in comparisons["stage3_vs_stage2"]:
        print(f"  B={row['capacity']:>2}: {100*row['improvement']:+.2f}% "
              f"[{100*row['ci_low']:.2f}, {100*row['ci_high']:.2f}]  "
              f"{row['sequences_improved']}/{row['paired_units']} sequences improved")
    print()
    print(f"policy arithmetic: {cost_overhead['arithmetic_ops_per_same_layer_event']:,} ops per "
          f"same-layer event, {cost_overhead['arithmetic_ops_per_generated_token']:,} per token")
    print(f"  = {cost_overhead['policy_fraction_of_decode_forward_pass']:.2e} of a decode forward pass")
    for capacity, entry in cost_overhead["by_capacity"].items():
        print(f"  B={capacity:>2}: {entry['bytes_avoided_per_policy_flop']:.1f} bytes of expert "
              f"transfer avoided per policy FLOP")


if __name__ == "__main__":
    main()
