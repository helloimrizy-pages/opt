"""Calibration-only exploration harness for RACE follow-on scorers.

Nothing here reads an evaluation sequence. Every number produced by this file is a
calibration-path measurement used to choose a design before it is frozen.
"""
from __future__ import annotations
import time
import numpy as np
from pathlib import Path


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / '.git').exists() and (candidate / 'stage3_residency').exists():
            return candidate
    raise SystemExit('Could not locate the repository root')


from race_stage1.models import TransitionModels
from race_stage1.simulation import simulate_causal_capacities
from race_stage2.frozen import load_and_verify_stage2_inputs, truncated_workload
from residency_headroom.simulator import simulate_oracle, simulate_policy
from residency_headroom.workloads import calibration_frequency_scores

ROOT = _repository_root()
PREREG = ROOT / 'stage3_residency/stage2_race/configs/stage2_preregistered.json'
CAPS = (12, 16, 24, 32)


def load():
    inputs = load_and_verify_stage2_inputs(ROOT, PREREG)
    models = TransitionModels.load(
        ROOT / 'stage3_residency/stage2_race/results/calibration/transition_models.npz')
    return inputs, models


def simulate_scorer(trace, workload, capacities, scorer):
    """Frozen Stage 1 eviction mechanism driven by an arbitrary causal scorer.

    `scorer.step(layer, request, gates, sorted_request) -> (num_experts,)` where a
    LARGER value means retain more strongly. State is per-layer and internal.
    """
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {l: i for i, l in enumerate(layers)}
    E, L, C = trace.num_experts, trace.num_layers, len(capacities)
    req_all = trace.requested_expert_ids.astype(np.int64, copy=False)
    srt_all = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64, copy=False)
    resident = np.zeros((C, L, E), dtype=bool)
    last_used = np.full((L, E), -1, dtype=np.int64)
    pos = np.zeros(L, dtype=np.int64)
    mask = np.zeros(E, dtype=bool)
    spares = [c - trace.top_k for c in capacities]
    misses = np.zeros(C, dtype=np.int64)
    requests = 0
    scorer.reset()
    for _seq, view in workload.iter_slices(trace):
        if hasattr(scorer, "begin_sequence"):
            scorer.begin_sequence()
        for idx in range(view.start, view.stop):
            o = l2o[int(trace.layer_index[idx])]
            req = req_all[idx]
            p = int(pos[o])
            scores = scorer.step(o, req, gate_all[idx], srt_all[idx], p)
            mask[req] = True
            last_used[o, req] = p
            requests += req.size
            for ci in range(C):
                row = resident[ci, o]
                hit = int(row[req].sum())
                misses[ci] += req.size - hit
                cand = np.flatnonzero(row & ~mask)
                sp = spares[ci]
                if sp <= 0:
                    row.fill(False); row[req] = True
                elif cand.size > sp:
                    s = scores[cand]
                    order = np.lexsort((cand, -last_used[o, cand], -s))
                    row.fill(False); row[req] = True; row[cand[order[:sp]]] = True
                else:
                    row[req] = True
            mask[req] = False
            pos[o] += 1
    return {int(c): int(m) for c, m in zip(capacities, misses)}, requests


def references(inputs, workload, capacities):
    out = {}
    out['oracle'] = {c: int(simulate_oracle(inputs.trace, workload, c).misses) for c in capacities}
    static = calibration_frequency_scores(inputs.trace, inputs.calibration).astype(np.float64)
    simple = {}
    for name, kw in (('lru', {}), ('lfu', {}), ('lfu_decay', {'alpha': 0.95}),
                     ('static_hotset', {'static_scores': static})):
        simple[name] = {c: int(simulate_policy(inputs.trace, workload, c, name, **kw).misses)
                        for c in capacities}
    out['simple_all'] = simple
    out['simple'] = {c: min(simple[n][c] for n in simple) for c in capacities}
    out['simple_which'] = {c: min(simple, key=lambda n: simple[n][c]) for c in capacities}
    return out


def report(name, costs, refs, capacities, elapsed=None):
    parts = []
    gaps = []
    for c in capacities:
        s, o = refs['simple'][c], refs['oracle'][c]
        g = (s - costs[c]) / (s - o) if s != o else float('nan')
        gaps.append(g)
        parts.append(f"{c}:{costs[c]:>7d}({100*g:5.1f}%)")
    tail = f"  [{elapsed:.0f}s]" if elapsed else ""
    print(f"{name:<44s} " + " ".join(parts) + f"  mean_gap={100*np.mean(gaps):5.2f}%" + tail)
    return float(np.mean(gaps))
