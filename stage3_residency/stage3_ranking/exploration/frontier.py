"""Non-causal diagnostic: how much ranking accuracy does each gap level require?

Blends the deployed causal score with the true next-use ordering inside each
candidate set. Purely a measurement of the achievability frontier; never a policy.
"""
import numpy as np
from harness import *
from features3 import FeatureState3
from collect2 import sub
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences
from residency_headroom.workloads import calibration_frequency_scores

CAPS4 = (12, 16, 24, 32)

def blended(trace, workload, capacities, state, weights, lam):
    L, E, C = trace.num_layers, trace.num_experts, len(capacities)
    req_all = trace.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    res = np.zeros((C, L, E), bool); lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64); mask = np.zeros(E, bool)
    spares = [c - trace.top_k for c in capacities]
    misses = np.zeros(C, np.int64)
    conc = np.zeros(C); tot = np.zeros(C)
    fut = build_future_occurrences(trace, workload, same_layer_indices(trace, workload))
    state.reset()
    for _s, view in workload.iter_slices(trace):
        state.begin_sequence()
        for idx in range(view.start, view.stop):
            o = int(trace.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
            td = fut.advance(o, p, r)
            f = state.features(o, r, gate_all[idx], srt[idx], p)
            state.absorb(o, r, gate_all[idx], p)
            mask[r] = True; lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]; cand = np.flatnonzero(row & ~mask)
                misses[ci] += r.size - int(row[r].sum())
                if cand.size > sp:
                    n = cand.size
                    model = weights[ci] @ f[:, cand]
                    d = np.minimum(td[cand], 33).astype(np.float64)
                    rm = np.argsort(np.argsort(model)) / max(n-1, 1)
                    rt = np.argsort(np.argsort(-d)) / max(n-1, 1)
                    s = (1.0 - lam) * rm + lam * rt
                    better = d[:, None] < d[None, :]; nb = better.sum()
                    if nb:
                        df = s[:, None] - s[None, :]
                        conc[ci] += ((df > 0) & better).sum() + 0.5*((df == 0) & better).sum()
                        tot[ci] += nb
                    ordr = np.lexsort((cand, -lu[o, cand], -s))
                    row.fill(False); row[r] = True; row[cand[ordr[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False; pos[o] += 1
    return ({int(c): int(m) for c, m in zip(capacities, misses)}, conc / np.maximum(tot, 1))

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA'); calB = sub(inputs.calibration, 40, 80, 'calB')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    refs = references(inputs, calB, CAPS4)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses) for x in simulate_causal_capacities(tr, calB, CAPS4, spec, models)}
    from committed import per_capacity_weights
    ws = per_capacity_weights(CAPS4)   # frozen Stage 3 primary, from the committed config
    st = FeatureState3(models, tr.num_layers, tr.num_experts, static)
    need = {}
    for cap in CAPS4:
        s, o = refs['simple'][cap], refs['oracle'][cap]
        need[cap] = (s - 0.90*s1[cap]) / (s - o)
    print("gap closure needed for +10% vs Stage 1 on calB: " +
          " ".join(f"B={c}:{100*need[c]:.1f}%" for c in CAPS4))
    print(f"\n{'lambda':>7} " + " ".join(f"{'B='+str(c):>22}" for c in CAPS4))
    for lam in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        c, acc = blended(tr, calB, CAPS4, st, ws, lam)
        cells = []
        for i, cap in enumerate(CAPS4):
            s, o = refs['simple'][cap], refs['oracle'][cap]
            g = (s - c[cap]) / (s - o)
            hit = "*" if g >= need[cap] else " "
            cells.append(f"acc{100*acc[i]:5.1f} gap{100*g:5.1f}{hit}")
        print(f"{lam:>7.1f} " + " ".join(f"{x:>22}" for x in cells))
    print("\n* marks a capacity where +10% over Stage 1 would be met.")
