import time, numpy as np
from harness import *
from features2 import FeatureState2, NAMES
from collect2 import sub
from fit import build_pairs, standardize, fit_pairwise, pairwise_accuracy, group_slices
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences
from residency_headroom.workloads import calibration_frequency_scores

CAPS4 = (12, 16, 24, 32)

def collect(trace, workload, models, static, stride=13):
    st = FeatureState2(models, trace.num_layers, trace.num_experts, static)
    fut = build_future_occurrences(trace, workload, same_layer_indices(trace, workload))
    req_all = trace.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    L, E, C = trace.num_layers, trace.num_experts, len(CAPS4)
    res = np.zeros((C, L, E), bool); lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64); mask = np.zeros(E, bool)
    hist = np.zeros((L, E)); M2 = np.stack([models.matrix(2, l) for l in range(L)])
    spares = [c - trace.top_k for c in CAPS4]
    X, Y, G, SP = [], [], [], []; gid = 0; n = 0
    for _s, view in workload.iter_slices(trace):
        for idx in range(view.start, view.stop):
            o = int(trace.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
            td = fut.advance(o, p, r)
            sample = (n % stride == 0) and p >= 40
            f = st.features(o, r, gate_all[idx], srt[idx], p) if sample else None
            cond = M2[o][srt[idx]].mean(axis=0)
            hist[o] *= 0.95; hist[o][r] += 0.05
            score = 0.5*cond + 0.5*hist[o]
            st.absorb(o, r, gate_all[idx], p)
            mask[r] = True; lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]; cand = np.flatnonzero(row & ~mask)
                if cand.size > sp:
                    if sample and cand.size >= 4:
                        X.append(f[:, cand].T.astype(np.float32))
                        Y.append(np.minimum(td[cand], 33).astype(np.float32))
                        G.append(np.full(cand.size, gid, np.int32))
                        SP.append(np.full(cand.size, sp, np.int16)); gid += 1
                    ordr = np.lexsort((cand, -lu[o, cand], -score[cand]))
                    row.fill(False); row[r] = True; row[cand[ordr[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False; pos[o] += 1; n += 1
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(SP)

class Deployed2:
    def __init__(self, state, w, name):
        self.state, self.w, self.name = state, np.asarray(w, float), name
    def reset(self): self.state.reset()
    def step(self, layer, request, gates, sorted_request, position):
        f = self.state.features(layer, request, gates, sorted_request, position)
        out = self.w @ f
        self.state.absorb(layer, request, gates, position)
        return out

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA'); calB = sub(inputs.calibration, 40, 80, 'calB')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    t0 = time.perf_counter(); X, Y, G, SP = collect(tr, calA, models, static)
    print(f"{X.shape[0]:,} rows, {X.shape[1]} features, {G.max()+1:,} groups  [{time.perf_counter()-t0:.0f}s]")
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    gs = group_slices(G); cut = gs[int(0.75*len(gs))][0]
    mu, sd = standardize(X[:cut]); Z = (X - mu)/sd
    D, w = build_pairs(Z[:cut], Y[:cut], G[:cut], max_pairs=30)
    print(f"pairs {D.shape[0]:,}")
    best, bestacc = None, -1
    for l2 in (3e-2, 1e-2, 3e-3, 1e-3):
        th, val = fit_pairwise(D, w, l2)
        acc = pairwise_accuracy(X[cut:], Y[cut:], G[cut:], th, mu, sd)
        print(f"  l2={l2:<7g} obj={val:.5f} holdout acc={100*acc:.2f}%")
        if acc > bestacc: best, bestacc = th, acc
    np.savez('fit2.npz', theta=best, mu=mu, sd=sd)
    refs = references(inputs, calB, CAPS)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses) for x in simulate_causal_capacities(tr, calB, CAPS, spec, models)}
    print(); report("STAGE1 winner", s1, refs, CAPS)
    st = FeatureState2(models, tr.num_layers, tr.num_experts, static)
    t0 = time.perf_counter()
    c, _ = simulate_scorer(tr, calB, CAPS, Deployed2(st, best/sd, 'expanded-linear'))
    report("expanded pairwise-logistic linear", c, refs, CAPS, time.perf_counter()-t0)
    print("\nvs Stage 1:", {k: f"{100*(s1[k]-c[k])/s1[k]:+.2f}%" for k in CAPS})
    o = np.argsort(-np.abs(best))
    print("\ntop coefficients:", ", ".join(f"{NAMES[k]}={best[k]:+.3f}" for k in o[:14]))
