"""Round 2: per-capacity models trained on the deployed policy's own trajectory."""
import time, numpy as np
from harness import *
from features3 import FeatureState3, NAMES
from collect2 import sub
from fit import group_slices, standardize, build_pairs, fit_pairwise
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences
from residency_headroom.workloads import calibration_frequency_scores
from sklearn.ensemble import HistGradientBoostingRegressor

CAPS4 = (12, 16, 24, 32)

def simulate_multi(trace, workload, capacities, state, scorers, collect_to=None, stride=13):
    """One score vector per capacity per event; optionally collect training rows."""
    L, E, C = trace.num_layers, trace.num_experts, len(capacities)
    req_all = trace.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    res = np.zeros((C, L, E), bool); lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64); mask = np.zeros(E, bool)
    spares = [c - trace.top_k for c in capacities]
    misses = np.zeros(C, np.int64)
    fut = build_future_occurrences(trace, workload, same_layer_indices(trace, workload)) \
        if collect_to is not None else None
    state.reset(); n = 0
    for _s, view in workload.iter_slices(trace):
        state.begin_sequence()
        for idx in range(view.start, view.stop):
            o = int(trace.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
            td = fut.advance(o, p, r) if fut is not None else None
            f = state.features(o, r, gate_all[idx], srt[idx], p)
            scores = [sc(f) for sc in scorers]
            state.absorb(o, r, gate_all[idx], p)
            sample = collect_to is not None and (n % stride == 0) and p >= 40
            mask[r] = True; lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]; cand = np.flatnonzero(row & ~mask)
                misses[ci] += r.size - int(row[r].sum())
                if cand.size > sp:
                    if sample and cand.size >= 4:
                        collect_to[ci][0].append(f[:, cand].T.astype(np.float32))
                        collect_to[ci][1].append(np.minimum(td[cand], 33).astype(np.float32))
                        collect_to[ci][2].append(np.full(cand.size, collect_to[ci][3], np.int32))
                        collect_to[ci][3] += 1
                    s = scores[ci][cand]
                    ordr = np.lexsort((cand, -lu[o, cand], -s))
                    row.fill(False); row[r] = True; row[cand[ordr[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False; pos[o] += 1; n += 1
    return {int(c): int(m) for c, m in zip(capacities, misses)}

def rank_target(Y, G):
    out = np.empty_like(Y)
    for a, b in group_slices(G):
        out[a:b] = np.argsort(np.argsort(Y[a:b])) / max(b - a - 1, 1)
    return out

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA'); calB = sub(inputs.calibration, 40, 80, 'calB')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    refs = references(inputs, calB, CAPS)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses) for x in simulate_causal_capacities(tr, calB, CAPS, spec, models)}
    report("STAGE1 winner", s1, refs, CAPS)
    from committed import seed_weights
    w0 = seed_weights()   # round-1 pooled model, from the committed calibration selection
    st = FeatureState3(models, tr.num_layers, tr.num_experts, static)

    # ---- round 2: collect under the round-1 linear policy, per capacity ----
    buckets = [[[], [], [], 0] for _ in CAPS4]
    t0 = time.perf_counter()
    simulate_multi(tr, calA, CAPS4, st, [lambda f: w0 @ f] * 4, collect_to=buckets)
    print(f"round-2 collection {time.perf_counter()-t0:.0f}s")

    linmods, gbtmods = [], []
    for ci, cap in enumerate(CAPS4):
        X = np.concatenate(buckets[ci][0]).astype(np.float64)
        Y = np.concatenate(buckets[ci][1]).astype(np.float64)
        G = np.concatenate(buckets[ci][2])
        gs = group_slices(G); cut = gs[int(0.8*len(gs))][0]
        mu, sd = standardize(X[:cut]); Z = (X - mu)/sd
        D, wp = build_pairs(Z[:cut], Y[:cut], G[:cut], max_pairs=30)
        th, _ = fit_pairwise(D, wp, 3e-3)
        linmods.append(th / sd)
        m = HistGradientBoostingRegressor(max_leaf_nodes=31, max_iter=250, learning_rate=0.1,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)
        m.fit(X[:cut], rank_target(Y[:cut], G[:cut]))
        gbtmods.append(m)
        # holdout accuracy
        pred = m.predict(X[cut:]); lin = Z[cut:] @ th
        for tag, v in (("lin", lin), ("gbt", -pred)):
            conc = tot = 0.0
            for a, b in group_slices(G[cut:]):
                yy = Y[cut:][a:b]; vv = v[a:b]
                better = yy[:, None] < yy[None, :]; nb = better.sum()
                if not nb: continue
                d = vv[:, None] - vv[None, :]
                conc += ((d > 0) & better).sum() + 0.5*((d == 0) & better).sum(); tot += nb
            print(f"  cap {cap:<2d} {tag} holdout acc={100*conc/tot:.2f}%  rows={X.shape[0]:,}")

    t0 = time.perf_counter()
    c = simulate_multi(tr, calB, CAPS4, st, [(lambda w: (lambda f: w @ f))(w) for w in linmods])
    report("round-2 per-capacity linear", c, refs, CAPS4, time.perf_counter()-t0)
    print("   vs Stage 1:", {k: f"{100*(s1[k]-c[k])/s1[k]:+.2f}%" for k in CAPS4})
    t0 = time.perf_counter()
    c = simulate_multi(tr, calB, CAPS4, st, [(lambda m: (lambda f: -m.predict(f.T)))(m) for m in gbtmods])
    report("round-2 per-capacity GBT", c, refs, CAPS4, time.perf_counter()-t0)
    print("   vs Stage 1:", {k: f"{100*(s1[k]-c[k])/s1[k]:+.2f}%" for k in CAPS4})
    np.savez('fit4_linear.npz', **{f"w{c}": w for c, w in zip(CAPS4, linmods)})
    import pickle; pickle.dump(gbtmods, open('fit4_gbt.pkl','wb'))
