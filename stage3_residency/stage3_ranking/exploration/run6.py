"""Boundary-weighted training: only comparisons that straddle the retention cutoff
can change an eviction, so train on those."""
import time, numpy as np
from harness import *
from features3 import FeatureState3
from collect2 import sub
from fit import group_slices, standardize, fit_pairwise
from run4 import simulate_multi
from residency_headroom.workloads import calibration_frequency_scores

CAPS4 = (12, 16, 24, 32)

def boundary_pairs(Z, Y, G, spare, max_pairs=40, band=None, seed=0):
    """Pairs (a,b) with y_a<y_b, weighted by proximity to the true retention cutoff."""
    rng = np.random.default_rng(seed)
    blocks, weights = [], []
    for a, b in group_slices(G):
        y = Y[a:b]; x = Z[a:b]; n = b - a
        if n <= spare:
            continue
        order = np.argsort(y, kind='stable')
        rank = np.empty(n, np.int64); rank[order] = np.arange(n)
        i, j = np.nonzero(y[:, None] < y[None, :])
        if i.size == 0:
            continue
        # distance of each pair from the cutoff that separates retained from evicted
        dist = np.maximum(np.abs(rank[i] - (spare - 0.5)), np.abs(rank[j] - (spare - 0.5)))
        if band is None:
            w = np.ones(i.size)
        else:
            w = np.exp(-dist / band)
        keep = w > 1e-3
        i, j, w = i[keep], j[keep], w[keep]
        if i.size == 0:
            continue
        if i.size > max_pairs:
            p = w / w.sum()
            pick = rng.choice(i.size, max_pairs, replace=False, p=p)
            i, j, w = i[pick], j[pick], np.ones(max_pairs)
        blocks.append(x[i] - x[j])
        weights.append(w / w.sum())
    return np.concatenate(blocks).astype(np.float64), np.concatenate(weights)

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA'); calB = sub(inputs.calibration, 40, 80, 'calB')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    refs = references(inputs, calB, CAPS4)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses) for x in simulate_causal_capacities(tr, calB, CAPS4, spec, models)}
    need = {c: (refs['simple'][c] - 0.90*s1[c])/(refs['simple'][c]-refs['oracle'][c]) for c in CAPS4}
    report("STAGE1 winner", s1, refs, CAPS4)
    print("  need for +10%: " + " ".join(f"B={c}:{100*need[c]:.1f}%" for c in CAPS4), "\n")
    from committed import seed_weights
    w0 = seed_weights()   # round-1 pooled model, from the committed calibration selection
    st = FeatureState3(models, tr.num_layers, tr.num_experts, static)
    buckets = [[[], [], [], 0] for _ in CAPS4]
    simulate_multi(tr, calA, CAPS4, st, [lambda f: w0 @ f]*4, collect_to=buckets)
    data = [(np.concatenate(b[0]).astype(np.float64), np.concatenate(b[1]).astype(np.float64),
             np.concatenate(b[2])) for b in buckets]
    for band in (None, 8.0, 4.0, 2.0, 1.0, 0.5):
        ws = []
        for ci, cap in enumerate(CAPS4):
            X, Y, G = data[ci]
            mu, sd = standardize(X)
            D, wp = boundary_pairs((X-mu)/sd, Y, G, cap - tr.top_k, band=band)
            th, _ = fit_pairwise(D, wp, 3e-3)
            ws.append(th/sd)
        c = simulate_multi(tr, calB, CAPS4, st, [(lambda w: (lambda f: w @ f))(w) for w in ws])
        tag = "all pairs" if band is None else f"boundary band={band}"
        report(f"{tag}", c, refs, CAPS4)
        imp = {k: 100*(s1[k]-c[k])/s1[k] for k in CAPS4}
        passed = sum(1 for k in CAPS4 if imp[k] >= 10.0)
        print(f"      vs Stage 1: " + " ".join(f"{k}:{v:+.2f}%" for k, v in imp.items())
              + f"   [{passed}/4 at +10%]")
        np.savez(f'fit6_band{band}.npz', **{f"w{c}": w for c, w in zip(CAPS4, ws)})
