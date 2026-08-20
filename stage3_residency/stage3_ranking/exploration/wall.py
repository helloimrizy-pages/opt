"""Is ~69% pairwise accuracy an information wall or a data/capacity limit?

Scales training data 13x and model capacity 8x. If accuracy does not move, the
limit is the information in causal routing history, not the estimator.
"""
import time, numpy as np
from harness import *
from features3 import FeatureState3
from collect2 import sub
from fit import group_slices, standardize, build_pairs, fit_pairwise
from run4 import simulate_multi
from residency_headroom.workloads import calibration_frequency_scores
from sklearn.ensemble import HistGradientBoostingRegressor

CAPS4 = (12, 16, 24, 32)

def acc_of(pred_desc_is_retain, Y, G):
    conc = tot = 0.0
    for a, b in group_slices(G):
        yy = Y[a:b]; vv = pred_desc_is_retain[a:b]
        better = yy[:, None] < yy[None, :]; n = better.sum()
        if not n: continue
        d = vv[:, None] - vv[None, :]
        conc += ((d > 0) & better).sum() + 0.5*((d == 0) & better).sum(); tot += n
    return conc / tot

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    from committed import seed_weights
    w0 = seed_weights()   # round-1 pooled model, from the committed calibration selection
    st = FeatureState3(models, tr.num_layers, tr.num_experts, static)
    for stride in (13, 3, 1):
        buckets = [[[], [], [], 0] for _ in CAPS4]
        t0 = time.perf_counter()
        simulate_multi(tr, calA, CAPS4, st, [lambda f: w0 @ f]*4, collect_to=buckets, stride=stride)
        ci = 2   # capacity 24
        X = np.concatenate(buckets[ci][0]).astype(np.float64)
        Y = np.concatenate(buckets[ci][1]).astype(np.float64)
        G = np.concatenate(buckets[ci][2])
        gs = group_slices(G); cut = gs[int(0.8*len(gs))][0]
        rank = np.empty(cut)
        for a, b in group_slices(G[:cut]):
            rank[a:b] = np.argsort(np.argsort(Y[a:b]))/max(b-a-1, 1)
        mu, sd = standardize(X[:cut])
        D, wp = build_pairs((X[:cut]-mu)/sd, Y[:cut], G[:cut], max_pairs=30)
        th, _ = fit_pairwise(D, wp, 3e-3)
        lin = acc_of(((X[cut:]-mu)/sd) @ th, Y[cut:], G[cut:])
        line = f"stride={stride:<3d} rows={X.shape[0]:>8,d}  linear={100*lin:.2f}%"
        for leaves, iters in ((31, 250), (127, 800)):
            m = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=iters,
                learning_rate=0.06, l2_regularization=1.0, early_stopping=False,
                min_samples_leaf=40, random_state=0)
            m.fit(X[:cut], rank)
            a = acc_of(-m.predict(X[cut:]), Y[cut:], G[cut:])
            line += f"  gbt({leaves},{iters})={100*a:.2f}%"
        print(line + f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
