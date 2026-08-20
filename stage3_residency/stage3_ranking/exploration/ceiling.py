"""Is the ~67% pairwise-accuracy plateau a feature limit or a linear-form limit?"""
import time, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from fit import group_slices, standardize, pairwise_accuracy, build_pairs, fit_pairwise
from features2 import NAMES

z = _ensure_groups()   # 45-feature candidate-set groups, built on demand
X19, Y, G = z['X'].astype(np.float64), z['Y'].astype(np.float64), z['G']
gs = group_slices(G); cut = gs[int(0.75*len(gs))][0]

def group_rank_target(Y, G):
    """Within-group normalized rank of the true distance: a pure ordering target."""
    out = np.empty_like(Y)
    for a, b in group_slices(G):
        y = Y[a:b]
        out[a:b] = (np.argsort(np.argsort(y)) / max(len(y) - 1, 1))
    return out

for tag, X in (("19-feature set", X19),):
    Xtr, Ytr, Gtr = X[:cut], Y[:cut], G[:cut]
    Xho, Yho, Gho = X[cut:], Y[cut:], G[cut:]
    mu, sd = standardize(Xtr)
    for target_name, Ttr in (("capped distance", Ytr),
                             ("within-group rank", group_rank_target(Ytr, Gtr))):
        for leaves, iters in ((31, 200), (63, 400)):
            t0 = time.perf_counter()
            m = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=iters,
                                              learning_rate=0.1, l2_regularization=1.0,
                                              early_stopping=False, random_state=0)
            m.fit(Xtr, Ttr)
            pred = m.predict(Xho)
            conc = tot = 0.0
            for a, b in group_slices(Gho):
                yy = Yho[a:b]
                vv = -pred[a:b]
                better = yy[:, None] < yy[None, :]
                n = better.sum()
                if not n: continue
                d = vv[:, None] - vv[None, :]
                conc += ((d > 0) & better).sum() + 0.5*((d == 0) & better).sum(); tot += n
            print(f"  GBT {tag} target={target_name:<18s} leaves={leaves:<3d} iters={iters:<4d}"
                  f" holdout acc={100*conc/tot:.2f}%  [{time.perf_counter()-t0:.0f}s]")
