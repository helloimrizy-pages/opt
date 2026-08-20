"""Fit ranking models on calibration candidate groups. Calibration data only."""
import numpy as np
from scipy.optimize import minimize
from features import NAMES

def group_slices(G):
    edges = np.flatnonzero(np.diff(G)) + 1
    return list(zip(np.r_[0, edges], np.r_[edges, G.size]))

def build_pairs(X, Y, G, max_pairs=30, seed=0):
    rng = np.random.default_rng(seed)
    out, wts = [], []
    for a, b in group_slices(G):
        y = Y[a:b]; x = X[a:b]
        i, j = np.nonzero(y[:, None] < y[None, :])
        if i.size == 0:
            continue
        if i.size > max_pairs:
            pick = rng.choice(i.size, max_pairs, replace=False)
            i, j = i[pick], j[pick]
        out.append(x[i] - x[j])
        wts.append(np.full(i.size, 1.0 / i.size))
    return np.concatenate(out).astype(np.float64), np.concatenate(wts)

def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd

def fit_pairwise(D, w, l2=1e-4):
    mass = w.sum()
    def obj(theta):
        m = D @ theta
        loss = float(w @ np.logaddexp(0.0, -m)) / mass + 0.5 * l2 * float(theta @ theta)
        grad = -(D.T @ (w / (1.0 + np.exp(m)))) / mass + l2 * theta
        return loss, grad
    res = minimize(obj, np.zeros(D.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10})
    return res.x, float(res.fun)

def pairwise_accuracy(X, Y, G, theta, mu, sd):
    s = ((X - mu) / sd) @ theta
    conc = tot = 0.0
    for a, b in group_slices(G):
        y = Y[a:b]; v = s[a:b]
        better = y[:, None] < y[None, :]
        n = better.sum()
        if not n: continue
        d = v[:, None] - v[None, :]
        conc += ((d > 0) & better).sum() + 0.5 * ((d == 0) & better).sum()
        tot += n
    return conc / tot


def _ensure_groups():
    """Load the collected calibration groups, building them first if absent."""
    import pathlib
    import numpy as _np
    path = pathlib.Path(__file__).with_name('calA_groups.npz')
    if not path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('collect2.py'))],
                       check=True, cwd=str(path.parent))
    return _np.load(path, allow_pickle=True)

if __name__ == "__main__":
    z = _ensure_groups()
    X, Y, G = z['X'].astype(np.float64), z['Y'].astype(np.float64), z['G']
    gs = group_slices(G)
    cut = gs[int(0.75 * len(gs))][0]
    Xtr, Ytr, Gtr = X[:cut], Y[:cut], G[:cut]
    Xho, Yho, Gho = X[cut:], Y[cut:], G[cut:]
    mu, sd = standardize(Xtr)
    Ztr, Zho = (Xtr - mu) / sd, (Xho - mu) / sd
    print(f"train rows {Xtr.shape[0]:,}  holdout rows {Xho.shape[0]:,}")

    # Reference: the frozen Stage 1 winner score, expressed in these features.
    ref = np.zeros(X.shape[1]); ref[NAMES.index("markov_h2")] = 0.5; ref[NAMES.index("freq_a95")] = 0.5*0.05
    print(f"  stage1-winner score holdout pairwise accuracy = "
          f"{100*pairwise_accuracy(Xho, Yho, Gho, ref*sd, np.zeros_like(mu), np.ones_like(sd)):.2f}%")

    D, w = build_pairs(Ztr, Ytr, Gtr)
    print(f"  pairs {D.shape[0]:,}")
    for l2 in (1e-2, 1e-3, 1e-4):
        theta, val = fit_pairwise(D, w, l2)
        acc = pairwise_accuracy(Xho, Yho, Gho, theta, mu, sd)
        print(f"  pairwise-logistic l2={l2:<7g} obj={val:.5f}  holdout accuracy = {100*acc:.2f}%")
        if l2 == 1e-3:
            best = theta
    # least squares on capped distance, for contrast
    A = np.c_[Ztr, np.ones(len(Ztr))]
    coef = np.linalg.lstsq(A, Ytr, rcond=None)[0]
    acc = pairwise_accuracy(Xho, Yho, Gho, -coef[:-1], mu, sd)
    print(f"  least-squares on capped distance      holdout accuracy = {100*acc:.2f}%")
    np.savez('fit_linear.npz', theta=best, mu=mu, sd=sd, ls=-coef[:-1])
    order = np.argsort(-np.abs(best))
    print("\n  strongest standardized pairwise-logistic coefficients:")
    for k in order[:12]:
        print(f"    {NAMES[k]:<16s} {best[k]:+.4f}")
