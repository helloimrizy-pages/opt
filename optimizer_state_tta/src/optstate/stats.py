"""Cluster bootstrap and rank-association helpers.

The experimental unit is a *transition* (one ordered domain pair inside one
corruption order).  Seeds are replicates *within* a transition, so the bootstrap
resamples transitions, never individual images or batches.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def cluster_bootstrap_mean(values_by_cluster: Dict[str, Sequence[float]],
                           n_boot: int = 10_000, alpha: float = 0.05,
                           seed: int = 12345) -> Dict[str, float]:
    """Bootstrap the grand mean, resampling clusters with replacement.

    Each cluster contributes the mean of its replicates.
    """
    keys = sorted(values_by_cluster)
    per_cluster = np.array([
        float(np.nanmean(np.asarray(values_by_cluster[k], dtype=float)))
        for k in keys
    ])
    per_cluster = per_cluster[np.isfinite(per_cluster)]
    n = len(per_cluster)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "median": float("nan"), "n_clusters": 0, "frac_positive": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = per_cluster[idx].mean(axis=1)
    return {
        "mean": float(per_cluster.mean()),
        "median": float(np.median(per_cluster)),
        "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n_clusters": int(n),
        "frac_positive": float((per_cluster > 0).mean()),
        "sd_clusters": float(per_cluster.std(ddof=1)) if n > 1 else float("nan"),
    }


def paired_wilcoxon(values_by_cluster: Dict[str, Sequence[float]]) -> Dict[str, float]:
    from scipy import stats as sps
    per_cluster = np.array([
        float(np.nanmean(np.asarray(v, dtype=float))) for v in values_by_cluster.values()
    ])
    per_cluster = per_cluster[np.isfinite(per_cluster)]
    if len(per_cluster) < 6 or np.allclose(per_cluster, 0):
        return {"statistic": float("nan"), "p_value": float("nan"), "n": len(per_cluster)}
    try:
        res = sps.wilcoxon(per_cluster, alternative="two-sided", zero_method="wilcox")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
                "n": len(per_cluster)}
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": len(per_cluster)}


def spearman_with_ci(x: Sequence[float], y: Sequence[float],
                     clusters: Optional[Sequence[str]] = None,
                     n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 999) -> Dict[str, float]:
    """Spearman rho with a cluster bootstrap CI (clusters = transitions)."""
    from scipy import stats as sps
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    cl = np.asarray(clusters)[ok] if clusters is not None else np.arange(len(x)).astype(str)
    if len(x) < 4:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(len(x)),
                "ci_low": float("nan"), "ci_high": float("nan")}
    res = sps.spearmanr(x, y)
    uniq = sorted(set(cl.tolist()))
    groups = {u: np.where(cl == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([groups[uniq[i]] for i in pick])
        if len(set(x[idx])) < 3 or len(set(y[idx])) < 3:
            continue
        boots.append(float(sps.spearmanr(x[idx], y[idx]).statistic))
    ci = (float(np.percentile(boots, 100 * alpha / 2)),
          float(np.percentile(boots, 100 * (1 - alpha / 2)))) if boots else (float("nan"),) * 2
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(x)),
            "ci_low": ci[0], "ci_high": ci[1], "n_boot_used": len(boots)}


def diff_of_means_ci(a_by_cluster: Dict[str, Sequence[float]],
                     b_by_cluster: Dict[str, Sequence[float]],
                     n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 4242) -> Dict[str, float]:
    """Unpaired difference (a - b) with independent cluster bootstraps."""
    ka, kb = sorted(a_by_cluster), sorted(b_by_cluster)
    va = np.array([float(np.nanmean(np.asarray(a_by_cluster[k], float))) for k in ka])
    vb = np.array([float(np.nanmean(np.asarray(b_by_cluster[k], float))) for k in kb])
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    if len(va) == 0 or len(vb) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    ba = va[rng.integers(0, len(va), size=(n_boot, len(va)))].mean(axis=1)
    bb = vb[rng.integers(0, len(vb), size=(n_boot, len(vb)))].mean(axis=1)
    d = ba - bb
    return {
        "diff": float(va.mean() - vb.mean()),
        "ci_low": float(np.percentile(d, 100 * alpha / 2)),
        "ci_high": float(np.percentile(d, 100 * (1 - alpha / 2))),
        "n_a": int(len(va)), "n_b": int(len(vb)),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
    }
