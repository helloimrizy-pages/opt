"""Metric definitions and the cluster bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optstate import metrics as MET   # noqa: E402
from optstate import stats as S       # noqa: E402
from optstate.tent_core import BatchRecord  # noqa: E402


def recs(accs, n=200):
    return [BatchRecord(batch_index=i, n=n, n_correct=int(round(a * n)),
                        entropy_loss=1.0, mean_pred_entropy=1.0, grad_norm=1.0)
            for i, a in enumerate(accs)]


def test_early10_is_sample_weighted_over_first_ten_batches():
    r = recs([0.5] * 10 + [1.0] * 40)
    assert MET.window_accuracy(r, 10) == pytest.approx(0.5)
    assert MET.window_accuracy(r, 50) == pytest.approx(0.9)
    assert MET.window_accuracy(r, 1) == pytest.approx(0.5)
    m = MET.window_metrics(r)
    assert m["acc_first10"] == pytest.approx(0.5)
    assert m["err_first10"] == pytest.approx(0.5)
    assert m["cumulative_errors_first10"] == pytest.approx(1000.0)


def test_window_accuracy_requires_a_full_window():
    assert MET.window_accuracy(recs([0.5] * 3), 10) is None


def test_recovery_rule_is_threshold_driven_and_censors():
    curve = [0.2, 0.3, 0.4, 0.7, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85]
    r = recs(curve)
    plateau = MET.plateau_accuracy(r, last=5)
    k = MET.recovery_batch(r, plateau - MET.RECOVERY_TOLERANCE)
    assert 1 <= k <= len(curve)
    assert MET.recovery_batch(recs([0.1] * 10), 0.9) == MET.RECOVERY_CENSORED


def test_collapse_indicator():
    assert MET.collapse_indicator(recs([0.1] * 10))["collapsed"] == 1.0
    assert MET.collapse_indicator(recs([0.8] * 10))["collapsed"] == 0.0


def test_cluster_bootstrap_resamples_clusters_not_replicates():
    """Correlated replicates inside two clusters must NOT buy false precision."""
    rng = np.random.default_rng(0)
    # 100 observations, but they come from only two clusters with a large
    # between-cluster gap: naive pooling would report a very tight interval.
    lumpy = {"c0": list(1.0 + rng.normal(0, 0.01, 50)),
             "c1": list(-1.0 + rng.normal(0, 0.01, 50))}
    clustered = S.cluster_bootstrap_mean(lumpy, n_boot=4000)
    pooled_vals = np.array(lumpy["c0"] + lumpy["c1"])
    naive = S.cluster_bootstrap_mean({f"o{i}": [v] for i, v in enumerate(pooled_vals)},
                                     n_boot=4000)
    assert clustered["n_clusters"] == 2 and naive["n_clusters"] == 100
    assert (clustered["ci_high"] - clustered["ci_low"]) > \
           (naive["ci_high"] - naive["ci_low"])
    # the honest interval must span the two cluster means
    assert clustered["ci_low"] < -0.9 and clustered["ci_high"] > 0.9


def test_cluster_bootstrap_recovers_a_known_mean():
    vals = {f"c{i}": [float(i % 5) - 2.0] for i in range(200)}
    res = S.cluster_bootstrap_mean(vals, n_boot=4000)
    assert res["mean"] == pytest.approx(0.0, abs=1e-9)
    assert res["ci_low"] < 0.0 < res["ci_high"]
    assert res["frac_positive"] == pytest.approx(0.4)


def test_spearman_sign_and_ci():
    x = list(np.linspace(-1, 1, 60))
    y = [-v + 0.01 * ((i % 7) - 3) for i, v in enumerate(x)]
    res = S.spearman_with_ci(x, y, clusters=[str(i // 3) for i in range(60)], n_boot=800)
    assert res["rho"] < -0.9
    assert res["ci_high"] < 0


def test_diff_of_means_ci_detects_a_real_gap():
    a = {f"a{i}": [2.0 + 0.05 * i] for i in range(40)}
    b = {f"b{i}": [0.0 + 0.05 * i] for i in range(40)}
    res = S.diff_of_means_ci(a, b, n_boot=2000)
    assert res["diff"] == pytest.approx(2.0, abs=1e-6)
    assert res["ci_low"] > 0


def test_nan_values_are_dropped_not_treated_as_zero():
    vals = {"a": [1.0], "b": [float("nan")], "c": [3.0]}
    res = S.cluster_bootstrap_mean(vals, n_boot=500)
    assert res["n_clusters"] == 2
    assert res["mean"] == pytest.approx(2.0)
