"""Tent online semantics, matched branching and label-leakage checks."""
from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optstate import adam_state as A          # noqa: E402
from optstate import model as M               # noqa: E402
from optstate import tent_core as T           # noqa: E402
from optstate.diagnostics import boundary_diagnostics, weights_fingerprint  # noqa: E402


def _bn_net(seed: int = 0):
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4),
    )
    return net


def test_configure_matches_tent_reference():
    net = M.configure_tent_model(_bn_net())
    info = M.check_tent_config(net)
    assert info["training"]
    assert info["some_params_require_grad"]
    assert info["not_all_params_require_grad"]
    assert info["has_batchnorm2d"]
    assert info["bn_running_stats_disabled"]
    params, names = M.collect_bn_params(net)
    assert len(params) == 4                      # 2 BN layers x (weight, bias)
    assert all(n.endswith((".weight", ".bias")) for n in names)
    assert all(p.requires_grad for p in params)


def test_prediction_is_recorded_before_the_update():
    net = M.configure_tent_model(_bn_net())
    params, _ = M.collect_bn_params(net)
    opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,))

    with torch.no_grad():
        pre_logits = net(x).clone()
    expected_correct = int((pre_logits.argmax(1) == y).sum())

    rec = T.tent_step(net, opt, x, y, 0, record_post_update=True)
    assert rec.n_correct == expected_correct, "recorded accuracy must be pre-update"

    with torch.no_grad():
        post_logits = net(x)
    assert not torch.equal(pre_logits, post_logits), "the step must have moved weights"
    assert rec.post_update_correct == int((post_logits.argmax(1) == y).sum())


def test_labels_never_enter_the_adaptation_loss():
    """Shuffling the labels must not change a single adapted weight."""
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,))
    y_shuffled = y[torch.randperm(len(y))]

    outs = []
    for labels in (y, y_shuffled):
        net = M.configure_tent_model(_bn_net())
        params, _ = M.collect_bn_params(net)
        opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
        T.tent_step(net, opt, x, labels, 0)
        outs.append(torch.cat([p.detach().reshape(-1) for p in params]))
    assert torch.equal(outs[0], outs[1])


def test_tent_source_matches_the_official_forward_and_adapt_text():
    src = inspect.getsource(T.tent_step)
    order = [src.index(tok) for tok in ("model(x)", "loss.backward()", "optimizer.step()")]
    assert order == sorted(order), "predict -> backward -> step ordering violated"


def test_matched_branches_are_bitwise_identical_and_diverge_only_via_state():
    net = M.configure_tent_model(_bn_net())
    params, _ = M.collect_bn_params(net)
    opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,))
    for i in range(5):                                  # warm up domain A
        T.tent_step(net, opt, x, y, i)

    snap = A.snapshot_adam(opt)
    fingerprint = weights_fingerprint(net)

    xb = torch.randn(16, 3, 8, 8)
    yb = torch.randint(0, 4, (16,))
    first_batch_correct, after = {}, {}
    for name in A.INTERVENTIONS:
        branch = copy.deepcopy(net)
        bp, _ = M.collect_bn_params(branch)
        bopt = A.build_branch_optimizer(bp, snap, name)
        assert weights_fingerprint(branch) == fingerprint, name
        rec = T.tent_step(branch, bopt, xb, yb, 0)
        first_batch_correct[name] = rec.n_correct
        after[name] = torch.cat([p.detach().reshape(-1) for p in bp])

    # identical weights at the boundary => identical first prediction for all
    assert len(set(first_batch_correct.values())) == 1
    # but different optimizer state => different weights after one update
    assert not torch.equal(after["CARRY_ALL"], after["FRESH_ADAM"])
    assert not torch.equal(after["CARRY_ALL"], after["RESET_M_KEEP_V_STEP"])


def test_boundary_diagnostics_do_not_mutate_the_checkpoint():
    net = M.configure_tent_model(_bn_net())
    params, _ = M.collect_bn_params(net)
    opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,))
    for i in range(4):
        T.tent_step(net, opt, x, y, i)
    snap = A.snapshot_adam(opt)
    fp = weights_fingerprint(net)
    m_before = A.flat_state(snap, "exp_avg").clone()

    d = boundary_diagnostics(net, opt, snap, torch.randn(16, 3, 8, 8),
                             torch.randint(0, 4, (16,)))
    assert weights_fingerprint(net) == fp
    assert torch.equal(A.flat_state(A.snapshot_adam(opt), "exp_avg"), m_before)
    assert all(p.grad is None for p in params)
    for key in ("cos_m_g", "grad_norm", "m_norm", "update_cos_neg_g[CARRY_ALL]",
                "update_cos_neg_g[FRESH_ADAM]", "step_prev"):
        assert key in d
    assert -1.0001 <= d["cos_m_g"] <= 1.0001


def test_boundary_diagnostic_update_predicts_the_real_branch_step():
    """update_cos/norm come from maths that matches the branch's actual step."""
    net = M.configure_tent_model(_bn_net())
    params, _ = M.collect_bn_params(net)
    opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 4, (16,))
    for i in range(4):
        T.tent_step(net, opt, x, y, i)
    snap = A.snapshot_adam(opt)
    xb, yb = torch.randn(16, 3, 8, 8), torch.randint(0, 4, (16,))
    d = boundary_diagnostics(net, opt, snap, xb, yb)

    for name in A.INTERVENTIONS:
        branch = copy.deepcopy(net)
        bp, _ = M.collect_bn_params(branch)
        bopt = A.build_branch_optimizer(bp, snap, name)
        before = torch.cat([p.detach().reshape(-1).clone() for p in bp])
        T.tent_step(branch, bopt, xb, yb, 0)
        after = torch.cat([p.detach().reshape(-1) for p in bp])
        real = float((after - before).norm())
        # float32 tolerance: RESET_V_KEEP_M_STEP divides by eps, which magnifies
        # rounding by several orders of magnitude, so compare relatively.
        assert abs(real - d[f"update_norm[{name}]"]) <= 1e-4 * max(1e-6, real), name


def test_matched_branches_see_identical_batches():
    from optstate.data import permutation_for
    a = permutation_for(1, "fog", 5)
    b = permutation_for(1, "fog", 5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, permutation_for(2, "fog", 5))
    assert np.array_equal(permutation_for(0, "fog", 5), np.arange(10000))
