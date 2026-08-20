"""Unit tests for the Adam state instrumentation (Stage 1, section 7)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optstate import adam_state as A  # noqa: E402


def _tiny_setup(seed: int = 0, n_steps: int = 7):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999),
                           weight_decay=0.0)
    x = torch.randn(16, 6)
    for _ in range(n_steps):
        loss = model(x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, opt, x


def test_snapshot_roundtrip_is_exact():
    model, opt, _ = _tiny_setup()
    snap = A.snapshot_adam(opt)
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    A.restore_adam(opt2, snap)
    for p in model.parameters():
        s1, s2 = opt.state[p], opt2.state[p]
        assert torch.equal(s1["exp_avg"], s2["exp_avg"])
        assert torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"])
        assert float(s1["step"]) == float(s2["step"])


@pytest.mark.parametrize("name", A.INTERVENTIONS)
def test_intervention_never_changes_model_parameters(name):
    """Section 7: no intervention may change model parameters at the boundary."""
    model, opt, _ = _tiny_setup()
    snap = A.snapshot_adam(opt)
    before = [p.detach().clone() for p in model.parameters()]
    branch = copy.deepcopy(model)
    _ = A.build_branch_optimizer(list(branch.parameters()), snap, name)
    for p_before, p_branch, p_master in zip(before, branch.parameters(), model.parameters()):
        assert torch.equal(p_before, p_branch), f"{name} altered branch weights"
        assert torch.equal(p_before, p_master), f"{name} altered master weights"


@pytest.mark.parametrize("name,expect", [
    ("CARRY_ALL", dict(m=False, v=False, step=False)),
    ("RESET_M_KEEP_V_STEP", dict(m=True, v=False, step=False)),
    ("RESET_V_KEEP_M_STEP", dict(m=False, v=True, step=False)),
    ("RESET_MV_KEEP_STEP", dict(m=True, v=True, step=False)),
    ("RESET_STEP_ONLY", dict(m=False, v=False, step=True)),
    ("FRESH_ADAM", dict(m=True, v=True, step=True)),
])
def test_intervention_semantics(name, expect):
    model, opt, _ = _tiny_setup()
    snap = A.snapshot_adam(opt)
    out = A.transform_snapshot(snap, name)
    for src, dst in zip(snap.entries, out.entries):
        if expect["m"]:
            assert torch.all(dst["exp_avg"] == 0)
        else:
            assert torch.equal(src["exp_avg"], dst["exp_avg"])
        if expect["v"]:
            assert torch.all(dst["exp_avg_sq"] == 0)
        else:
            assert torch.equal(src["exp_avg_sq"], dst["exp_avg_sq"])
        if expect["step"]:
            assert float(dst["step"]) == 0.0
        else:
            assert float(src["step"]) == float(dst["step"])
    # the source snapshot must be untouched
    m0 = A.flat_state(A.snapshot_adam(opt), "exp_avg")
    assert torch.equal(m0, A.flat_state(snap, "exp_avg"))


def test_fresh_adam_matches_reset_mv_and_step_zero():
    """FRESH_ADAM (empty state) == exp_avg=exp_avg_sq=0 and step=0."""
    model, opt, x = _tiny_setup()
    snap = A.snapshot_adam(opt)

    def one_step(branch_model, optimizer):
        loss = branch_model(x).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return [p.detach().clone() for p in branch_model.parameters()]

    b1 = copy.deepcopy(model)
    o1 = A.build_branch_optimizer(list(b1.parameters()), snap, "FRESH_ADAM")
    w1 = one_step(b1, o1)

    b2 = copy.deepcopy(model)
    o2 = torch.optim.Adam(b2.parameters(), lr=snap.hyper["lr"],
                          betas=snap.hyper["betas"], eps=snap.hyper["eps"])
    zeroed = A.transform_snapshot(snap, "FRESH_ADAM")
    A.restore_adam(o2, zeroed)
    w2 = one_step(b2, o2)

    for a, b in zip(w1, w2):
        assert torch.allclose(a, b, atol=0, rtol=0)


def test_branches_start_from_bitwise_identical_weights():
    model, opt, _ = _tiny_setup()
    snap = A.snapshot_adam(opt)
    ref = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    for name in A.INTERVENTIONS:
        branch = copy.deepcopy(model)
        A.build_branch_optimizer(list(branch.parameters()), snap, name)
        got = torch.cat([p.detach().reshape(-1) for p in branch.parameters()])
        assert torch.equal(ref, got)


def test_implied_update_matches_real_adam_step():
    """The non-mutating update predictor reproduces a real Adam step exactly."""
    model, opt, x = _tiny_setup()
    snap = A.snapshot_adam(opt)

    loss = model(x).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    g = A.flat_grads(opt)
    m = A.flat_state(snap, "exp_avg")
    v = A.flat_state(snap, "exp_avg_sq")
    step = torch.tensor(A.steps_of(snap)[0])
    lr, (b1, b2), eps = snap.hyper["lr"], snap.hyper["betas"], snap.hyper["eps"]
    predicted = A.implied_adam_update(m, v, step, g, lr, b1, b2, eps)

    before = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
    opt.step()
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    actual = after - before
    assert torch.allclose(predicted, actual, atol=1e-7, rtol=1e-5)


def test_reset_step_only_changes_bias_correction_not_moments():
    model, opt, x = _tiny_setup()
    snap = A.snapshot_adam(opt)
    out = A.transform_snapshot(snap, "RESET_STEP_ONLY")
    assert torch.equal(A.flat_state(snap, "exp_avg"), A.flat_state(out, "exp_avg"))
    assert torch.equal(A.flat_state(snap, "exp_avg_sq"), A.flat_state(out, "exp_avg_sq"))
    assert A.steps_of(out) == [0.0] * len(A.steps_of(snap))


def test_state_summary_fields():
    _, opt, _ = _tiny_setup()
    s = A.state_summary(A.snapshot_adam(opt))
    for k in ("m_norm", "v_norm", "sqrt_v_mean", "sqrt_v_median", "step_min", "step_max"):
        assert k in s
    assert s["step_min"] == s["step_max"] == 7.0


def test_cosine_edge_cases():
    a = torch.zeros(4)
    b = torch.ones(4)
    assert A.cosine(a, b) != A.cosine(a, b) or True  # nan-safe
    import math
    assert math.isnan(A.cosine(a, b))
    assert abs(A.cosine(b, b) - 1.0) < 1e-6
    assert abs(A.cosine(b, -b) + 1.0) < 1e-6


def test_restored_carry_all_branch_steps_identically_to_the_continuing_optimizer():
    """A CARRY_ALL clone must be numerically indistinguishable from continuing."""
    model, opt, x = _tiny_setup()
    snap = A.snapshot_adam(opt)

    master = model
    master_before = [p.detach().clone() for p in master.parameters()]
    loss = master(x).pow(2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    master_after = [p.detach().clone() for p in master.parameters()]

    branch = copy.deepcopy(model)
    for p_b, p0 in zip(branch.parameters(), master_before):
        p_b.data.copy_(p0)
    bopt = A.build_branch_optimizer(list(branch.parameters()), snap, "CARRY_ALL")
    loss = branch(x).pow(2).mean()
    bopt.zero_grad(set_to_none=True); loss.backward(); bopt.step()

    for a, b in zip(master_after, branch.parameters()):
        assert torch.equal(a, b.detach()), "CARRY_ALL clone diverged from the master"


def test_step_counter_keeps_its_original_device_through_a_roundtrip():
    model, opt, _ = _tiny_setup()
    p0 = next(model.parameters())
    dev = opt.state[p0]["step"].device
    snap = A.snapshot_adam(opt)
    assert snap.entries[0]["step"].device == dev
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-2)
    A.restore_adam(opt2, snap)
    assert opt2.state[p0]["step"].device == dev
