"""Data-loading equivalence with RobustBench and BatchNorm-state confound checks."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optstate import adam_state as A                          # noqa: E402
from optstate import model as M                               # noqa: E402
from optstate import tent_core as T                           # noqa: E402
from optstate.data import (CONVENTIONAL_ORDER, Cifar10CStore,  # noqa: E402
                           DomainSpec, DomainStream, corruption_orders)

DATA = ROOT / "data"
has_data = (DATA / "CIFAR-10-C").exists()
needs_data = pytest.mark.skipif(not has_data, reason="CIFAR-10-C not downloaded")


@needs_data
def test_stream_matches_robustbench_loader_exactly():
    """Seed 0 (unshuffled) must equal robustbench.load_cifar10c byte for byte."""
    from robustbench.data import load_cifar10c
    x_ref, y_ref = load_cifar10c(400, 5, str(DATA), False, ["gaussian_noise"])
    store = Cifar10CStore(DATA, 5)
    stream = DomainStream(store, DomainSpec("gaussian_noise", 5, 0), 200,
                          torch.device("cpu"))
    xs, ys = [], []
    for _, x, y in stream.batches(0, 2):
        xs.append(x); ys.append(y)
    x = torch.cat(xs); y = torch.cat(ys)
    assert x.shape == x_ref.shape and y.shape == y_ref.shape
    assert torch.equal(x, x_ref)
    assert torch.equal(y, y_ref)


@needs_data
def test_severity_slice_selects_the_right_block():
    store = Cifar10CStore(DATA, 5)
    raw = np.load(DATA / "CIFAR-10-C/fog.npy", mmap_mode="r")
    assert np.array_equal(store.images("fog", 5), np.asarray(raw[40000:50000]))
    assert np.array_equal(store.images("fog", 1), np.asarray(raw[0:10000]))


@needs_data
def test_permutations_are_bijections_and_seed_stable():
    store = Cifar10CStore(DATA, 5)
    for seed in (0, 1, 2):
        s = DomainStream(store, DomainSpec("snow", 5, seed), 200, torch.device("cpu"))
        assert sorted(s.order.tolist()) == list(range(10000))
        s2 = DomainStream(store, DomainSpec("snow", 5, seed), 200, torch.device("cpu"))
        assert np.array_equal(s.order, s2.order)


def test_corruption_orders_are_permutations_of_the_conventional_set():
    orders = corruption_orders(3)
    assert orders[0]["name"] == "conventional"
    assert tuple(orders[0]["order"]) == CONVENTIONAL_ORDER
    assert len(orders) == 4
    seen = set()
    for o in orders[1:]:
        assert sorted(o["order"]) == sorted(CONVENTIONAL_ORDER)
        assert o["perm_seed"] is not None
        seen.add(tuple(o["order"]))
    assert len(seen) == 3, "the three permutations must differ"
    # regenerating with the same seeds must reproduce them
    assert [o["order"] for o in corruption_orders(3)] == [o["order"] for o in orders]


def _bn_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))


def test_interventions_do_not_alter_batchnorm_configuration():
    """Section 19: optimizer-state changes must not touch normalisation state."""
    net = M.configure_tent_model(_bn_net())
    params, _ = M.collect_bn_params(net)
    opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
    x = torch.randn(16, 3, 8, 8); y = torch.randint(0, 4, (16,))
    for i in range(4):
        T.tent_step(net, opt, x, y, i)
    snap = A.snapshot_adam(opt)

    def bn_config(m):
        return [(mod.training, mod.track_running_stats,
                 mod.running_mean is None, mod.running_var is None,
                 float(mod.eps), mod.momentum)
                for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]

    ref_cfg = bn_config(net)
    ref_keys = sorted(net.state_dict().keys())
    for name in A.INTERVENTIONS:
        branch = copy.deepcopy(net)
        bp, _ = M.collect_bn_params(branch)
        A.build_branch_optimizer(bp, snap, name)
        assert bn_config(branch) == ref_cfg, name
        assert sorted(branch.state_dict().keys()) == ref_keys, name
        assert branch.training is net.training
        # Tent discards the running statistics themselves; the vestigial
        # num_batches_tracked buffer survives but must be identical everywhere.
        assert not any(k.endswith(("running_mean", "running_var"))
                       for k in branch.state_dict()), name
        for k, v in branch.state_dict().items():
            assert torch.equal(v, net.state_dict()[k]), f"{name}: {k} changed"


def test_all_branches_predict_identically_on_the_first_boundary_batch():
    """Structural consequence of prediction-before-update + matched weights."""
    net = M.configure_tent_model(_bn_net())
    params, _ = M.collect_bn_params(net)
    opt = M.make_adam(params, 1e-2, 0.9, 0.999, 0.0)
    x = torch.randn(16, 3, 8, 8); y = torch.randint(0, 4, (16,))
    for i in range(4):
        T.tent_step(net, opt, x, y, i)
    snap = A.snapshot_adam(opt)
    xb, yb = torch.randn(16, 3, 8, 8), torch.randint(0, 4, (16,))
    first = set()
    for name in A.INTERVENTIONS:
        branch = copy.deepcopy(net)
        bp, _ = M.collect_bn_params(branch)
        bopt = A.build_branch_optimizer(bp, snap, name)
        first.add(T.tent_step(branch, bopt, xb, yb, 0).n_correct)
    assert len(first) == 1
