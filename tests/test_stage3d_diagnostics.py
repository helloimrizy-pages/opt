from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from expert_analysis.protection_optimization import (
    BASE_BITS_BY_REGIME,
    PROTECTED_BITS,
    build_expert_memory_matrix,
)
from expert_analysis.specialist_preservation import (
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
)
from expert_analysis.stage3d_diagnostics import (
    DROP,
    FLAT,
    HEADROOM,
    INCONCLUSIVE,
    PRIMARY_REGIME,
    SECONDARY_REGIME,
    SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME,
    SWEEP_A_RANDOM_SEEDS,
    SWEEP_B_BITS,
    SWEEP_B_LAYERS,
    SWEEP_C_EXPERT_BITS,
    SWEEP_C_ROUTER_BF16_REFERENCE_RUN_ID,
    ReversibleRouterQuantization,
    append_run_record,
    budget_protected_count,
    completed_run_ids,
    read_run_records,
    routing_ordered_protection_set,
    run_config_fingerprint,
    shared_random_protection_sets,
    summarize_run_losses,
    sweep_a_decision,
    sweep_a_protection_sets,
    sweep_b_decision,
    sweep_b_protection_sets,
    sweep_c_protection_sets,
    sweep_c_report,
    sweep_jsonl_path,
)

# OLMoE's per-expert weight shapes, the same ones the frozen memory matrix holds.
EXPERT_SHAPES = [(2048, 1024), (2048, 2048)]


@pytest.fixture(scope="module")
def memory():
    return build_expert_memory_matrix([EXPERT_SHAPES] * NUM_MOE_LAYERS)


@pytest.fixture(scope="module")
def routing_counts():
    rng = np.random.default_rng(7)
    return rng.integers(10, 6000, size=(NUM_MOE_LAYERS, NUM_EXPERTS)).astype(np.float64)


def test_both_regimes_protect_the_same_expert_count(memory):
    counts = {
        regime: budget_protected_count(memory, regime)
        for regime in (PRIMARY_REGIME, SECONDARY_REGIME)
    }
    assert counts[PRIMARY_REGIME] == counts[SECONDARY_REGIME] == 204


def test_random_sets_are_shared_across_regimes_and_fit_both_budgets(memory):
    sets = shared_random_protection_sets(memory)
    assert set(sets) == set(SWEEP_A_RANDOM_SEEDS)
    for regime in (PRIMARY_REGIME, SECONDARY_REGIME):
        base_bits = BASE_BITS_BY_REGIME[regime]
        delta = memory.delta_protection_bytes(base_bits)
        budget = memory.protection_budget_bytes(base_bits, 0.20)
        for protected in sets.values():
            assert int(protected.sum()) == 204
            assert int((delta * protected).sum()) <= budget


def test_sweep_a_arms_use_identical_random_sets(memory, routing_counts):
    by_id = {item.run_id: item for item in sweep_a_protection_sets(memory, routing_counts)}
    for seed in SWEEP_A_RANDOM_SEEDS[
        : SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME[SECONDARY_REGIME]
    ]:
        primary = by_id[f"a_{PRIMARY_REGIME}_random_seed{seed}"]
        secondary = by_id[f"a_{SECONDARY_REGIME}_random_seed{seed}"]
        assert primary.protection_sha256 == secondary.protection_sha256
        assert primary.bits_sha256 != secondary.bits_sha256


def test_sweep_a_produces_thirty_six_runs(memory, routing_counts):
    sets = sweep_a_protection_sets(memory, routing_counts)
    assert len(sets) == 36
    assert sum(1 for item in sets if item.regime == PRIMARY_REGIME) == 23
    assert sum(1 for item in sets if item.regime == SECONDARY_REGIME) == 13
    for item in sets:
        assert item.bits.shape == (NUM_MOE_LAYERS, NUM_EXPERTS)
        widths = set(np.unique(item.bits).tolist())
        if item.run_id.endswith("no_protection"):
            assert widths == {BASE_BITS_BY_REGIME[item.regime]}
            assert item.protected_expert_count == 0
        else:
            assert widths == {BASE_BITS_BY_REGIME[item.regime], PROTECTED_BITS}
            assert item.protected_expert_count == 204


def test_routing_ordered_sets_are_disjoint_and_correctly_ordered(routing_counts):
    most = routing_ordered_protection_set(routing_counts, 204, most_routed=True)
    least = routing_ordered_protection_set(routing_counts, 204, most_routed=False)
    assert int(most.sum()) == int(least.sum()) == 204
    assert int((most & least).sum()) == 0
    assert routing_counts[most == 1].min() >= routing_counts[least == 1].max()


def test_routing_order_breaks_ties_deterministically():
    counts = np.ones((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.float64)
    most = routing_ordered_protection_set(counts, 3, most_routed=True)
    least = routing_ordered_protection_set(counts, 3, most_routed=False)
    assert np.array_equal(most, least)
    assert [list(pair) for pair in zip(*np.nonzero(most))] == [[0, 0], [0, 1], [0, 2]]


def test_sweep_b_touches_one_layer_at_a_time():
    sets = sweep_b_protection_sets()
    assert len(sets) == len(SWEEP_B_LAYERS) == NUM_MOE_LAYERS
    for layer, item in zip(SWEEP_B_LAYERS, sets, strict=True):
        assert np.all(item.bits[layer, :] == SWEEP_B_BITS)
        untouched = np.delete(item.bits, layer, axis=0)
        assert np.all(untouched == 16)


def test_sweep_c_is_one_run_reusing_sweep_a_as_its_reference():
    sets = sweep_c_protection_sets()
    assert len(sets) == 1
    assert sets[0].router_bits == SWEEP_C_EXPERT_BITS
    assert np.all(sets[0].bits == SWEEP_C_EXPERT_BITS)
    assert SWEEP_C_ROUTER_BF16_REFERENCE_RUN_ID == f"a_{PRIMARY_REGIME}_no_protection"


def test_worst_domain_uses_relative_increase_not_raw_loss():
    baseline = {"general": 3.0, "math": 2.0, "coding": 1.0, "reasoning": 1.5}
    # coding is the largest relative jump; general stays the largest raw loss.
    losses = {"general": 3.03, "math": 2.02, "coding": 1.10, "reasoning": 1.52}
    summary = summarize_run_losses(losses, baseline)
    assert summary["worst_domain_relative_domain"] == "coding"
    assert summary["worst_domain_raw_domain"] == "general"
    assert summary["worst_domain_relative"] == pytest.approx(0.10)
    assert summary["worst_domain_raw"] == pytest.approx(3.03)


def _sweep_a_records(
    regime: str, random_values, most: float, least: float, none: float = 0.05
):
    baseline = {domain: 2.0 for domain in STAGE2B_DOMAINS}

    def record(run_id: str, worst: float, seed=None):
        losses = dict(baseline)
        losses["coding"] = baseline["coding"] * (1.0 + worst)
        summary = summarize_run_losses(losses, baseline)
        return {
            "run_id": run_id,
            "sweep": "a",
            "regime": regime,
            "seed": seed,
            "protected_expert_count": 204,
            "bits_matrix_sha256": f"{run_id}-bits",
            **summary,
        }

    seeds = SWEEP_A_RANDOM_SEEDS[: len(random_values)]
    records = [
        record(f"a_{regime}_random_seed{seed}", value, seed)
        for seed, value in zip(seeds, random_values, strict=True)
    ]
    records.append(record(f"a_{regime}_most_routed", most))
    records.append(record(f"a_{regime}_least_routed", least))
    records.append(record(f"a_{regime}_no_protection", none))
    return records


def test_sweep_a_flat_when_gap_is_small():
    rng = np.random.default_rng(1)
    randoms = 0.02 + rng.normal(0, 0.001, size=20)
    records = _sweep_a_records(PRIMARY_REGIME, randoms, most=0.0200, least=0.0201)
    decision = sweep_a_decision(records)
    assert decision["outcome"] == FLAT
    assert decision["arms"][PRIMARY_REGIME]["gap_over_sd_random"] < 2.0


def test_sweep_a_headroom_when_gap_is_large():
    rng = np.random.default_rng(2)
    randoms = 0.02 + rng.normal(0, 0.0005, size=20)
    records = _sweep_a_records(PRIMARY_REGIME, randoms, most=0.015, least=0.035)
    decision = sweep_a_decision(records)
    assert decision["outcome"] == HEADROOM
    assert decision["arms"][PRIMARY_REGIME]["gap_over_sd_random"] > 4.0


def test_sweep_a_inconclusive_between_the_multiples():
    randoms = list(np.linspace(0.019, 0.021, 20))
    standard_deviation = float(np.std(randoms, ddof=1))
    records = _sweep_a_records(
        PRIMARY_REGIME, randoms, most=0.020, least=0.020 + 3.0 * standard_deviation
    )
    decision = sweep_a_decision(records)
    assert decision["outcome"] == INCONCLUSIVE


def test_secondary_arm_escalates_a_flat_primary_to_inconclusive():
    rng = np.random.default_rng(3)
    primary = _sweep_a_records(
        PRIMARY_REGIME,
        0.02 + rng.normal(0, 0.001, size=20),
        most=0.0200,
        least=0.0201,
    )
    secondary = _sweep_a_records(
        SECONDARY_REGIME,
        0.05 + rng.normal(0, 0.0005, size=10),
        most=0.040,
        least=0.070,
    )
    decision = sweep_a_decision(primary + secondary)
    assert decision["arms"][PRIMARY_REGIME]["outcome"] == FLAT
    assert decision["arms"][SECONDARY_REGIME]["outcome"] == HEADROOM
    assert decision["outcome"] == INCONCLUSIVE
    assert "full 20 random sets" in decision["note"]


def test_secondary_arm_cannot_produce_headroom_on_its_own():
    rng = np.random.default_rng(4)
    primary = _sweep_a_records(
        PRIMARY_REGIME,
        0.02 + rng.normal(0, 0.001, size=20),
        most=0.0200,
        least=0.0195,
    )
    secondary = _sweep_a_records(
        SECONDARY_REGIME,
        0.05 + rng.normal(0, 0.0005, size=10),
        most=0.040,
        least=0.070,
    )
    assert sweep_a_decision(primary + secondary)["outcome"] != HEADROOM


def test_sweep_a_flags_a_negative_gap_but_still_calls_it_flat():
    rng = np.random.default_rng(5)
    records = _sweep_a_records(
        PRIMARY_REGIME, 0.02 + rng.normal(0, 0.001, size=20), most=0.030, least=0.010
    )
    decision = sweep_a_decision(records)
    assert decision["arms"][PRIMARY_REGIME]["gap_is_negative"] is True
    assert decision["outcome"] == FLAT


def test_sweep_a_refuses_to_decide_without_the_primary_arm():
    rng = np.random.default_rng(6)
    secondary = _sweep_a_records(
        SECONDARY_REGIME, 0.05 + rng.normal(0, 0.001, size=10), most=0.05, least=0.05
    )
    with pytest.raises(RuntimeError, match="4to8 arm"):
        sweep_a_decision(secondary)


def _sweep_b_records(worst_by_layer):
    baseline = {domain: 2.0 for domain in STAGE2B_DOMAINS}
    records = []
    for layer, worst in zip(SWEEP_B_LAYERS, worst_by_layer, strict=True):
        losses = dict(baseline)
        losses["coding"] = baseline["coding"] * (1.0 + worst)
        records.append(
            {
                "run_id": f"b_layer{layer:02d}",
                "sweep": "b",
                **summarize_run_losses(losses, baseline),
            }
        )
    return records


def test_sweep_b_headroom_drop_and_inconclusive():
    assert sweep_b_decision(
        _sweep_b_records([0.001 * (i + 1) for i in range(16)])
    )["outcome"] == HEADROOM
    assert sweep_b_decision(
        _sweep_b_records([0.001 + 0.00001 * i for i in range(16)])
    )["outcome"] == DROP
    assert sweep_b_decision(
        _sweep_b_records([0.001 * (1.0 + 0.05 * i) for i in range(16)])
    )["outcome"] == INCONCLUSIVE


def test_sweep_b_inconclusive_when_the_smallest_increase_is_not_positive():
    values = [0.002] * 16
    values[5] = -0.0001
    decision = sweep_b_decision(_sweep_b_records(values))
    assert decision["outcome"] == INCONCLUSIVE
    assert decision["ratio"] is None
    assert decision["smallest_increase_layer"] == 5


def test_sweep_c_requires_a_matching_expert_assignment():
    baseline = {domain: 2.0 for domain in STAGE2B_DOMAINS}
    quantized_losses = dict(baseline)
    quantized_losses["math"] = 2.06
    reference_losses = dict(baseline)
    reference_losses["math"] = 2.04
    quantized = {
        "run_id": f"c_uniform{SWEEP_C_EXPERT_BITS}_routers_quantized",
        "router_bits": 4,
        "bits_matrix_sha256": "same",
        "router_memory": {"router_parameters": 2097152},
        **summarize_run_losses(quantized_losses, baseline),
    }
    reference = {
        "run_id": SWEEP_C_ROUTER_BF16_REFERENCE_RUN_ID,
        "bits_matrix_sha256": "same",
        **summarize_run_losses(reference_losses, baseline),
    }
    report = sweep_c_report([quantized], [reference])
    assert report["worst_domain_relative_difference"] == pytest.approx(0.01)

    reference["bits_matrix_sha256"] = "different"
    with pytest.raises(RuntimeError, match="same expert bit assignment"):
        sweep_c_report([quantized], [reference])


def test_jsonl_append_and_resume(tmp_path: Path):
    path = sweep_jsonl_path(tmp_path, "a")
    append_run_record(path, {"run_id": "a_one", "config_sha256": "cfg"})
    append_run_record(path, {"run_id": "a_two", "config_sha256": "cfg"})
    append_run_record(path, {"run_id": "a_three", "config_sha256": "other"})
    assert len(read_run_records(path)) == 3
    assert completed_run_ids(path, "cfg") == {"a_one", "a_two"}
    assert completed_run_ids(path, "missing") == set()


def test_a_truncated_jsonl_line_is_an_error_not_a_silent_drop(tmp_path: Path):
    path = sweep_jsonl_path(tmp_path, "b")
    append_run_record(path, {"run_id": "b_layer00", "config_sha256": "cfg"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "b_lay')
    with pytest.raises(RuntimeError, match="interrupted mid-write"):
        read_run_records(path)


def test_config_fingerprint_changes_with_the_evaluation_set():
    common = ("model", "rev", "bfloat16", 1, 128)
    determinism = {"use_deterministic_algorithms": True, "torch_version": "2.8.0"}
    first = run_config_fingerprint(*common, {"general": "aaa"}, determinism)
    second = run_config_fingerprint(*common, {"general": "bbb"}, determinism)
    assert first != second
    # The torch version is deliberately excluded so a patch bump does not
    # invalidate a resumable run.
    third = run_config_fingerprint(
        *common,
        {"general": "aaa"},
        {"use_deterministic_algorithms": True, "torch_version": "2.9.0"},
    )
    assert first == third


class _Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 256, dtype=torch.float32))


class _Spec:
    def __init__(self, router: nn.Module, name: str) -> None:
        self.router = router
        self.router_name = name


def test_router_quantization_changes_weights_then_restores_them_bitwise():
    torch.manual_seed(0)
    model = nn.ModuleList([_Router() for _ in range(3)])
    specs = [_Spec(router, f"layer{index}.gate") for index, router in enumerate(model)]
    originals = [router.weight.detach().clone() for router in model]

    context = ReversibleRouterQuantization(specs, model, bits=4, group_size=128)
    with context:
        for router, original in zip(model, originals, strict=True):
            assert not torch.equal(router.weight, original)
    assert context.restoration_verified
    for router, original in zip(model, originals, strict=True):
        assert torch.equal(router.weight, original)
    assert len(context.diagnostics()["tensors"]) == 3


def test_router_quantization_restores_even_when_the_body_raises():
    torch.manual_seed(1)
    model = nn.ModuleList([_Router()])
    specs = [_Spec(model[0], "layer0.gate")]
    original = model[0].weight.detach().clone()
    with pytest.raises(ValueError):
        with ReversibleRouterQuantization(specs, model, bits=4):
            raise ValueError("evaluation failed")
    assert torch.equal(model[0].weight, original)


def test_router_discovery_rejects_a_multi_matrix_router():
    class _Wide(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.zeros(4, 8))
            self.b = nn.Parameter(torch.zeros(4, 8))

    from expert_analysis.stage3d_diagnostics import router_weight_references

    with pytest.raises(RuntimeError, match="exactly one"):
        router_weight_references([_Spec(_Wide(), "layer0.gate")])
