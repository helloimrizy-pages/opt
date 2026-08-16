"""Stage 3D diagnostics: does the mixed-precision objective have usable structure?

Stages 2A, 2B, and 2C each predicted which experts to keep at 8 bits under a
20% budget, and none beat the baseline. Two explanations remain. Either the
predictors were bad and per-expert sensitivity really is heterogeneous, or
per-expert sensitivity is near-uniform in this model, so the objective is flat
and no selection rule can win. Sweep A separates them at expert granularity,
Sweep B asks the same question at layer granularity, and Sweep C asks whether
the routers carry signal the expert weights do not.

This module measures. It selects nothing, fits nothing, and tunes nothing. The
random protection sets come from seeded permutations. The two deliberate sets
come from routing counts Stage 2B already froze. Every threshold in
``DECISION_THRESHOLDS`` was fixed here before the first evaluation ran, and
``prereg/stage3d.md`` records them verbatim.

Nothing here reads a Stage 2A, 2B, or 2C outcome. The frozen negative results
of those stages are untouched.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .balanced import array_sha256, canonical_sha256
from .controlled import PreparedDomainExamples
from .fragility import load_frozen_stage2b_scores
from .fragility_evaluation import load_stage2c_development_split
from .heldout_splits import load_heldout_split
from .io_utils import read_json
from .masking import LossStatistics
from .modeling import MoeLayerSpec
from .protection_optimization import (
    BASE_BITS_BY_REGIME,
    GROUP_SIZE,
    MEMORY_BIT_WIDTHS,
    PROTECTED_BITS,
    ExpertMemoryMatrix,
    bits_matrix_for_allocation,
    build_expert_memory_matrix,
    random_allocation,
    uniform_bits_matrix,
)
from .quantization import module_hook_count, symmetric_groupwise_qdq
from .specialist_preservation import (
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
    build_importance_tensor,
)

STAGE3D_STAGE = "stage3d_selection_headroom_diagnostics"
STAGE3D_RESULTS_DIRNAME = "stage3d_diagnostics"
EVALUATION_SET_SCHEMA = "stage3d_evaluation_set_v1"
RUN_RECORD_SCHEMA = "stage3d_run_v1"

# Evaluation set: the Stage 2B seed-43 and Stage 2C seed-45 development splits
# concatenated, seed-43 rows first. Both are frozen, both are disjoint from each
# other, from the seed-44 final reserve, and from the frozen controlled
# 100/domain set the calibration routing counts are computed on. A fresh split
# is not buildable: the mbpp test pool has only 32 unused eligible examples
# left, against the 50/domain the prior stages used.
EVALUATION_SPLIT_SEEDS = (43, 45)
EVALUATION_EXAMPLES_PER_DOMAIN = 100

# ---------------------------------------------------------------------------
# Sweep definitions. Frozen before the first evaluation.
# ---------------------------------------------------------------------------

SWEEP_A_BUDGET_FRACTION = 0.20
# The primary arm carries the Sweep A decision. The secondary arm is a check on
# whether the picture looks qualitatively different where damage is larger; it
# can escalate the outcome to inconclusive but can never on its own authorize
# the full 1024-expert leave-one-in sweep.
PRIMARY_REGIME = "4to8"
SECONDARY_REGIME = "3to8"
SWEEP_A_REGIMES = (PRIMARY_REGIME, SECONDARY_REGIME)
SWEEP_A_RANDOM_SEEDS = tuple(range(46, 66))
SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME = {PRIMARY_REGIME: 20, SECONDARY_REGIME: 10}
SWEEP_A_DELIBERATE_SETS = ("most_routed", "least_routed", "no_protection")

SWEEP_B_BITS = 4
SWEEP_B_LAYERS = tuple(range(NUM_MOE_LAYERS))

SWEEP_C_EXPERT_BITS = 4
SWEEP_C_ROUTER_BITS = 4

# ---------------------------------------------------------------------------
# Step 5 decision thresholds. Fixed here before any result existed.
# ---------------------------------------------------------------------------

# Sample standard deviation, ddof=1, of worst-domain relative increase across
# the random protection sets of one regime.
SWEEP_A_STANDARD_DEVIATION_DDOF = 1
SWEEP_A_HEADROOM_MULTIPLE = 4.0
SWEEP_A_FLAT_MULTIPLE = 2.0
SWEEP_B_HEADROOM_RATIO = 3.0
SWEEP_B_DROP_RATIO = 1.2

DECISION_THRESHOLDS = {
    "sweep_a": {
        "statistic": (
            "gap = worst-domain relative increase of the least-routed set minus "
            "that of the most-routed set; sd_random = sample standard deviation "
            "(ddof=1) of worst-domain relative increase across the random sets"
        ),
        "deciding_arm": PRIMARY_REGIME,
        "headroom": f"gap > {SWEEP_A_HEADROOM_MULTIPLE} * sd_random",
        "flat": f"gap < {SWEEP_A_FLAT_MULTIPLE} * sd_random",
        "inconclusive": (
            f"gap between {SWEEP_A_FLAT_MULTIPLE} and "
            f"{SWEEP_A_HEADROOM_MULTIPLE} times sd_random"
        ),
        "secondary_arm_rule": (
            f"the {SECONDARY_REGIME} arm can escalate a FLAT primary outcome to "
            "INCONCLUSIVE when it returns HEADROOM, which then requires running "
            "its full 20 random sets before any conclusion; it can never on its "
            "own authorize the 1024-expert leave-one-in sweep"
        ),
    },
    "sweep_b": {
        "statistic": (
            "ratio = largest per-layer worst-domain relative increase divided by "
            "the smallest"
        ),
        "headroom": f"ratio >= {SWEEP_B_HEADROOM_RATIO}",
        "drop": f"ratio <= {SWEEP_B_DROP_RATIO}",
        "inconclusive": (
            f"ratio strictly between {SWEEP_B_DROP_RATIO} and "
            f"{SWEEP_B_HEADROOM_RATIO}, or the smallest per-layer increase is "
            "not strictly positive, which leaves the ratio undefined"
        ),
    },
    "sweep_c": {
        "statistic": (
            "worst-domain relative increase with routers quantized minus the "
            "same quantity with routers at BF16, alongside the router parameter "
            "count and its share of deployed memory"
        ),
        "rule": "no threshold; this is a fact to record",
    },
}

HEADROOM = "HEADROOM"
FLAT = "FLAT"
DROP = "DROP"
INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Evaluation set
# ---------------------------------------------------------------------------


def pooled_evaluation_examples(
    stage2b_results_dir: Path, stage2c_results_dir: Path, domain: str
) -> PreparedDomainExamples:
    """Concatenate the frozen seed-43 and seed-45 rows for one domain.

    Both source splits verify themselves against their own manifests as they
    load. Row order is seed-43 first, then seed-45, and never varies.
    """

    seed43 = load_heldout_split(stage2b_results_dir / "splits", "development", domain)
    seed45 = load_stage2c_development_split(stage2c_results_dir / "splits", domain)
    if seed43.sequence_length != seed45.sequence_length:
        raise RuntimeError(
            f"Split geometry differs for {domain}: {seed43.sequence_length} vs "
            f"{seed45.sequence_length} tokens"
        )
    pooled = PreparedDomainExamples(
        domain=domain,
        input_ids=np.concatenate([seed43.input_ids, seed45.input_ids], axis=0),
        attention_mask=np.concatenate(
            [seed43.attention_mask, seed45.attention_mask], axis=0
        ),
        measurement_mask=np.concatenate(
            [seed43.measurement_mask, seed45.measurement_mask], axis=0
        ),
        metadata={
            "domain": domain,
            "pooled_from": [
                {"seed": 43, "source": "stage2b_development", "rows": seed43.num_examples},
                {"seed": 45, "source": "stage2c_development", "rows": seed45.num_examples},
            ],
            "row_order": "seed43 rows first, then seed45 rows",
        },
    )
    pooled.validate()
    if pooled.num_examples != EVALUATION_EXAMPLES_PER_DOMAIN:
        raise RuntimeError(
            f"Pooled evaluation set for {domain} has {pooled.num_examples} rows, "
            f"expected {EVALUATION_EXAMPLES_PER_DOMAIN}"
        )
    duplicates = pooled.num_examples - len(
        {array_sha256(np.ascontiguousarray(row)) for row in pooled.input_ids}
    )
    if duplicates:
        raise RuntimeError(
            f"Pooled evaluation set for {domain} repeats {duplicates} rows; the "
            "two source splits are supposed to be disjoint"
        )
    return pooled


def load_evaluation_set(
    stage2b_results_dir: Path, stage2c_results_dir: Path
) -> dict[str, PreparedDomainExamples]:
    return {
        domain: pooled_evaluation_examples(
            stage2b_results_dir, stage2c_results_dir, domain
        )
        for domain in STAGE2B_DOMAINS
    }


def evaluation_set_manifest(
    examples_by_domain: Mapping[str, PreparedDomainExamples],
    stage2b_results_dir: Path,
    stage2c_results_dir: Path,
) -> dict[str, Any]:
    """Freeze the pooled evaluation set's identity, and prove seed 44 is absent."""

    stage2b_manifest = read_json(
        stage2b_results_dir / "splits" / "split_manifest.json"
    )
    stage2c_manifest = read_json(
        stage2c_results_dir / "splits" / "split_manifest.json"
    )
    domains: dict[str, Any] = {}
    for domain in STAGE2B_DOMAINS:
        examples = examples_by_domain[domain]
        seed43_texts = set(
            stage2b_manifest["domains"][domain]["development"]["example_text_sha256"]
        )
        seed44_texts = set(
            stage2b_manifest["domains"][domain]["final"]["example_text_sha256"]
        )
        seed45_texts = set(stage2c_manifest["domains"][domain]["example_text_sha256"])
        if seed43_texts & seed45_texts:
            raise RuntimeError(
                f"The seed-43 and seed-45 splits overlap for {domain}; refusing "
                "to pool them"
            )
        contaminating = (seed43_texts | seed45_texts) & seed44_texts
        if contaminating:
            raise RuntimeError(
                f"The pooled evaluation set for {domain} would contain "
                f"{len(contaminating)} seed-44 final texts"
            )
        domains[domain] = {
            "num_examples": examples.num_examples,
            "sequence_length": examples.sequence_length,
            "measured_tokens_per_example": int(examples.measurement_mask.sum(axis=1)[0]),
            "input_ids_sha256": array_sha256(examples.input_ids),
            "measurement_mask_sha256": array_sha256(examples.measurement_mask),
            "seed43_rows": len(seed43_texts),
            "seed45_rows": len(seed45_texts),
            "seed44_rows": 0,
            "disjointness_verified": True,
        }
    return {
        "schema": EVALUATION_SET_SCHEMA,
        "stage": STAGE3D_STAGE,
        "composition": (
            "the frozen Stage 2B seed-43 development split concatenated with the "
            "frozen Stage 2C seed-45 development split, seed-43 rows first"
        ),
        "split_seeds": list(EVALUATION_SPLIT_SEEDS),
        "examples_per_domain": EVALUATION_EXAMPLES_PER_DOMAIN,
        "seed44_final_reserve_untouched": True,
        "seed46_unused": True,
        "why_not_a_fresh_split": (
            "the mbpp full test split has 500 rows, 332 of them long enough for "
            "the 68-token geometry, and 300 are already spent across the frozen "
            "controlled set and seeds 43, 44, and 45; only 32 eligible examples "
            "remain, so a balanced fresh split caps at 32 per domain"
        ),
        "contamination_note": (
            "seeds 43 and 45 were observed by Stage 2B and Stage 2C. Stage 3D "
            "selects nothing, fits nothing, and tunes nothing on them: its "
            "protection sets come from seeded permutations and from Stage 2B "
            "calibration routing counts measured on the disjoint frozen "
            "controlled set, so no Stage 3D quantity depends on any seed-43 or "
            "seed-45 outcome"
        ),
        "domains": domains,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluation_split_hashes(
    examples_by_domain: Mapping[str, PreparedDomainExamples]
) -> dict[str, str]:
    return {
        domain: array_sha256(examples.input_ids)
        for domain, examples in examples_by_domain.items()
    }


# ---------------------------------------------------------------------------
# Protection sets
# ---------------------------------------------------------------------------


def load_frozen_memory_matrix(stage2b_results_dir: Path) -> ExpertMemoryMatrix:
    """Reload the frozen per-expert byte accounting and recompute it as a check."""

    path = stage2b_results_dir / "calibration" / "memory_matrix.npz"
    with np.load(path, allow_pickle=False) as data:
        bytes_by_bits = {
            bits: np.asarray(data[f"bytes_bits{bits}"], dtype=np.int64)
            for bits in MEMORY_BIT_WIDTHS
        }
        weight_count = np.asarray(data["weight_count"], dtype=np.int64)
        group_count = np.asarray(data["group_count"], dtype=np.int64)
        group_size = int(data["group_size"][0])
        tensor_shapes = [tuple(int(v) for v in shape) for shape in data["tensor_shapes"]]
    if group_size != GROUP_SIZE:
        raise RuntimeError(f"Frozen memory matrix uses group size {group_size}")
    recomputed = build_expert_memory_matrix(
        [tensor_shapes] * NUM_MOE_LAYERS, group_size=group_size, num_experts=NUM_EXPERTS
    )
    for bits in MEMORY_BIT_WIDTHS:
        if not np.array_equal(recomputed.bytes_by_bits[bits], bytes_by_bits[bits]):
            raise RuntimeError(
                f"Frozen memory matrix disagrees with the recomputed {bits}-bit bytes"
            )
    return ExpertMemoryMatrix(
        bytes_by_bits=bytes_by_bits,
        weight_count=weight_count,
        group_count=group_count,
        tensor_shapes=[list(tensor_shapes) for _ in range(NUM_MOE_LAYERS)],
        group_size=group_size,
    )


def calibration_routing_counts(stage2b_results_dir: Path) -> np.ndarray:
    """Total routing counts per (layer, expert) over the frozen calibration set.

    Reads the raw counts Stage 2B froze and proves they still reproduce the
    layer-equalized ``routing`` array whose SHA-256 the Stage 2B metadata
    records, so the raw array is anchored to a hashed artifact.
    """

    load_frozen_stage2b_scores(stage2b_results_dir)
    path = stage2b_results_dir / "calibration" / "routing_specialization.npz"
    with np.load(path, allow_pickle=False) as data:
        routing_raw = np.asarray(data["routing_raw"], dtype=np.float64)
        routing = np.asarray(data["routing"], dtype=np.float64)
    expected_shape = (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS))
    if routing_raw.shape != expected_shape:
        raise RuntimeError("Frozen raw routing counts have the wrong shape")
    recomputed = build_importance_tensor(
        {
            domain: routing_raw[:, :, index]
            for index, domain in enumerate(STAGE2B_DOMAINS)
        }
    )
    if not np.allclose(recomputed, routing, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Frozen raw routing counts do not reproduce the frozen layer-"
            "equalized routing array"
        )
    totals = routing_raw.sum(axis=2)
    if np.any(totals < 0) or not np.all(np.isfinite(totals)):
        raise RuntimeError("Frozen routing counts contain invalid values")
    return totals


def routing_ordered_protection_set(
    routing_counts: np.ndarray, count: int, most_routed: bool
) -> np.ndarray:
    """Protect the ``count`` most or least frequently routed experts.

    Ties break on ``(layer, expert)`` ascending in both directions, so the
    selection is fully determined by the frozen counts.
    """

    counts = np.asarray(routing_counts, dtype=np.float64)
    if counts.shape != (NUM_MOE_LAYERS, NUM_EXPERTS):
        raise ValueError("Routing counts must have shape [layer, expert]")
    if not 0 < count <= counts.size:
        raise ValueError("The protected count must be within the expert grid")
    cells = [
        (float(counts[layer, expert]), layer, expert)
        for layer in range(NUM_MOE_LAYERS)
        for expert in range(NUM_EXPERTS)
    ]
    if most_routed:
        cells.sort(key=lambda item: (-item[0], item[1], item[2]))
    else:
        cells.sort(key=lambda item: (item[0], item[1], item[2]))
    protected = np.zeros((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.uint8)
    for _, layer, expert in cells[:count]:
        protected[layer, expert] = 1
    if int(protected.sum()) != count:
        raise RuntimeError("Routing-ordered selection produced the wrong count")
    return protected


def shared_random_protection_sets(
    memory: ExpertMemoryMatrix,
    seeds: Sequence[int] = SWEEP_A_RANDOM_SEEDS,
    fraction: float = SWEEP_A_BUDGET_FRACTION,
) -> dict[int, np.ndarray]:
    """One protection set per seed, reused unchanged by both precision regimes.

    Sets are drawn once under the primary regime's byte budget and then checked
    against the secondary regime's budget, so the two arms compare the same
    experts rather than two independent draws.
    """

    primary_bits = BASE_BITS_BY_REGIME[PRIMARY_REGIME]
    secondary_bits = BASE_BITS_BY_REGIME[SECONDARY_REGIME]
    primary_delta = memory.delta_protection_bytes(primary_bits)
    secondary_delta = memory.delta_protection_bytes(secondary_bits)
    primary_budget = memory.protection_budget_bytes(primary_bits, fraction)
    secondary_budget = memory.protection_budget_bytes(secondary_bits, fraction)
    sets: dict[int, np.ndarray] = {}
    for seed in seeds:
        protected = random_allocation(primary_delta, primary_budget, int(seed)).protected
        used_secondary = int((secondary_delta * protected).sum())
        if used_secondary > secondary_budget:
            raise RuntimeError(
                f"Random set for seed {seed} costs {used_secondary} bytes under "
                f"{SECONDARY_REGIME}, above its {secondary_budget}-byte budget; "
                "the two arms cannot share this set"
            )
        sets[int(seed)] = protected
    counts = {int(value.sum()) for value in sets.values()}
    if len(counts) != 1:
        raise RuntimeError(
            f"Random protection sets differ in size: {sorted(counts)}; the "
            "budget is supposed to fix the count"
        )
    return sets


def budget_protected_count(
    memory: ExpertMemoryMatrix,
    regime: str,
    fraction: float = SWEEP_A_BUDGET_FRACTION,
) -> int:
    """How many equal-sized experts the byte budget of one regime pays for."""

    base_bits = BASE_BITS_BY_REGIME[regime]
    delta = memory.delta_protection_bytes(base_bits)
    per_expert = int(delta.flat[0])
    if not np.all(delta == per_expert):
        raise RuntimeError(
            "Experts have unequal protection cost; a single protected count is "
            "not well defined"
        )
    return int(memory.protection_budget_bytes(base_bits, fraction) // per_expert)


@dataclass(frozen=True)
class ProtectionSet:
    """One named per-expert bit assignment, plus how it was chosen."""

    run_id: str
    sweep: str
    description: str
    regime: str | None
    base_bits: int | None
    seed: int | None
    selection_rule: str
    protected: np.ndarray | None
    bits: np.ndarray
    router_bits: int | None = None

    @property
    def protected_expert_count(self) -> int:
        return 0 if self.protected is None else int(self.protected.sum())

    @property
    def protected_experts(self) -> list[list[int]]:
        if self.protected is None:
            return []
        return [[int(l), int(e)] for l, e in zip(*np.nonzero(self.protected))]

    @property
    def protection_sha256(self) -> str:
        source = (
            np.zeros((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.uint8)
            if self.protected is None
            else self.protected
        )
        return array_sha256(source)

    @property
    def bits_sha256(self) -> str:
        return array_sha256(np.asarray(self.bits, dtype=np.int64))


def sweep_a_protection_sets(
    memory: ExpertMemoryMatrix, routing_counts: np.ndarray
) -> list[ProtectionSet]:
    """The 23 primary-arm and 13 secondary-arm configurations of Sweep A."""

    random_sets = shared_random_protection_sets(memory)
    output: list[ProtectionSet] = []
    for regime in SWEEP_A_REGIMES:
        base_bits = BASE_BITS_BY_REGIME[regime]
        count = budget_protected_count(memory, regime)
        seed_count = SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME[regime]
        for seed in SWEEP_A_RANDOM_SEEDS[:seed_count]:
            protected = random_sets[seed]
            output.append(
                ProtectionSet(
                    run_id=f"a_{regime}_random_seed{seed}",
                    sweep="a",
                    description=(
                        f"random protection set, seed {seed}, {regime} at the "
                        f"{SWEEP_A_BUDGET_FRACTION:.0%} Stage 2C budget"
                    ),
                    regime=regime,
                    base_bits=base_bits,
                    seed=int(seed),
                    selection_rule=(
                        "seeded random permutation filled to the byte budget; "
                        "reads no expert score; identical set in both regimes"
                    ),
                    protected=protected,
                    bits=bits_matrix_for_allocation(protected, base_bits),
                )
            )
        for name, most in (("most_routed", True), ("least_routed", False)):
            protected = routing_ordered_protection_set(routing_counts, count, most)
            end = "top" if most else "bottom"
            output.append(
                ProtectionSet(
                    run_id=f"a_{regime}_{name}",
                    sweep="a",
                    description=(
                        f"{'most' if most else 'least'} frequently routed "
                        f"{count} experts, {regime} at the "
                        f"{SWEEP_A_BUDGET_FRACTION:.0%} Stage 2C budget"
                    ),
                    regime=regime,
                    base_bits=base_bits,
                    seed=None,
                    selection_rule=(
                        f"{end} {count} by total routing count over the frozen "
                        "Stage 2B calibration set, ties broken on "
                        "(layer, expert) ascending"
                    ),
                    protected=protected,
                    bits=bits_matrix_for_allocation(protected, base_bits),
                )
            )
        output.append(
            ProtectionSet(
                run_id=f"a_{regime}_no_protection",
                sweep="a",
                description=f"every expert at {base_bits} bits, nothing protected",
                regime=regime,
                base_bits=base_bits,
                seed=None,
                selection_rule="no protection",
                protected=np.zeros((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.uint8),
                bits=uniform_bits_matrix(base_bits),
            )
        )
    return output


def sweep_b_protection_sets() -> list[ProtectionSet]:
    """One configuration per layer: that layer's 64 experts at 4 bits, rest BF16."""

    output: list[ProtectionSet] = []
    for layer in SWEEP_B_LAYERS:
        bits = uniform_bits_matrix(16)
        bits[layer, :] = SWEEP_B_BITS
        output.append(
            ProtectionSet(
                run_id=f"b_layer{layer:02d}",
                sweep="b",
                description=(
                    f"all {NUM_EXPERTS} experts of layer {layer} at "
                    f"{SWEEP_B_BITS} bits, every other parameter at BF16"
                ),
                regime=None,
                base_bits=SWEEP_B_BITS,
                seed=None,
                selection_rule=f"single layer {layer}",
                protected=None,
                bits=bits,
            )
        )
    return output


def sweep_c_protection_sets() -> list[ProtectionSet]:
    """The one new Sweep C run: uniform experts plus quantized routers.

    The pipeline never quantized routers, so the routers-at-BF16 comparison
    point is already produced by Sweep A's ``a_4to8_no_protection`` run, which
    is the identical expert configuration. Only the quantized-router state is
    new, and the preregistration labels it a diagnostic on baseline strength.
    """

    return [
        ProtectionSet(
            run_id=f"c_uniform{SWEEP_C_EXPERT_BITS}_routers_quantized",
            sweep="c",
            description=(
                f"every expert at {SWEEP_C_EXPERT_BITS} bits and every MoE "
                f"router weight at {SWEEP_C_ROUTER_BITS} bits"
            ),
            regime=None,
            base_bits=SWEEP_C_EXPERT_BITS,
            seed=None,
            selection_rule="uniform experts, routers additionally quantized",
            protected=np.zeros((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.uint8),
            bits=uniform_bits_matrix(SWEEP_C_EXPERT_BITS),
            router_bits=SWEEP_C_ROUTER_BITS,
        )
    ]


SWEEP_C_ROUTER_BF16_REFERENCE_RUN_ID = f"a_{PRIMARY_REGIME}_no_protection"


# ---------------------------------------------------------------------------
# Router quantization
# ---------------------------------------------------------------------------


def router_weight_references(
    layer_specs: Sequence[MoeLayerSpec],
) -> list[tuple[str, nn.Parameter]]:
    """Every matrix-shaped router parameter, one entry per MoE layer.

    OLMoE stores the router as a single ``[num_experts, hidden]`` weight on the
    block's ``gate`` module. Vector state is excluded, matching the weight-only
    scope of the expert quantizer. Note that ``experts.gate_up_proj`` also
    contains the word gate but is expert weight and is quantized separately.
    """

    references: list[tuple[str, nn.Parameter]] = []
    for spec in layer_specs:
        matrices = [
            (f"{spec.router_name}.{name}", parameter)
            for name, parameter in spec.router.named_parameters(recurse=True)
            if parameter.ndim >= 2
        ]
        if len(matrices) != 1:
            raise RuntimeError(
                f"Router {spec.router_name} exposes {len(matrices)} matrix "
                "parameters; expected exactly one"
            )
        references.append(matrices[0])
    if len(references) != len(layer_specs):
        raise RuntimeError("Did not find one router weight per MoE layer")
    return references


def router_memory_accounting(
    layer_specs: Sequence[MoeLayerSpec],
    model: nn.Module,
    expert_bits_matrix: np.ndarray,
    memory: ExpertMemoryMatrix,
    router_bits: int = SWEEP_C_ROUTER_BITS,
    group_size: int = GROUP_SIZE,
) -> dict[str, Any]:
    """Router parameter count and its share of the deployed model's bytes."""

    references = router_weight_references(layer_specs)
    parameters = sum(int(parameter.numel()) for _, parameter in references)
    bf16_bytes = parameters * 2
    groups = 0
    for _, parameter in references:
        rows = math.prod(parameter.shape[:-1])
        groups += rows * math.ceil(int(parameter.shape[-1]) / group_size)
    quantized_bytes = math.ceil(parameters * router_bits / 8) + groups * 2
    expert_bytes = memory.allocation_bytes(expert_bits_matrix)
    expert_parameters = int(memory.weight_count.sum())
    total_parameters = sum(int(p.numel()) for p in model.parameters())
    non_expert_parameters = total_parameters - expert_parameters
    non_expert_bytes = non_expert_parameters * 2
    deployed_bytes = expert_bytes + non_expert_bytes
    return {
        "router_tensors": [name for name, _ in references],
        "router_tensor_count": len(references),
        "router_parameters": parameters,
        "router_bf16_bytes": bf16_bytes,
        "router_quantized_bits": router_bits,
        "router_quantized_bytes": quantized_bytes,
        "router_group_size": group_size,
        "router_scale_groups": groups,
        "bytes_saved_by_quantizing_routers": bf16_bytes - quantized_bytes,
        "total_model_parameters": total_parameters,
        "expert_parameters": expert_parameters,
        "router_share_of_all_parameters": parameters / total_parameters,
        "deployed_bytes_with_bf16_routers": deployed_bytes,
        "router_share_of_deployed_bytes": bf16_bytes / deployed_bytes,
        "saving_share_of_deployed_bytes": (bf16_bytes - quantized_bytes) / deployed_bytes,
        "expert_bytes_at_this_allocation": expert_bytes,
        "non_expert_bf16_bytes": non_expert_bytes,
    }


class ReversibleRouterQuantization:
    """Quantize every MoE router weight in place and restore it bitwise.

    Uses the same ``symmetric_groupwise_qdq`` the experts use. Enter it only
    while the expert allocation is already applied, and exit it before
    restoring the experts, so the mixed-precision manager's non-expert
    integrity check sees clean routers on both sides.
    """

    def __init__(
        self,
        layer_specs: Sequence[MoeLayerSpec],
        model: nn.Module,
        bits: int = SWEEP_C_ROUTER_BITS,
        group_size: int = GROUP_SIZE,
    ) -> None:
        self.references = router_weight_references(layer_specs)
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self._snapshots: list[torch.Tensor] = []
        self._hooks_before: int | None = None
        self._entered = False
        self.restoration_verified = False
        self.distortions: list[dict[str, Any]] = []

    def __enter__(self) -> "ReversibleRouterQuantization":
        if self._entered:
            raise RuntimeError("A router quantization context cannot be entered twice")
        self._entered = True
        self._hooks_before = module_hook_count(self.model)
        self._snapshots = [
            parameter.detach().clone() for _, parameter in self.references
        ]
        try:
            with torch.no_grad():
                for (name, parameter), original in zip(
                    self.references, self._snapshots, strict=True
                ):
                    result = symmetric_groupwise_qdq(
                        original, bits=self.bits, group_size=self.group_size
                    )
                    parameter.copy_(result.dequantized)
                    self.distortions.append(
                        {
                            "tensor": name,
                            "shape": list(original.shape),
                            "number_of_groups": result.number_of_groups,
                            "relative_squared_error": result.relative_squared_error,
                        }
                    )
            if module_hook_count(self.model) != self._hooks_before:
                raise RuntimeError("Router QDQ changed registered model hooks")
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self._restore()
        for (name, parameter), original in zip(
            self.references, self._snapshots, strict=True
        ):
            if not torch.equal(parameter, original):
                raise RuntimeError(f"Router tensor {name} was not restored bitwise")
        if module_hook_count(self.model) != self._hooks_before:
            raise RuntimeError("Router QDQ leaked or removed a registered hook")
        self.restoration_verified = True
        self._snapshots.clear()

    def _restore(self) -> None:
        if not self._snapshots:
            return
        with torch.no_grad():
            for (_, parameter), original in zip(
                self.references, self._snapshots, strict=True
            ):
                parameter.copy_(original)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "symmetric_groupwise_weight_only_qdq",
            "scope": "MoE router weights",
            "bits": self.bits,
            "group_size": self.group_size,
            "tensors": self.distortions,
            "exact_restoration_verified": self.restoration_verified,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def domain_loss(statistics: LossStatistics) -> float:
    """Mean over examples of each example's mean token cross entropy.

    Identical to the Stage 2B and 2C definition. Every example carries the same
    64 measured tokens, so this equals the token-weighted mean.
    """

    return float(np.mean(statistics.per_token_nll))


def reset_peak_memory(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device_type: str) -> int | None:
    if device_type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    if device_type == "mps" and hasattr(torch, "mps"):
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        if callable(driver):
            return int(driver())
    return None


def evaluate_all_domains(
    bundle: Any,
    examples_by_domain: Mapping[str, PreparedDomainExamples],
    batch_size: int,
    losses_dir: Path | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> tuple[dict[str, LossStatistics], float]:
    """Evaluate every domain once, optionally checkpointing each domain.

    Checkpointing reuses the Stage 1 helper, so an interrupted run resumes at
    domain granularity and a resumed checkpoint is rejected unless its recorded
    configuration matches exactly.
    """

    from .masking import evaluate_next_token_loss
    from .quantization import load_or_compute_loss_checkpoint

    started = time.monotonic()
    statistics: dict[str, LossStatistics] = {}
    for domain in STAGE2B_DOMAINS:
        examples = examples_by_domain[domain]
        token_counts = examples.measurement_mask.sum(axis=1).astype(np.uint32)
        if losses_dir is None:
            result, _ = evaluate_next_token_loss(
                bundle, examples, batch_size=batch_size
            )
        else:
            checkpoint = load_or_compute_loss_checkpoint(
                losses_dir / f"{domain}.npz",
                {**dict(expected_metadata or {}), "domain": domain},
                token_counts,
                lambda examples=examples: evaluate_next_token_loss(
                    bundle, examples, batch_size=batch_size
                ),
                resume=resume,
            )
            result = checkpoint.statistics
        statistics[domain] = result
    return statistics, time.monotonic() - started


def summarize_run_losses(
    loss_by_domain: Mapping[str, float],
    baseline_by_domain: Mapping[str, float],
) -> dict[str, Any]:
    """Both worst-domain definitions, frozen together before any result existed.

    The relative definition is the maximum over domains of the increase over the
    BF16 baseline divided by that baseline, which is what Stages 1 through 2C
    used. The raw definition is the maximum over domains of the absolute loss.
    Step 5 is applied to the relative definition only; the raw one is reported.
    """

    relative = {
        domain: (loss_by_domain[domain] - baseline_by_domain[domain])
        / baseline_by_domain[domain]
        for domain in STAGE2B_DOMAINS
    }
    delta = {
        domain: loss_by_domain[domain] - baseline_by_domain[domain]
        for domain in STAGE2B_DOMAINS
    }
    worst_relative_domain = max(STAGE2B_DOMAINS, key=lambda d: relative[d])
    worst_raw_domain = max(STAGE2B_DOMAINS, key=lambda d: loss_by_domain[d])
    return {
        "loss_by_domain": {d: float(loss_by_domain[d]) for d in STAGE2B_DOMAINS},
        "baseline_loss_by_domain": {
            d: float(baseline_by_domain[d]) for d in STAGE2B_DOMAINS
        },
        "delta_by_domain": {d: float(delta[d]) for d in STAGE2B_DOMAINS},
        "relative_delta_by_domain": {d: float(relative[d]) for d in STAGE2B_DOMAINS},
        "worst_domain_relative": float(relative[worst_relative_domain]),
        "worst_domain_relative_domain": worst_relative_domain,
        "worst_domain_raw": float(loss_by_domain[worst_raw_domain]),
        "worst_domain_raw_domain": worst_raw_domain,
        "mean_relative_delta": float(np.mean([relative[d] for d in STAGE2B_DOMAINS])),
    }


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


def git_commit() -> dict[str, Any]:
    """The commit these results were produced at, and whether the tree was dirty."""

    def capture(arguments: list[str]) -> str | None:
        try:
            return subprocess.run(
                arguments, capture_output=True, text=True, check=True, timeout=30
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = capture(["git", "rev-parse", "HEAD"])
    status = capture(["git", "status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_tree_dirty": None if status is None else bool(status),
    }


def run_config_fingerprint(
    model: str,
    resolved_revision: str | None,
    dtype: str,
    batch_size: int,
    group_size: int,
    split_hashes: Mapping[str, str],
    determinism: Mapping[str, Any],
) -> str:
    """One hash covering everything a run's numbers depend on except its bits."""

    return canonical_sha256(
        {
            "stage": STAGE3D_STAGE,
            "model": model,
            "resolved_model_revision": resolved_revision,
            "dtype": dtype,
            "batch_size": batch_size,
            "group_size": group_size,
            "quantizer": "stage1_symmetric_groupwise_qdq",
            "loss": "teacher_forced_next_token_cross_entropy_fp32",
            "evaluation_set_schema": EVALUATION_SET_SCHEMA,
            "split_input_hashes": dict(split_hashes),
            "deterministic_settings": {
                key: value
                for key, value in determinism.items()
                if key != "torch_version"
            },
        }
    )


def build_run_record(
    protection: ProtectionSet,
    summary: Mapping[str, Any],
    token_counts: Mapping[str, int],
    example_counts: Mapping[str, int],
    config_sha256: str,
    timings: Mapping[str, float],
    peak_memory_bytes: int | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema": RUN_RECORD_SCHEMA,
        "stage": STAGE3D_STAGE,
        "sweep": protection.sweep,
        "run_id": protection.run_id,
        "description": protection.description,
        "selection_rule": protection.selection_rule,
        "regime": protection.regime,
        "base_bits": protection.base_bits,
        "protected_bits": PROTECTED_BITS if protection.regime else None,
        "router_bits": protection.router_bits,
        "seed": protection.seed,
        "protected_expert_count": protection.protected_expert_count,
        "protected_experts": protection.protected_experts,
        "protection_sha256": protection.protection_sha256,
        "bits_matrix_sha256": protection.bits_sha256,
        "config_sha256": config_sha256,
        **git_commit(),
        **dict(summary),
        "tokens_by_domain": {d: int(token_counts[d]) for d in STAGE2B_DOMAINS},
        "examples_by_domain": {d: int(example_counts[d]) for d in STAGE2B_DOMAINS},
        "quantization_seconds": float(timings.get("quantization_seconds", 0.0)),
        "evaluation_seconds": float(timings.get("evaluation_seconds", 0.0)),
        "restoration_seconds": float(timings.get("restoration_seconds", 0.0)),
        "wall_clock_seconds": float(timings.get("wall_clock_seconds", 0.0)),
        "peak_memory_bytes": peak_memory_bytes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        record.update(dict(extra))
    return record


def append_run_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record durably, so a crash loses at most the run in flight."""

    from .io_utils import json_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(dict(record)), sort_keys=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_run_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path} line {number} is not valid JSON; a run was "
                    "interrupted mid-write"
                ) from exc
    return records


def completed_run_ids(path: Path, config_sha256: str) -> set[str]:
    """Run ids already recorded under this exact configuration."""

    return {
        str(record["run_id"])
        for record in read_run_records(path)
        if record.get("config_sha256") == config_sha256
    }


def sweep_jsonl_path(results_dir: Path, sweep: str) -> Path:
    if sweep not in ("a", "b", "c"):
        raise ValueError(f"Unknown sweep {sweep!r}")
    return results_dir / f"stage3d_{sweep}.jsonl"


# ---------------------------------------------------------------------------
# Step 5 decisions
# ---------------------------------------------------------------------------


def _records_by_id(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(record["run_id"]): record for record in records}


def sweep_a_arm_outcome(
    records: Sequence[Mapping[str, Any]], regime: str
) -> dict[str, Any]:
    """Apply the Sweep A rule to one precision arm, mechanically."""

    by_id = _records_by_id(records)
    seed_count = SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME[regime]
    seeds = SWEEP_A_RANDOM_SEEDS[:seed_count]
    missing = [
        run_id
        for run_id in (
            [f"a_{regime}_random_seed{seed}" for seed in seeds]
            + [f"a_{regime}_{name}" for name in SWEEP_A_DELIBERATE_SETS]
        )
        if run_id not in by_id
    ]
    if missing:
        raise RuntimeError(f"Sweep A arm {regime} is missing runs: {missing}")

    random_worst = np.asarray(
        [
            float(by_id[f"a_{regime}_random_seed{seed}"]["worst_domain_relative"])
            for seed in seeds
        ],
        dtype=np.float64,
    )
    most = by_id[f"a_{regime}_most_routed"]
    least = by_id[f"a_{regime}_least_routed"]
    none = by_id[f"a_{regime}_no_protection"]
    standard_deviation = float(
        np.std(random_worst, ddof=SWEEP_A_STANDARD_DEVIATION_DDOF)
    )
    gap = float(least["worst_domain_relative"]) - float(most["worst_domain_relative"])

    if standard_deviation <= 0.0:
        outcome = INCONCLUSIVE
        multiple: float | None = None
    else:
        multiple = gap / standard_deviation
        if gap > SWEEP_A_HEADROOM_MULTIPLE * standard_deviation:
            outcome = HEADROOM
        elif gap < SWEEP_A_FLAT_MULTIPLE * standard_deviation:
            outcome = FLAT
        else:
            outcome = INCONCLUSIVE

    def position(record: Mapping[str, Any]) -> float | None:
        if standard_deviation <= 0.0:
            return None
        return float(
            (float(record["worst_domain_relative"]) - random_worst.mean())
            / standard_deviation
        )

    return {
        "regime": regime,
        "random_sets": len(seeds),
        "random_seeds": [int(seed) for seed in seeds],
        "sd_random": standard_deviation,
        "gap": gap,
        "gap_over_sd_random": multiple,
        "gap_is_negative": bool(gap < 0.0),
        "outcome": outcome,
        "random_worst_domain_relative": {
            "mean": float(random_worst.mean()),
            "sd": standard_deviation,
            "min": float(random_worst.min()),
            "max": float(random_worst.max()),
        },
        "deliberate_sets_in_standard_deviations": {
            "most_routed": position(most),
            "least_routed": position(least),
            "no_protection": position(none),
        },
        "most_routed_worst_domain_relative": float(most["worst_domain_relative"]),
        "least_routed_worst_domain_relative": float(least["worst_domain_relative"]),
        "no_protection_worst_domain_relative": float(none["worst_domain_relative"]),
    }


def sweep_a_decision(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine the two arms under the preregistered precedence rule."""

    arms = {
        regime: sweep_a_arm_outcome(records, regime)
        for regime in SWEEP_A_REGIMES
        if any(record.get("regime") == regime for record in records)
    }
    primary = arms.get(PRIMARY_REGIME)
    if primary is None:
        raise RuntimeError(f"Sweep A needs the {PRIMARY_REGIME} arm to decide")
    secondary = arms.get(SECONDARY_REGIME)

    outcome = primary["outcome"]
    note = f"the {PRIMARY_REGIME} arm decides"
    if (
        outcome == FLAT
        and secondary is not None
        and secondary["outcome"] == HEADROOM
    ):
        outcome = INCONCLUSIVE
        note = (
            f"the {PRIMARY_REGIME} arm is flat but the {SECONDARY_REGIME} arm "
            f"shows headroom on {secondary['random_sets']} random sets; the "
            f"preregistration escalates this to inconclusive and requires the "
            f"full 20 random sets at {BASE_BITS_BY_REGIME[SECONDARY_REGIME]} "
            "bits before any conclusion"
        )
    return {
        "sweep": "a",
        "deciding_arm": PRIMARY_REGIME,
        "outcome": outcome,
        "note": note,
        "arms": arms,
        "meaning": {
            HEADROOM: (
                "expert-level selection has headroom; the full 1024-expert "
                "leave-one-in sweep becomes worth running"
            ),
            FLAT: (
                "the objective is flat at expert granularity; stop pursuing "
                "expert selection"
            ),
            INCONCLUSIVE: "report and ask",
        }[outcome],
        "thresholds": DECISION_THRESHOLDS["sweep_a"],
    }


def sweep_b_decision(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the Sweep B rule, mechanically."""

    by_id = _records_by_id(records)
    missing = [
        f"b_layer{layer:02d}"
        for layer in SWEEP_B_LAYERS
        if f"b_layer{layer:02d}" not in by_id
    ]
    if missing:
        raise RuntimeError(f"Sweep B is missing runs: {missing}")
    increases = {
        layer: float(by_id[f"b_layer{layer:02d}"]["worst_domain_relative"])
        for layer in SWEEP_B_LAYERS
    }
    values = np.asarray([increases[layer] for layer in SWEEP_B_LAYERS])
    largest_layer = int(max(SWEEP_B_LAYERS, key=lambda l: increases[l]))
    smallest_layer = int(min(SWEEP_B_LAYERS, key=lambda l: increases[l]))
    smallest = float(values.min())
    largest = float(values.max())

    if smallest <= 0.0:
        ratio: float | None = None
        outcome = INCONCLUSIVE
        note = (
            f"layer {smallest_layer} has a worst-domain increase of "
            f"{smallest:.6g}, which is not strictly positive, so the ratio is "
            "undefined"
        )
    else:
        ratio = largest / smallest
        note = ""
        if ratio >= SWEEP_B_HEADROOM_RATIO:
            outcome = HEADROOM
        elif ratio <= SWEEP_B_DROP_RATIO:
            outcome = DROP
        else:
            outcome = INCONCLUSIVE
    return {
        "sweep": "b",
        "outcome": outcome,
        "note": note,
        "ratio": ratio,
        "largest_increase": largest,
        "largest_increase_layer": largest_layer,
        "smallest_increase": smallest,
        "smallest_increase_layer": smallest_layer,
        "worst_domain_relative_by_layer": {
            str(layer): increases[layer] for layer in SWEEP_B_LAYERS
        },
        "meaning": {
            HEADROOM: "layer-wise bit allocation has headroom",
            DROP: "drop layer-wise allocation",
            INCONCLUSIVE: "report and ask",
        }[outcome],
        "thresholds": DECISION_THRESHOLDS["sweep_b"],
    }


def sweep_c_report(
    sweep_c_records: Sequence[Mapping[str, Any]],
    sweep_a_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record the router fact. No threshold applies."""

    quantized = _records_by_id(sweep_c_records).get(
        f"c_uniform{SWEEP_C_EXPERT_BITS}_routers_quantized"
    )
    if quantized is None:
        raise RuntimeError("Sweep C is missing its quantized-router run")
    reference = _records_by_id(sweep_a_records).get(
        SWEEP_C_ROUTER_BF16_REFERENCE_RUN_ID
    )
    if reference is None:
        raise RuntimeError(
            "Sweep C needs Sweep A's "
            f"{SWEEP_C_ROUTER_BF16_REFERENCE_RUN_ID} run as its BF16-router "
            "comparison point"
        )
    if reference.get("bits_matrix_sha256") != quantized.get("bits_matrix_sha256"):
        raise RuntimeError(
            "The Sweep C run and its BF16-router reference do not share the "
            "same expert bit assignment"
        )
    difference = float(quantized["worst_domain_relative"]) - float(
        reference["worst_domain_relative"]
    )
    return {
        "sweep": "c",
        "quantized_router_run_id": quantized["run_id"],
        "bf16_router_run_id": reference["run_id"],
        "router_bits": quantized.get("router_bits"),
        "expert_bits": SWEEP_C_EXPERT_BITS,
        "worst_domain_relative_quantized_routers": float(
            quantized["worst_domain_relative"]
        ),
        "worst_domain_relative_bf16_routers": float(
            reference["worst_domain_relative"]
        ),
        "worst_domain_relative_difference": difference,
        "per_domain_difference": {
            domain: float(quantized["relative_delta_by_domain"][domain])
            - float(reference["relative_delta_by_domain"][domain])
            for domain in STAGE2B_DOMAINS
        },
        "router_memory": quantized.get("router_memory"),
        "thresholds": DECISION_THRESHOLDS["sweep_c"],
        "note": (
            "the pipeline never quantized routers, so this is a diagnostic on "
            "baseline strength: it says how much a reviewer could discount a "
            "baseline that leaves routers at BF16"
        ),
    }
