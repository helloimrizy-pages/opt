#!/usr/bin/env python3
"""Independent Stage 2C audit.

This auditor deliberately imports no production analysis code from
``expert_analysis``. Every hash, fragility value, coverage, residual risk,
MILP constraint, budget, split, metric, bootstrap interval, gate, and decision
is recomputed from raw saved artifacts using only the standard library and
numpy. Audit failure blocks progression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DOMAINS = ("general", "math", "coding", "reasoning")
DOMAIN_SEED_OFFSETS = {"general": 11, "math": 23, "coding": 37, "reasoning": 53}
CALIBRATION_SEED = 20260815
CALIBRATION_PER_DOMAIN = 25
NUM_LAYERS = 16
NUM_EXPERTS = 64
PREFIX_IDS = (8982, 27, 187)
SEQUENCE_LENGTH = 68
MEASURED_PER_EXAMPLE = 64
PROTECTED_BITS = 8
BASE_BITS = {"4to8": 4, "3to8": 3}
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RANDOM_SEEDS = (1001, 1002, 1003, 1004, 1005)
SCALE_BITS_PER_GROUP = 16
DEVELOPMENT_BUDGET = 0.20
DEVELOPMENT_SEED = 45
FINAL_SEED = 44
DEVELOPMENT_EXAMPLES = 50
GATE_E_RELATIVE_TOLERANCE = 0.10
GATE_E_EPSILON = 1e-12
FINAL_REQUIRED_POINT_BUDGETS = 3
QUALIFIED_REQUIRED_POINT_BUDGETS = 2
FINAL_REQUIRED_CI_WINS = 2
FINAL_REQUIRED_GLOBAL_POINT_WINS = 2
SYSTEMATIC_NEGATIVE_BUDGETS = 3
EXPECTED_STAGE2B_REGISTRY_SHA = (
    "b0221262f0e51700cc16fa5e6a681f63ab6507a9d768714f853f3dfc3f87aa34"
)
PAIRED_COMPARATORS = (
    "robust_functional", "global_importance", "average_specialization",
    "robust_routing",
)


class Audit:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[dict[str, Any]] = []
        self.sections: dict[str, dict[str, int]] = {}
        self.max_numeric_difference = 0.0

    def check(self, section: str, name: str, condition: bool, detail: Any = None) -> None:
        entry = self.sections.setdefault(section, {"passed": 0, "failed": 0})
        if condition:
            self.passed += 1
            entry["passed"] += 1
        else:
            self.failed += 1
            entry["failed"] += 1
            self.failures.append({"section": section, "check": name, "detail": detail})

    def close(self, value: float, expected: float, tolerance: float = 1e-9) -> bool:
        if value is None:
            value = float("nan")
        if expected is None:
            expected = float("nan")
        value = float(value)
        expected = float(expected)
        if not (math.isfinite(value) and math.isfinite(expected)):
            return value == expected or (math.isnan(value) and math.isnan(expected))
        difference = abs(value - expected)
        self.max_numeric_difference = max(self.max_numeric_difference, difference)
        return difference <= tolerance


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a = rankdata(np.asarray(first, dtype=np.float64))
    b = rankdata(np.asarray(second, dtype=np.float64))
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denominator == 0:
        return float("nan")
    return float((a * b).sum() / denominator)


def expert_bytes(shapes: list[list[int]], bits: int, group_size: int) -> int:
    total = 0
    groups = 0
    for shape in shapes:
        weights = int(np.prod(shape))
        rows = int(np.prod(shape[:-1]))
        if bits == 16:
            total += weights * 2
            continue
        total += math.ceil(weights * bits / 8)
        groups += rows * math.ceil(shape[-1] / group_size)
    return total + groups * (SCALE_BITS_PER_GROUP // 8)


def layer_equalized(raw: np.ndarray) -> np.ndarray:
    totals = raw.sum(axis=1, keepdims=True)
    return raw / totals / raw.shape[0]


def specialization(imp: np.ndarray) -> np.ndarray:
    margins = np.empty_like(imp)
    for d in range(imp.shape[2]):
        others = np.delete(imp, d, axis=2)
        margins[:, :, d] = imp[:, :, d] - others.max(axis=2)
    positive = np.clip(margins, 0.0, None)
    return positive / positive.sum(axis=(0, 1))[None, None, :]


def audit_prior_state(audit: Audit, results_root: Path) -> None:
    section = "frozen_prior_state"
    stage2b = read_json(
        results_root / "robust_specialist_preservation" / "stage2b_decision.json"
    )
    audit.check(
        section, "stage2b_no_go",
        stage2b.get("decision") == "ROBUST_PRESERVATION_NO_GO",
    )
    audit.check(
        section, "stage2b_no_passing_regimes",
        stage2b.get("development_decision", {}).get("passing_regimes") == [],
    )
    audit.check(
        section, "stage2b_registry_sha_recorded",
        stage2b.get("registry_sha256") == EXPECTED_STAGE2B_REGISTRY_SHA,
    )
    surrogate = read_json(
        results_root / "quantization_cost_surrogate" / "surrogate_decision.json"
    )
    audit.check(
        section, "stage2a_no_go", surrogate.get("decision") == "SURROGATE_NO_GO"
    )
    audit.check(
        section, "cost_matrix_unauthorized",
        surrogate.get("full_cost_matrix_authorized") is False,
    )
    stage1 = read_json(
        results_root / "expert_quantization_pilot" / "stage1_decision.json"
    )
    audit.check(section, "stage1_go", stage1.get("decision") == "GO")
    balanced = read_json(
        results_root / "expert_domain_balanced_causal_validation" / "results.json"
    )
    audit.check(
        section, "balanced_strong_go",
        balanced["balanced_analysis"]["decision"].get("label") == "STRONG GO",
    )


def audit_scores(audit: Audit, stage2b_dir: Path, source_dir: Path) -> dict[str, Any]:
    section = "reused_stage2b_scores"
    metadata = read_json(stage2b_dir / "calibration" / "calibration_metadata.json")
    functional_raw = np.zeros((NUM_LAYERS, NUM_EXPERTS, 4))
    calibration_indices: dict[str, list[int]] = {}
    source_rows: dict[str, np.ndarray] = {}
    for index, domain in enumerate(DOMAINS):
        rng = np.random.default_rng([CALIBRATION_SEED, DOMAIN_SEED_OFFSETS[domain]])
        expected_indices = np.sort(
            rng.choice(100, size=CALIBRATION_PER_DOMAIN, replace=False)
        ).tolist()
        recorded = metadata["domains_detail"][domain][
            "calibration_indices_into_frozen_set"
        ]
        audit.check(
            section, f"{domain}_indices_deterministic", recorded == expected_indices
        )
        calibration_indices[domain] = recorded
        with np.load(source_dir / "domains" / f"{domain}.npz", allow_pickle=False) as data:
            contribution = np.asarray(data["contribution_sums"], dtype=np.float64)
        functional_raw[:, :, index] = contribution[recorded].sum(axis=0)
        with np.load(
            source_dir / "controlled_inputs" / f"{domain}.npz", allow_pickle=False
        ) as inputs:
            source_rows[domain] = np.asarray(inputs["input_ids"])
        row_hashes = [
            array_sha256(np.ascontiguousarray(source_rows[domain][i]))
            for i in recorded
        ]
        audit.check(
            section, f"{domain}_calibration_row_hashes",
            row_hashes
            == metadata["domains_detail"][domain]["calibration_input_row_sha256"],
        )
    functional = np.stack(
        [layer_equalized(functional_raw[:, :, d]) for d in range(4)], axis=-1
    )
    functional_spec = specialization(functional)
    with np.load(
        stage2b_dir / "calibration" / "functional_specialization.npz",
        allow_pickle=False,
    ) as data:
        saved_spec = np.asarray(data["specialization"], dtype=np.float64)
    audit.check(
        section, "functional_specialization_matches",
        np.allclose(saved_spec, functional_spec, atol=1e-12),
    )
    audit.check(
        section, "specialization_hash_matches_metadata",
        array_sha256(saved_spec)
        == metadata["score_hashes"]["functional_specialization_sha256"],
    )
    return {
        "functional_spec": functional_spec,
        "calibration_indices": calibration_indices,
        "source_rows": source_rows,
        "metadata": metadata,
    }


def load_losses(losses_dir: Path, slug: str) -> dict[str, np.ndarray]:
    output = {}
    for domain in DOMAINS:
        with np.load(losses_dir / slug / f"{domain}.npz", allow_pickle=False) as data:
            loss_sums = np.asarray(data["loss_sums"], dtype=np.float64)
            token_counts = np.asarray(data["token_counts"], dtype=np.float64)
        output[domain] = loss_sums / token_counts
    return output


def audit_fragility(
    audit: Audit, root: Path, scores: dict[str, Any]
) -> dict[str, Any] | None:
    section = "calibration_fragility"
    path = root / "calibration" / "calibration_fragility.json"
    if not path.is_file():
        return None
    record = read_json(path)
    expected_sha = canonical_sha256(
        {
            k: v
            for k, v in record.items()
            if k not in ("fragility_sha256", "created_at_utc")
        }
    )
    audit.check(
        section, "fragility_sha", record.get("fragility_sha256") == expected_sha
    )
    for domain in DOMAINS:
        subset = record["calibration_subset_hashes"][domain]
        audit.check(
            section, f"{domain}_subset_indices",
            subset["calibration_indices_into_frozen_set"]
            == scores["calibration_indices"][domain],
        )
        expected_rows = array_sha256(
            np.ascontiguousarray(
                scores["source_rows"][domain][scores["calibration_indices"][domain]]
            )
        )
        audit.check(
            section, f"{domain}_subset_input_hash",
            subset["input_ids_sha256"] == expected_rows,
        )
    losses_root = root / "calibration" / "losses"
    nll = {}
    for state in ("bf16", "uniform4", "uniform3"):
        state_nll = {}
        for domain in DOMAINS:
            with np.load(
                losses_root / state / f"{domain}.npz", allow_pickle=False
            ) as data:
                loss_sums = np.asarray(data["loss_sums"], dtype=np.float64)
                token_counts = np.asarray(data["token_counts"], dtype=np.float64)
            audit.check(
                section, f"{state}_{domain}_geometry",
                loss_sums.shape == (CALIBRATION_PER_DOMAIN,)
                and bool(np.all(token_counts == MEASURED_PER_EXAMPLE)),
            )
            state_nll[domain] = float((loss_sums / token_counts).mean())
        nll[state] = state_nll
    fragility_by_regime: dict[str, dict[str, float]] = {}
    for regime, base in BASE_BITS.items():
        entry = record["regimes"][regime]
        audit.check(section, f"{regime}_base_bits", entry["base_bits"] == base)
        clipped = {}
        for domain in DOMAINS:
            values = entry["domains"][domain]
            bf16 = nll["bf16"][domain]
            base_nll = nll[f"uniform{base}"][domain]
            raw = (base_nll - bf16) / bf16
            clip = max(raw, 0.0)
            clipped[domain] = clip
            audit.check(
                section, f"{regime}_{domain}_bf16_nll",
                audit.close(values["bf16_nll"], bf16),
            )
            audit.check(
                section, f"{regime}_{domain}_base_nll",
                audit.close(values["base_nll"], base_nll),
            )
            audit.check(
                section, f"{regime}_{domain}_relative_delta",
                audit.close(values["relative_delta"], raw),
            )
            audit.check(
                section, f"{regime}_{domain}_clipped",
                audit.close(values["clipped_fragility"], clip)
                and values["clipped_fragility"] >= 0,
            )
        mean_clip = float(np.mean([clipped[d] for d in DOMAINS]))
        regime_valid = mean_clip > 0
        audit.check(
            section, f"{regime}_valid_flag", entry["regime_valid"] == regime_valid
        )
        if regime_valid:
            fragility_by_regime[regime] = {}
            for domain in DOMAINS:
                expected_norm = clipped[domain] / mean_clip
                audit.check(
                    section, f"{regime}_{domain}_normalized",
                    audit.close(
                        entry["domains"][domain]["normalized_fragility"],
                        expected_norm,
                    ),
                )
                fragility_by_regime[regime][domain] = expected_norm
            audit.check(
                section, f"{regime}_normalized_mean_one",
                audit.close(
                    float(np.mean(list(fragility_by_regime[regime].values()))), 1.0
                ),
            )
    record["_recomputed_q_norm"] = fragility_by_regime
    return record


def audit_allocations(
    audit: Audit,
    root: Path,
    stage2b_dir: Path,
    scores: dict[str, Any],
    fragility: dict[str, Any] | None,
) -> dict[str, Any] | None:
    section = "stage2c_allocations"
    allocations_dir = root / "allocations"
    registry_path = allocations_dir / "allocation_registry.json"
    if not registry_path.is_file():
        return None
    registry = read_json(registry_path)
    audit.check(section, "registry_frozen", registry.get("frozen") is True)
    expected_sha = canonical_sha256(
        {
            k: v
            for k, v in registry.items()
            if k not in ("registry_sha256", "created_at_utc")
        }
    )
    audit.check(
        section, "registry_sha", registry["registry_sha256"] == expected_sha
    )
    audit.check(
        section, "stage2b_registry_sha",
        registry["stage2b_registry_sha256"] == EXPECTED_STAGE2B_REGISTRY_SHA,
    )
    audit.check(
        section, "seeds_recorded",
        registry["development_seed"] == DEVELOPMENT_SEED
        and registry["final_seed"] == FINAL_SEED,
    )
    with np.load(
        stage2b_dir / "calibration" / "memory_matrix.npz", allow_pickle=False
    ) as data:
        shapes = [list(map(int, shape)) for shape in data["tensor_shapes"].tolist()]
        group_size = int(data["group_size"][0])
        saved_bytes = {
            bits: np.asarray(data[f"bytes_bits{bits}"]) for bits in (3, 4, 8, 16)
        }
    for bits in (3, 4, 8, 16):
        audit.check(
            section, f"memory_bytes_bits{bits}",
            bool(np.all(saved_bytes[bits] == expert_bytes(shapes, bits, group_size))),
        )
    delta = {
        regime: saved_bytes[8] - saved_bytes[base]
        for regime, base in BASE_BITS.items()
    }
    budgets: dict[tuple[str, float], int] = {}
    for regime, base in BASE_BITS.items():
        total = int(delta[regime].sum())
        audit.check(
            section, f"{regime}_total_increment",
            registry["regimes"][regime]["total_increment_bytes"] == total,
        )
        for fraction in FRACTIONS:
            expected_budget = int(math.floor(fraction * total))
            audit.check(
                section, f"{regime}_budget_{fraction}",
                registry["regimes"][regime]["budgets_bytes"][str(fraction)]
                == expected_budget,
            )
            budgets[(regime, fraction)] = expected_budget

    stage2b_registry = read_json(
        stage2b_dir / "allocations" / "allocation_registry.json"
    )
    stage2b_entries = {
        (entry["method"], entry["regime"], entry["budget_fraction"]): entry
        for entry in stage2b_registry["entries"]
    }
    valid_regimes = registry.get("valid_regimes", list(BASE_BITS))
    audit.check(
        section, "new_entry_count",
        len(registry["new_entries"]) == len(valid_regimes) * len(FRACTIONS),
    )
    records: dict[str, dict[str, Any]] = {}
    coverage_by_key: dict[tuple[str, str, float], np.ndarray] = {}
    for entry in registry["reused_entries"]:
        path = stage2b_dir / "allocations" / entry["file"]
        audit.check(section, f"reused_{entry['file']}_exists", path.is_file())
        audit.check(
            section, f"reused_{entry['file']}_file_sha",
            file_sha256(path) == entry["file_sha256"],
        )
        key = (entry["method"], entry["regime"], entry["budget_fraction"])
        stage2b_entry = stage2b_entries.get(key)
        audit.check(
            section, f"reused_{entry['file']}_matches_stage2b_registry",
            stage2b_entry is not None
            and stage2b_entry["allocation_sha256"] == entry["allocation_sha256"]
            and stage2b_entry["file_sha256"] == entry["file_sha256"],
        )
        record = read_json(path)
        records[entry["file"]] = record
        if record["method_kind"] != "uniform_reference":
            protected = (np.asarray(record["expert_bits"]) == PROTECTED_BITS).astype(
                np.float64
            )
            coverage_by_key[key] = np.einsum(
                "led,le->d", scores["functional_spec"], protected
            )
    for entry in registry["new_entries"]:
        path = allocations_dir / entry["file"]
        audit.check(section, f"{entry['file']}_exists", path.is_file())
        audit.check(
            section, f"{entry['file']}_file_sha",
            file_sha256(path) == entry["file_sha256"],
        )
        record = read_json(path)
        records[entry["file"]] = record
        expected_record_sha = canonical_sha256(
            {
                k: v
                for k, v in record.items()
                if k not in ("allocation_sha256", "created_at_utc")
            }
        )
        audit.check(
            section, f"{entry['file']}_allocation_sha",
            record["allocation_sha256"] == expected_record_sha
            and record["allocation_sha256"] == entry["allocation_sha256"],
        )
        audit.check(
            section, f"{entry['file']}_method",
            record["method"] == "fragility_robust",
        )
        regime = record["regime"]
        base = BASE_BITS[regime]
        bits = np.asarray(record["expert_bits"])
        audit.check(
            section, f"{entry['file']}_bits_valid",
            bool(np.all(np.isin(bits, (base, PROTECTED_BITS)))),
        )
        protected = (bits == PROTECTED_BITS).astype(np.int64)
        listed = {(e["layer"], e["expert"]) for e in record["protected_experts"]}
        observed = {tuple(map(int, pair)) for pair in np.argwhere(protected == 1)}
        audit.check(section, f"{entry['file']}_protected_list", listed == observed)
        audit.check(
            section, f"{entry['file']}_protected_count",
            record["protected_expert_count"] == int(protected.sum()),
        )
        used = int((delta[regime] * protected).sum())
        budget = budgets[(regime, record["budget_fraction"])]
        audit.check(
            section, f"{entry['file']}_used_bytes",
            record["used_protection_bytes"] == used,
        )
        audit.check(section, f"{entry['file']}_feasible", used <= budget)
        audit.check(
            section, f"{entry['file']}_budget_bytes", record["budget_bytes"] == budget
        )
        coverage = np.einsum(
            "led,le->d", scores["functional_spec"], protected.astype(np.float64)
        )
        for index, domain in enumerate(DOMAINS):
            audit.check(
                section, f"{entry['file']}_coverage_{domain}",
                audit.close(
                    record["functional_specialist_coverage"][domain],
                    float(coverage[index]),
                ),
            )
        key = (record["method"], regime, record["budget_fraction"])
        coverage_by_key[key] = coverage
        if fragility is not None and regime in fragility["_recomputed_q_norm"]:
            q_norm = np.asarray(
                [fragility["_recomputed_q_norm"][regime][d] for d in DOMAINS]
            )
            residual = q_norm * (1.0 - coverage)
            for index, domain in enumerate(DOMAINS):
                audit.check(
                    section, f"{entry['file']}_residual_{domain}",
                    audit.close(
                        record["predicted_residual_risk"][domain],
                        float(residual[index]),
                        1e-9,
                    ),
                )
            audit.check(
                section, f"{entry['file']}_max_residual",
                audit.close(
                    record["predicted_max_residual_risk"], float(residual.max()), 1e-9
                ),
            )
            audit.check(
                section, f"{entry['file']}_fragility_sha_link",
                record["fragility_sha256"] == fragility["fragility_sha256"],
            )
            # MILP optimality sanity: no frozen comparator at the same
            # regime/budget achieves a lower maximum residual risk.
            optimum = float(residual.max())
            for method in (
                "robust_functional", "robust_routing", "average_specialization",
                "global_importance", "general_only", "math_only", "coding_only",
                "reasoning_only",
                *[f"random_seed{seed}" for seed in RANDOM_SEEDS],
            ):
                comparator_key = (method, regime, record["budget_fraction"])
                comparator_coverage = coverage_by_key.get(comparator_key)
                audit.check(
                    "milp_optimality",
                    f"{entry['file']}_{method}_not_below_optimum",
                    comparator_coverage is not None
                    and float((q_norm * (1.0 - comparator_coverage)).max())
                    >= optimum - 1e-6,
                )
    return {"registry": registry, "records": records, "budgets": budgets}


def audit_preregistration(
    audit: Audit,
    root: Path,
    allocation_state: dict[str, Any] | None,
    fragility: dict[str, Any] | None,
) -> dict[str, Any] | None:
    section = "preregistration"
    path = root / "stage2c_preregistration.json"
    sha_path = root / "preregistration_sha256.txt"
    if not path.is_file():
        return None
    audit.check(section, "sha_file_exists", sha_path.is_file())
    if sha_path.is_file():
        audit.check(
            section, "file_sha_matches",
            file_sha256(path) == sha_path.read_text().strip(),
        )
    prereg = read_json(path)
    audit.check(
        section, "seeds",
        prereg["development_seed"] == DEVELOPMENT_SEED
        and prereg["final_seed"] == FINAL_SEED,
    )
    audit.check(
        section, "budgets", sorted(prereg["protection_budgets"]) == sorted(FRACTIONS)
    )
    if allocation_state is not None:
        registry = allocation_state["registry"]
        audit.check(
            section, "registry_sha_linked",
            prereg["allocation_registry_sha256"] == registry["registry_sha256"],
        )
        expected_hashes = {
            f"{entry['regime']}_budget{int(round(entry['budget_fraction'] * 100))}": (
                entry["allocation_sha256"]
            )
            for entry in registry["new_entries"]
        }
        audit.check(
            section, "allocation_hashes_match",
            prereg["fragility_robust_allocation_hashes"] == expected_hashes,
        )
    if fragility is not None:
        audit.check(
            section, "fragility_sha_linked",
            prereg["calibration"]["fragility_sha256"] == fragility["fragility_sha256"],
        )
        for regime, values in fragility["_recomputed_q_norm"].items():
            for domain in DOMAINS:
                audit.check(
                    section, f"fragility_value_{regime}_{domain}",
                    audit.close(
                        prereg["calibration"]["fragility_values"][regime][domain],
                        values[domain],
                    ),
                )
    return prereg


def audit_seed45_split(
    audit: Audit, root: Path, stage2b_dir: Path, source_dir: Path
) -> dict[str, Any] | None:
    section = "seed45_split"
    manifest_path = root / "splits" / "split_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    audit.check(
        section, "seed", manifest["development_seed"] == DEVELOPMENT_SEED
    )
    audit.check(
        section, "geometry",
        manifest["measured_tokens_per_example"] == MEASURED_PER_EXAMPLE
        and manifest["model_sequence_length"] == SEQUENCE_LENGTH,
    )
    stage2b_manifest = read_json(stage2b_dir / "splits" / "split_manifest.json")
    for domain in DOMAINS:
        entry = manifest["domains"][domain]
        with np.load(
            root / "splits" / "development" / f"{domain}.npz", allow_pickle=False
        ) as data:
            input_ids = np.asarray(data["input_ids"])
            mask = np.asarray(data["measurement_mask"])
        audit.check(
            section, f"{domain}_shape",
            input_ids.shape == (DEVELOPMENT_EXAMPLES, SEQUENCE_LENGTH),
        )
        audit.check(
            section, f"{domain}_hash",
            array_sha256(input_ids) == entry["input_ids_sha256"],
        )
        audit.check(
            section, f"{domain}_prefix",
            bool(np.all(input_ids[:, :3] == np.asarray(PREFIX_IDS))),
        )
        audit.check(
            section, f"{domain}_measured",
            bool(np.all(mask.sum(axis=1) == MEASURED_PER_EXAMPLE)),
        )
        rows = {
            bytes(np.ascontiguousarray(row[3:]).view(np.uint8)) for row in input_ids
        }
        audit.check(
            section, f"{domain}_unique_rows", len(rows) == DEVELOPMENT_EXAMPLES
        )
        with np.load(
            source_dir / "controlled_inputs" / f"{domain}.npz", allow_pickle=False
        ) as data:
            prior_rows = {
                bytes(np.ascontiguousarray(row[3:]).view(np.uint8))
                for row in np.asarray(data["input_ids"])
            }
        audit.check(
            section, f"{domain}_disjoint_from_prior", not (rows & prior_rows)
        )
        for split_name in ("development", "final"):
            with np.load(
                stage2b_dir / "splits" / split_name / f"{domain}.npz",
                allow_pickle=False,
            ) as data:
                other_ids = np.asarray(data["input_ids"])
            audit.check(
                section, f"{domain}_stage2b_{split_name}_hash_intact",
                array_sha256(other_ids)
                == stage2b_manifest["domains"][domain][split_name]["input_ids_sha256"],
            )
            other_rows = {
                bytes(np.ascontiguousarray(row[3:]).view(np.uint8))
                for row in other_ids
            }
            audit.check(
                section, f"{domain}_disjoint_from_seed_{split_name}",
                not (rows & other_rows),
            )
    return manifest


def audit_seed44_isolation(
    audit: Audit, root: Path, stage2b_dir: Path, decision: dict[str, Any] | None
) -> None:
    section = "seed44_isolation"
    stage2b_final_losses = stage2b_dir / "final" / "losses"
    audit.check(
        section, "stage2b_final_losses_absent",
        not (
            stage2b_final_losses.exists() and any(stage2b_final_losses.iterdir())
        ),
    )
    stage2c_final_losses = root / "final_seed44" / "losses"
    final_outputs_exist = stage2c_final_losses.exists() and any(
        stage2c_final_losses.iterdir()
    )
    authorized = decision is not None and decision.get("decision") in (
        "FINAL_CONFIRMATION_GO",
    )
    if decision is not None and decision.get("phase") == "final":
        authorized = True
    audit.check(
        section, "stage2c_final_outputs_only_if_authorized",
        (not final_outputs_exist) or authorized,
        {"final_outputs_exist": final_outputs_exist, "authorized": authorized},
    )


def replicate_indices(count: int, replicates: int, seed: int, domain: str) -> np.ndarray:
    rng = np.random.default_rng([seed, DOMAIN_SEED_OFFSETS[domain]])
    return rng.integers(0, count, size=(replicates, count))


def method_metrics(
    alloc: dict[str, np.ndarray],
    bf16: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative, recovery, delta = {}, {}, {}
    replicates = next(iter(indices.values())).shape[0]
    rel_reps = np.zeros((replicates, 4))
    rec_reps = np.zeros((replicates, 4))
    for j, domain in enumerate(DOMAINS):
        a, b, u = alloc[domain], bf16[domain], base[domain]
        relative[domain] = (a.mean() - b.mean()) / b.mean()
        delta[domain] = a.mean() - b.mean()
        recovery[domain] = u.mean() - a.mean()
        idx = indices[domain]
        rel_reps[:, j] = (a[idx].mean(axis=1) - b[idx].mean(axis=1)) / b[idx].mean(axis=1)
        rec_reps[:, j] = u[idx].mean(axis=1) - a[idx].mean(axis=1)
    point = np.asarray([relative[d] for d in DOMAINS])
    return {
        "relative": relative,
        "delta": delta,
        "recovery": recovery,
        "mean_relative": float(point.mean()),
        "worst_relative": float(point.max()),
        "worst_domain": DOMAINS[int(point.argmax())],
        "worst_reps": rel_reps.max(axis=1),
        "mean_reps": rel_reps.mean(axis=1),
        "worst_ci": (
            float(np.quantile(rel_reps.max(axis=1), 0.025)),
            float(np.quantile(rel_reps.max(axis=1), 0.975)),
        ),
    }


def audit_phase(
    audit: Audit,
    root: Path,
    phase: str,
    allocation_state: dict[str, Any] | None,
    fragility: dict[str, Any] | None,
) -> dict[str, Any] | None:
    phase_dir_names = {"development": "development_seed45", "final": "final_seed44"}
    section = f"{phase}_results"
    phase_dir = root / phase_dir_names[phase]
    results_path = phase_dir / f"{phase}_results.json"
    if not results_path.is_file() or allocation_state is None:
        return None
    results = read_json(results_path)
    run_config = read_json(phase_dir / "run_config.json")
    audit.check(
        section, "run_fingerprint_consistent",
        results["run_fingerprint"] == run_config["run_fingerprint"],
    )
    losses_dir = phase_dir / "losses"
    seed = results["bootstrap_seed"]
    replicates = results["bootstrap_replicates"]
    audit.check(section, "bootstrap_replicates", replicates == 1000)

    bf16 = load_losses(losses_dir, "bf16_reference")
    for domain in DOMAINS:
        audit.check(
            section, f"bf16_{domain}_finite",
            bool(np.all(np.isfinite(bf16[domain])) and np.all(bf16[domain] > 0)),
        )
    base_by_regime = {
        "4to8": load_losses(losses_dir, "uniform_4bit_reference"),
        "3to8": load_losses(losses_dir, "uniform_3bit_reference"),
    }
    counts = {domain: len(bf16[domain]) for domain in DOMAINS}
    expected_count = DEVELOPMENT_EXAMPLES if phase == "development" else 100
    audit.check(
        section, "example_counts",
        all(count == expected_count for count in counts.values()),
    )
    indices = {
        domain: replicate_indices(counts[domain], replicates, seed, domain)
        for domain in DOMAINS
    }

    expected_budgets = (
        [DEVELOPMENT_BUDGET] if phase == "development" else list(FRACTIONS)
    )
    metrics: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for file_name, record in allocation_state["records"].items():
        if record["method_kind"] == "uniform_reference":
            continue
        if record["budget_fraction"] not in expected_budgets:
            continue
        slug = file_name[: -len(".json")]
        if not (losses_dir / slug).exists():
            continue
        alloc = load_losses(losses_dir, slug)
        for domain in DOMAINS:
            metadata = read_json(losses_dir / slug / f"{domain}.metadata.json")
            audit.check(
                section, f"{slug}_{domain}_checkpoint_fingerprints",
                metadata.get("run_fingerprint") == run_config["run_fingerprint"]
                and metadata.get("allocation_sha256") == record["allocation_sha256"],
            )
            audit.check(
                section, f"{slug}_{domain}_finite",
                bool(np.all(np.isfinite(alloc[domain]))),
            )
        key = (record["regime"], record["budget_fraction"])
        metrics.setdefault(key, {})[record["method"]] = method_metrics(
            alloc, bf16, base_by_regime[record["regime"]], indices
        )

    rows_by_key = {
        (row["method"], row["regime"], row["budget_fraction"]): row
        for row in results["method_rows"]
    }
    for (regime, budget), methods in metrics.items():
        for method, metric in methods.items():
            row = rows_by_key[(method, regime, budget)]
            checks = [
                ("worst_relative_delta", metric["worst_relative"]),
                ("mean_relative_delta", metric["mean_relative"]),
                ("worst_relative_delta_ci_low", metric["worst_ci"][0]),
                ("worst_relative_delta_ci_high", metric["worst_ci"][1]),
            ]
            for domain in DOMAINS:
                checks.append((f"relative_delta_{domain}", metric["relative"][domain]))
                checks.append((f"delta_nll_{domain}", metric["delta"][domain]))
                checks.append((f"recovery_{domain}", metric["recovery"][domain]))
            for field, expected in checks:
                audit.check(
                    section, f"{method}_{regime}_{budget}_{field}",
                    audit.close(row[field], expected, 1e-9),
                    {"recorded": row[field], "recomputed": expected},
                )
            audit.check(
                section, f"{method}_{regime}_{budget}_worst_domain",
                row["worst_domain"] == metric["worst_domain"],
            )

    # Recompute paired bootstrap comparisons for Fragility-Robust.
    comparisons_by_key = {
        (c["second"], c["regime"], c["budget_fraction"], c["metric"]): c
        for c in results["comparisons"]
        if c["first"] == "fragility_robust"
    }
    for (regime, budget), methods in metrics.items():
        fragility_robust = methods.get("fragility_robust")
        if fragility_robust is None:
            continue
        randoms = [m for name, m in methods.items() if name.startswith("random_seed")]
        random_mean_worst = float(np.mean([m["worst_relative"] for m in randoms]))
        random_reps = np.stack([m["worst_reps"] for m in randoms]).mean(axis=0)
        recorded = comparisons_by_key[
            ("random_mean", regime, budget, "worst_relative_delta")
        ]
        difference = fragility_robust["worst_reps"] - random_reps
        audit.check(
            section, f"cmp_random_{regime}_{budget}_point",
            audit.close(
                recorded["difference"],
                fragility_robust["worst_relative"] - random_mean_worst,
                1e-9,
            ),
        )
        audit.check(
            section, f"cmp_random_{regime}_{budget}_ci",
            audit.close(
                recorded["difference_ci_low"],
                float(np.quantile(difference, 0.025)), 1e-9,
            )
            and audit.close(
                recorded["difference_ci_high"],
                float(np.quantile(difference, 0.975)), 1e-9,
            ),
        )
        for other in PAIRED_COMPARATORS:
            recorded = comparisons_by_key[
                (other, regime, budget, "worst_relative_delta")
            ]
            difference = (
                fragility_robust["worst_reps"] - methods[other]["worst_reps"]
            )
            audit.check(
                section, f"cmp_{other}_{regime}_{budget}_point",
                audit.close(
                    recorded["difference"],
                    fragility_robust["worst_relative"]
                    - methods[other]["worst_relative"],
                    1e-9,
                ),
            )
            audit.check(
                section, f"cmp_{other}_{regime}_{budget}_ci",
                audit.close(
                    recorded["difference_ci_low"],
                    float(np.quantile(difference, 0.025)), 1e-9,
                )
                and audit.close(
                    recorded["difference_ci_high"],
                    float(np.quantile(difference, 0.975)), 1e-9,
                ),
            )
            audit.check(
                section, f"cmp_{other}_{regime}_{budget}_ci_favors",
                recorded["ci_favors_first"]
                == bool(float(np.quantile(difference, 0.975)) < 0),
            )

    if phase == "development":
        decision = read_json(root / "stage2c_decision.json")
        gates_ok = {}
        for (regime, budget), methods in metrics.items():
            if budget != DEVELOPMENT_BUDGET or "fragility_robust" not in methods:
                continue
            fragility_robust = methods["fragility_robust"]
            randoms = [
                m for name, m in methods.items() if name.startswith("random_seed")
            ]
            random_mean_worst = float(
                np.mean([m["worst_relative"] for m in randoms])
            )
            gate_a = (
                fragility_robust["worst_relative"]
                < methods["robust_functional"]["worst_relative"]
            )
            gate_b = fragility_robust["worst_relative"] < random_mean_worst
            gate_c = (
                fragility_robust["worst_relative"]
                < methods["global_importance"]["worst_relative"]
                and fragility_robust["worst_relative"]
                < methods["average_specialization"]["worst_relative"]
            )
            gate_d = (
                sum(1 for d in DOMAINS if fragility_robust["recovery"][d] > 0) >= 3
            )
            comparator_mean = min(
                methods["global_importance"]["mean_relative"],
                methods["average_specialization"]["mean_relative"],
            )
            relative_worseness = (
                fragility_robust["mean_relative"] - comparator_mean
            ) / max(abs(comparator_mean), GATE_E_EPSILON)
            gate_e = relative_worseness <= GATE_E_RELATIVE_TOLERANCE
            gates_ok[regime] = gate_a and gate_b and gate_c and gate_d and gate_e
            recorded_gates = decision["development_gates"][regime]
            for name, value in (
                ("gate_a", gate_a), ("gate_b", gate_b), ("gate_c", gate_c),
                ("gate_d", gate_d), ("gate_e", gate_e),
            ):
                audit.check(
                    section, f"{regime}_{name}_recomputed",
                    recorded_gates[name]["passed"] == value,
                    {"recorded": recorded_gates[name]["passed"], "recomputed": value},
                )
        expected_decision = (
            "FINAL_CONFIRMATION_GO"
            if any(gates_ok.values())
            else "FRAGILITY_ROBUST_NO_GO"
        )
        audit.check(
            section, "development_decision_recomputed",
            decision["decision"] == expected_decision,
            {"recorded": decision["decision"], "recomputed": expected_decision},
        )
        expected_authorized = sorted(
            regime for regime, value in gates_ok.items() if value
        )
        audit.check(
            section, "authorized_regimes_recomputed",
            sorted(decision["development_decision"]["authorized_regimes"])
            == expected_authorized,
        )
    else:
        regimes = sorted({key[0] for key in metrics})
        for regime in regimes:
            all_four = 0
            improvements = []
            ci_wins = 0
            global_point_wins = 0
            beats_rf = 0
            beats_random = 0
            beats_both_simple = 0
            for budget in FRACTIONS:
                methods = metrics[(regime, budget)]
                fragility_robust = methods["fragility_robust"]
                randoms = [
                    m for name, m in methods.items()
                    if name.startswith("random_seed")
                ]
                random_mean_worst = float(
                    np.mean([m["worst_relative"] for m in randoms])
                )
                win_rf = (
                    fragility_robust["worst_relative"]
                    < methods["robust_functional"]["worst_relative"]
                )
                win_random = fragility_robust["worst_relative"] < random_mean_worst
                win_global = (
                    fragility_robust["worst_relative"]
                    < methods["global_importance"]["worst_relative"]
                )
                win_average = (
                    fragility_robust["worst_relative"]
                    < methods["average_specialization"]["worst_relative"]
                )
                all_four += int(win_rf and win_random and win_global and win_average)
                beats_rf += int(win_rf)
                beats_random += int(win_random)
                beats_both_simple += int(win_global and win_average)
                improvements.append(
                    methods["average_specialization"]["worst_relative"]
                    - fragility_robust["worst_relative"]
                )
                difference = (
                    fragility_robust["worst_reps"]
                    - methods["average_specialization"]["worst_reps"]
                )
                if float(np.quantile(difference, 0.975)) < 0:
                    ci_wins += 1
                if (
                    fragility_robust["worst_relative"]
                    < methods["global_importance"]["worst_relative"]
                ):
                    global_point_wins += 1
            systematic = []
            for domain in DOMAINS:
                negative = sum(
                    1
                    for budget in FRACTIONS
                    if metrics[(regime, budget)]["fragility_robust"]["recovery"][
                        domain
                    ]
                    < 0
                )
                if negative >= SYSTEMATIC_NEGATIVE_BUDGETS:
                    systematic.append(domain)
            average_improvement = float(np.mean(improvements))
            requirement_1 = all_four >= FINAL_REQUIRED_POINT_BUDGETS
            requirement_2 = average_improvement > 0
            requirement_3 = ci_wins >= FINAL_REQUIRED_CI_WINS
            requirement_4 = global_point_wins >= FINAL_REQUIRED_GLOBAL_POINT_WINS
            requirement_5 = not systematic
            strong = (
                requirement_1 and requirement_2 and requirement_3
                and requirement_4 and requirement_5
            )
            qualified = (
                not strong
                and requirement_2
                and requirement_5
                and (
                    beats_both_simple >= QUALIFIED_REQUIRED_POINT_BUDGETS
                    or (
                        beats_rf >= FINAL_REQUIRED_POINT_BUDGETS
                        and beats_random >= FINAL_REQUIRED_POINT_BUDGETS
                    )
                )
            )
            recorded = results["final_regime_assessments"][regime]
            audit.check(
                section, f"{regime}_strong_recomputed",
                recorded["strong_success"] == strong,
            )
            audit.check(
                section, f"{regime}_qualified_recomputed",
                recorded["qualified_success"] == qualified,
            )
            audit.check(
                section, f"{regime}_average_improvement",
                audit.close(
                    recorded["average_improvement_over_average_specialization"],
                    average_improvement, 1e-9,
                ),
            )
        strong_any = any(
            results["final_regime_assessments"][r]["strong_success"] for r in regimes
        )
        qualified_any = any(
            results["final_regime_assessments"][r]["qualified_success"]
            for r in regimes
        )
        if strong_any:
            expected_label = "STRONG SUCCESS"
        elif qualified_any:
            expected_label = "SUCCESS WITH QUALIFICATIONS"
        else:
            expected_label = "NEGATIVE RESULT"
        audit.check(
            section, "final_decision_recomputed",
            results["final_decision"]["decision"] == expected_label,
        )
        # Mechanism analyses: recompute Spearman values descriptively.
        if fragility is not None:
            for regime, entry in results.get(
                "fragility_transfer_check", {}
            ).get("regimes", {}).items():
                q_norm = np.asarray(
                    [fragility["_recomputed_q_norm"][regime][d] for d in DOMAINS]
                )
                base = base_by_regime[regime]
                observed = np.asarray(
                    [
                        (base[d].mean() - bf16[d].mean()) / bf16[d].mean()
                        for d in DOMAINS
                    ]
                )
                audit.check(
                    section, f"transfer_{regime}_spearman",
                    audit.close(entry["spearman"], spearman(q_norm, observed), 1e-9),
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
    )
    parser.add_argument(
        "--stage2b-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.results_dir
    audit = Audit()

    audit_prior_state(audit, args.results_root)
    scores = audit_scores(audit, args.stage2b_dir, args.source_dir)
    fragility = audit_fragility(audit, root, scores)
    allocation_state = audit_allocations(
        audit, root, args.stage2b_dir, scores, fragility
    )
    audit_preregistration(audit, root, allocation_state, fragility)
    audit_seed45_split(audit, root, args.stage2b_dir, args.source_dir)
    decision_path = root / "stage2c_decision.json"
    decision = read_json(decision_path) if decision_path.is_file() else None
    audit_seed44_isolation(audit, root, args.stage2b_dir, decision)
    development = audit_phase(audit, root, "development", allocation_state, fragility)
    final = audit_phase(audit, root, "final", allocation_state, fragility)

    report = {
        "auditor": "standalone_stage2c_audit_v1",
        "production_analysis_functions_imported": False,
        "passed": audit.failed == 0,
        "checks_passed": audit.passed,
        "checks_failed": audit.failed,
        "max_numeric_difference": audit.max_numeric_difference,
        "sections": audit.sections,
        "fragility_audited": fragility is not None,
        "allocations_audited": allocation_state is not None,
        "development_results_audited": development is not None,
        "final_results_audited": final is not None,
        "failures": audit.failures[:200],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output or (root / "audits" / "independent_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Independent audit: {audit.passed} passed, {audit.failed} failed; "
        f"max numeric difference {audit.max_numeric_difference:.3e}"
    )
    print(f"Report: {output}")
    return 0 if audit.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
