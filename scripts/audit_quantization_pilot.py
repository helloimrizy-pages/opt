#!/usr/bin/env python3
"""Independent raw-artifact audit for the frozen Stage-1 quantization pilot.

This module deliberately does not import the production quantization analysis. It
reconstructs the preregistered statistics directly from checkpoint NPZ/JSON files
and compares them with every published JSON/CSV result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr


DOMAINS = ("general", "math", "coding", "reasoning")
MODEL = "allenai/OLMoE-1B-7B-0924"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
PANEL_FINGERPRINT = "404927664048259fb623a7b3181e811c8f18c68d5e32825b943b056257220af7"
EXAMPLES = 100
TOKENS_PER_EXAMPLE = 64
BOOTSTRAP_REPLICATES = 1000
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--balanced-results-dir",
        type=Path,
        default=Path("results/expert_domain_balanced_causal_validation"),
    )
    parser.add_argument(
        "--pilot-results-dir",
        type=Path,
        default=Path("results/expert_quantization_pilot"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/expert_quantization_pilot/independent_audit.json"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def derived_seed(label: str) -> int:
    digest = canonical_sha256({"seed": SEED, "label": label})
    return int(digest[:16], 16) % (2**63 - 1)


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def finite_ci(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return (None, None) if finite.size == 0 else ci(finite)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.all(first == first[0]) or np.all(second == second[0]):
        return math.nan
    return float(spearmanr(first, second).statistic)


def safe_kendall(first: np.ndarray, second: np.ndarray) -> float:
    if np.all(first == first[0]) or np.all(second == second[0]):
        return math.nan
    return float(kendalltau(first, second).statistic)


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.comparisons = 0
        self.maximum_absolute_numeric_difference = 0.0

    def require(self, condition: bool, message: str) -> None:
        self.comparisons += 1
        if not condition:
            self.errors.append(message)

    def close(
        self,
        observed: Any,
        expected: Any,
        message: str,
        *,
        atol: float = 1e-15,
    ) -> None:
        self.comparisons += 1
        try:
            difference = abs(float(observed) - float(expected))
        except (TypeError, ValueError):
            self.errors.append(f"{message}: non-numeric value")
            return
        if math.isfinite(difference):
            self.maximum_absolute_numeric_difference = max(
                self.maximum_absolute_numeric_difference, difference
            )
        if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=atol):
            self.errors.append(
                f"{message}: observed={observed!r}, expected={expected!r}, "
                f"abs_diff={difference:.17g}"
            )


def load_loss(path: Path, checks: Checks) -> tuple[np.ndarray, np.ndarray]:
    checks.require(path.is_file(), f"missing loss checkpoint {path}")
    with np.load(path, allow_pickle=False) as data:
        checks.require(
            set(data.files) == {"loss_sums", "token_counts"},
            f"unexpected checkpoint fields in {path}",
        )
        loss_sums = data["loss_sums"].astype(np.float64)
        token_counts = data["token_counts"].astype(np.uint32)
    checks.require(loss_sums.shape == (EXAMPLES,), f"wrong loss shape in {path}")
    checks.require(token_counts.shape == (EXAMPLES,), f"wrong token shape in {path}")
    checks.require(np.all(np.isfinite(loss_sums)), f"non-finite losses in {path}")
    checks.require(np.all(token_counts == TOKENS_PER_EXAMPLE), f"token mismatch in {path}")
    return loss_sums, token_counts


def panel_rows(panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in panel["pairs"]:
        specialist = pair["specialist"]
        control = pair["matched_control"]
        rows.append(
            {
                "intervention_id": f"specialist_{pair['pair_id']}",
                "pair_id": pair["pair_id"],
                "role": "specialist",
                "target_domain": pair["target_domain"],
                "layer": int(specialist["layer"]),
                "expert_id": int(specialist["expert_id"]),
                "record": specialist,
            }
        )
        rows.append(
            {
                "intervention_id": (
                    f"control_{pair['pair_id']}_L{control['layer']}_E{control['expert_id']}"
                ),
                "pair_id": pair["pair_id"],
                "role": "control",
                "target_domain": pair["target_domain"],
                "layer": int(control["layer"]),
                "expert_id": int(control["expert_id"]),
                "record": control,
            }
        )
    return rows


def compare_csv(
    path: Path, expected_rows: Sequence[Mapping[str, Any]], checks: Checks
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        observed_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    checks.require(len(observed_rows) == len(expected_rows), f"row count mismatch in {path}")
    if expected_rows:
        checks.require(
            set(fieldnames) == set(expected_rows[0]), f"column mismatch in {path}"
        )
    for row_index, (observed, expected) in enumerate(
        zip(observed_rows, expected_rows, strict=False)
    ):
        for key, value in expected.items():
            cell = observed.get(key)
            label = f"{path.name} row {row_index} field {key}"
            if value is None:
                checks.require(cell == "", label)
            elif isinstance(value, bool):
                checks.require(cell == str(value), label)
            elif isinstance(value, (int, np.integer)):
                checks.require(cell == str(int(value)), label)
            elif isinstance(value, (float, np.floating)):
                checks.close(cell, value, label)
            else:
                checks.require(cell == str(value), label)
    return {"rows": len(observed_rows), "columns": fieldnames}


def published_by(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    balanced = args.balanced_results_dir.resolve()
    pilot = args.pilot_results_dir.resolve()
    checks = Checks()

    panel = read_json(pilot / "pilot_panel_preregistered.json")
    run_config = read_json(pilot / "run_config.json")
    results = read_json(pilot / "results.json")
    decision_file = read_json(pilot / "stage1_decision.json")
    balanced_results = read_json(balanced / "results.json")
    balanced_integrity = read_json(balanced / "integrity_validation.json")
    analysis = results["quantization_analysis"]

    panel_keys = (
        "schema_version",
        "selection_version",
        "source",
        "selection",
        "quantization_preregistration",
        "analysis_preregistration",
        "pairs",
    )
    panel_hash = canonical_sha256({key: panel[key] for key in panel_keys})
    checks.require(panel_hash == PANEL_FINGERPRINT, "pilot panel canonical hash mismatch")
    checks.require(
        panel.get("pilot_panel_fingerprint") == PANEL_FINGERPRINT,
        "pilot panel recorded fingerprint mismatch",
    )
    checks.require(panel.get("status") == "FROZEN_BEFORE_QUANTIZATION", "panel not frozen")
    checks.require(len(panel["pairs"]) == 8, "pilot does not contain eight pairs")
    checks.require(
        all(
            len([pair for pair in panel["pairs"] if pair["target_domain"] == domain]) == 2
            for domain in DOMAINS
        ),
        "pilot is not two pairs per domain",
    )
    checks.require(panel["selection"]["masking_effect_sizes_used"] is False, "masking used")
    checks.require(panel["selection"]["quantization_results_used"] is False, "QDQ used")

    expected_config = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "pilot_panel_fingerprint": PANEL_FINGERPRINT,
        "method": "symmetric_groupwise_weight_only_fake_quantization_qdq",
        "expert_scope": "one_expert_ffn_at_a_time",
        "group_axis": "last_dimension_input_features",
        "group_size": 128,
        "scale_storage_dtype": "float16",
        "primary_bits": 4,
        "fallback_bits": 3,
        "fallback_trigger": "only_preregistered_too_small_to_measure_condition",
        "device": "cuda",
        "required_hardware": "NVIDIA A40",
        "dtype": "bfloat16",
        "batch_size": 1,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
    }
    for key, value in expected_config.items():
        checks.require(run_config.get(key) == value, f"run_config mismatch for {key}")
    checks.require(
        results["run_config"].get("runtime_description") == "NVIDIA A40",
        "runtime is not NVIDIA A40",
    )
    fingerprint_keys = (
        "model",
        "model_revision",
        "source_collection_fingerprint",
        "source_input_fingerprint",
        "pilot_panel_fingerprint",
        "balanced_masking_raw_sha256",
        "method",
        "expert_scope",
        "group_axis",
        "group_size",
        "scale_storage_dtype",
        "primary_bits",
        "fallback_bits",
        "fallback_trigger",
        "device",
        "required_hardware",
        "dtype",
        "batch_size",
        "seed",
        "package_versions",
        "interventions",
    )
    inference_hash = canonical_sha256({key: run_config[key] for key in fingerprint_keys})
    checks.require(
        inference_hash == run_config.get("inference_fingerprint"),
        "inference fingerprint mismatch",
    )

    input_hash_checks: dict[str, str] = {}
    for relative, expected_hash in panel["source"]["controlled_input_file_sha256"].items():
        path = source / "controlled_inputs" / f"{relative}.npz"
        observed_hash = file_sha256(path)
        input_hash_checks[relative] = observed_hash
        checks.require(observed_hash == expected_hash, f"controlled input hash mismatch: {relative}")
    balanced_panel_path = balanced / "selected_experts_preregistered.json"
    matched_controls_path = balanced / "matched_controls.csv"
    checks.require(
        file_sha256(balanced_panel_path)
        == panel["source"]["balanced_preregistration_sha256"],
        "balanced preregistration file hash mismatch",
    )
    checks.require(
        file_sha256(matched_controls_path)
        == panel["source"]["balanced_matched_controls_sha256"],
        "matched-control file hash mismatch",
    )
    checks.require(balanced_integrity.get("passed") is True, "balanced integrity failed")

    balanced_raw_path = balanced / "per_example_loss_changes.npz"
    balanced_raw_hash = file_sha256(balanced_raw_path)
    checks.require(
        balanced_raw_hash == run_config["balanced_masking_raw_sha256"],
        "balanced raw masking hash differs from run config",
    )
    checks.require(
        balanced_raw_hash
        == balanced_results["artifact_manifest"]["file_sha256"][
            "per_example_loss_changes.npz"
        ],
        "balanced raw masking hash differs from balanced manifest",
    )

    rows = panel_rows(panel)
    expected_identity_rows = [
        {key: row[key] for key in ("intervention_id", "pair_id", "role", "target_domain", "layer", "expert_id")}
        for row in rows
    ]
    checks.require(run_config["interventions"] == expected_identity_rows, "run panel mismatch")

    baseline_nll = np.empty((len(DOMAINS), EXAMPLES), dtype=np.float64)
    token_counts = np.empty((len(DOMAINS), EXAMPLES), dtype=np.uint32)
    baseline_reproduction = read_json(pilot / "baseline_reproduction.json")
    for domain_index, domain in enumerate(DOMAINS):
        loss_sums, tokens = load_loss(
            pilot / "quantization" / "baseline" / f"{domain}.npz", checks
        )
        balanced_sums, balanced_tokens = load_loss(
            balanced / "masking" / "baseline" / f"{domain}.npz", checks
        )
        checks.require(np.array_equal(loss_sums, balanced_sums), f"baseline drift: {domain}")
        checks.require(np.array_equal(tokens, balanced_tokens), f"token drift: {domain}")
        baseline_nll[domain_index] = loss_sums / tokens
        token_counts[domain_index] = tokens
        baseline_meta = read_json(
            pilot / "quantization" / "baseline" / f"{domain}.metadata.json"
        )
        checks.require(
            baseline_meta.get("inference_fingerprint") == inference_hash,
            f"baseline metadata fingerprint mismatch: {domain}",
        )
        checks.require(
            baseline_meta.get("diagnostics", {}).get("hooks_before")
            == baseline_meta.get("diagnostics", {}).get("hooks_after"),
            f"baseline hook leakage: {domain}",
        )
        reproduction = baseline_reproduction["domains"][domain]
        checks.require(reproduction.get("bitwise_equal") is True, f"baseline not bitwise: {domain}")
        checks.close(
            reproduction["max_absolute_per_token_nll_difference"],
            0.0,
            f"baseline numerical noise: {domain}",
        )

    quantized_nll = np.empty((len(rows), len(DOMAINS), EXAMPLES), dtype=np.float64)
    distortions = np.empty(len(rows), dtype=np.float64)
    metadata_summary: list[dict[str, Any]] = []
    for intervention_index, row in enumerate(rows):
        directory = (
            pilot
            / "quantization"
            / "bit_4"
            / f"layer_{row['layer']}_expert_{row['expert_id']}"
        )
        qmeta = read_json(directory / "quantization.metadata.json")
        checks.require(qmeta.get("inference_fingerprint") == inference_hash, "QDQ fingerprint mismatch")
        checks.require(qmeta.get("bits") == 4 and qmeta.get("group_size") == 128, "QDQ config mismatch")
        checks.require(qmeta.get("exact_restoration_verified") is True, "restoration not exact")
        checks.require(
            qmeta.get("unrelated_experts_verified_unchanged") is True,
            "unrelated experts not verified",
        )
        checks.require(qmeta.get("hooks_before") == qmeta.get("hooks_after"), "QDQ hook leakage")
        checks.require(
            qmeta.get("original_expert_fingerprint") != qmeta.get("quantized_expert_fingerprint"),
            "QDQ did not change expert fingerprint",
        )
        for key in ("original_expert_fingerprint", "quantized_expert_fingerprint"):
            checks.require(
                isinstance(qmeta.get(key), str) and len(qmeta[key]) == 64,
                f"invalid {key}",
            )
        distortion = float(qmeta["quantization_distortion"])
        checks.require(math.isfinite(distortion) and distortion > 0, "invalid distortion")
        distortions[intervention_index] = distortion
        memory = qmeta["memory_accounting"]
        exact_memory = {
            "weight_count": 6291456,
            "number_of_groups": 49152,
            "quantized_weight_payload_bits": 25165824,
            "scale_storage_bits": 786432,
            "projected_bytes": 3244032,
            "effective_bits_per_weight": 4.125,
        }
        for key, value in exact_memory.items():
            if isinstance(value, float):
                checks.close(memory.get(key), value, f"memory accounting {key}")
            else:
                checks.require(memory.get(key) == value, f"memory accounting {key}")
        for domain_index, domain in enumerate(DOMAINS):
            loss_sums, tokens = load_loss(directory / f"{domain}.npz", checks)
            checks.require(
                np.array_equal(tokens, token_counts[domain_index]),
                f"quantized token mismatch: {row['intervention_id']}/{domain}",
            )
            quantized_nll[intervention_index, domain_index] = loss_sums / tokens
            domain_meta = read_json(directory / f"{domain}.metadata.json")
            for key, expected in (
                ("inference_fingerprint", inference_hash),
                ("pilot_panel_fingerprint", PANEL_FINGERPRINT),
                ("intervention_id", row["intervention_id"]),
                ("pair_id", row["pair_id"]),
                ("role", row["role"]),
                ("target_domain", row["target_domain"]),
                ("layer", row["layer"]),
                ("expert_id", row["expert_id"]),
                ("bit_width", 4),
                ("group_size", 128),
                ("domain", domain),
            ):
                checks.require(domain_meta.get(key) == expected, f"checkpoint metadata {key}")
            checks.require(
                domain_meta.get("original_expert_fingerprint")
                == qmeta.get("original_expert_fingerprint"),
                "checkpoint original fingerprint mismatch",
            )
            checks.require(
                domain_meta.get("quantized_expert_fingerprint")
                == qmeta.get("quantized_expert_fingerprint"),
                "checkpoint quantized fingerprint mismatch",
            )
            checks.require(
                domain_meta.get("diagnostics", {}).get("hooks_before")
                == domain_meta.get("diagnostics", {}).get("hooks_after"),
                "checkpoint hook leakage",
            )
        metadata_summary.append(
            {
                "intervention_id": row["intervention_id"],
                "original_expert_fingerprint": qmeta["original_expert_fingerprint"],
                "quantized_expert_fingerprint": qmeta["quantized_expert_fingerprint"],
                "distortion": distortion,
                "exact_restoration_verified": True,
            }
        )

    checks.require(not (pilot / "quantization" / "bit_3").exists(), "unexpected 3-bit output")
    delta = quantized_nll - baseline_nll[None, :, :]
    checks.require(np.all(np.isfinite(delta)), "raw quantization deltas are non-finite")
    domain_means = delta.mean(axis=-1)

    with np.load(pilot / "per_example_quantization_losses.npz", allow_pickle=False) as data:
        checks.require(data["per_example_loss_changes"].shape == (1, 16, 4, 100), "final NPZ shape")
        checks.require(np.array_equal(data["bit_widths"], np.asarray([4], dtype=np.int8)), "bit widths")
        checks.require(
            np.array_equal(data["intervention_ids"], np.asarray([row["intervention_id"] for row in rows])),
            "final NPZ intervention order",
        )
        checks.require(np.array_equal(data["domain_names"], np.asarray(DOMAINS)), "domain order")
        checks.require(np.array_equal(data["token_counts"], token_counts), "final NPZ tokens")
        checks.require(np.array_equal(data["baseline_per_example_nll"], baseline_nll), "final NPZ baselines")
        checks.require(
            np.array_equal(data["quantized_per_example_nll"][0], quantized_nll),
            "final NPZ quantized losses",
        )
        checks.require(
            np.array_equal(data["per_example_loss_changes"][0], delta),
            "final NPZ loss changes",
        )
        checks.require(
            np.array_equal(data["quantization_distortion"][0], distortions),
            "final NPZ distortions",
        )
        for key in data.files:
            if np.issubdtype(data[key].dtype, np.floating):
                checks.require(np.all(np.isfinite(data[key])), f"non-finite final NPZ field {key}")

    indices = {}
    boot_domain = np.empty((len(rows), len(DOMAINS), BOOTSTRAP_REPLICATES))
    for domain_index, domain in enumerate(DOMAINS):
        rng = np.random.default_rng(derived_seed(f"pilot-bootstrap-{domain}"))
        indices[domain] = rng.integers(
            0, EXAMPLES, size=(BOOTSTRAP_REPLICATES, EXAMPLES), dtype=np.int64
        )
        boot_domain[:, domain_index, :] = delta[:, domain_index, :][
            :, indices[domain]
        ].mean(axis=-1)

    contrast_boot = np.empty((len(rows), BOOTSTRAP_REPLICATES), dtype=np.float64)
    contrast_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    pairwise_records: list[dict[str, Any]] = []
    for intervention_index, row in enumerate(rows):
        target_index = DOMAINS.index(row["target_domain"])
        other_indices = [index for index in range(4) if index != target_index]
        boot = boot_domain[intervention_index]
        contrast_boot[intervention_index] = boot[target_index] - boot[other_indices].mean(axis=0)
        target_low, target_high = ci(boot[target_index])
        other_low, other_high = ci(boot[other_indices].mean(axis=0))
        contrast_low, contrast_high = ci(contrast_boot[intervention_index])
        contrast_records.append(
            {
                "bit_width": 4,
                **{key: row[key] for key in ("intervention_id", "pair_id", "role", "target_domain", "layer", "expert_id")},
                "target_delta_nll": float(domain_means[intervention_index, target_index]),
                "target_delta_nll_ci_low": target_low,
                "target_delta_nll_ci_high": target_high,
                "mean_non_target_delta_nll": float(domain_means[intervention_index, other_indices].mean()),
                "mean_non_target_delta_nll_ci_low": other_low,
                "mean_non_target_delta_nll_ci_high": other_high,
                "target_minus_mean_other_contrast": float(
                    domain_means[intervention_index, target_index]
                    - domain_means[intervention_index, other_indices].mean()
                ),
                "contrast_ci_low": contrast_low,
                "contrast_ci_high": contrast_high,
            }
        )
        record = row["record"]
        for domain_index, domain in enumerate(DOMAINS):
            low, high = ci(boot[domain_index])
            functional = float(record["normalized_contribution_by_domain"][domain])
            routing = float(record["routing_frequency_by_domain"][domain])
            result_records.append(
                {
                    "bit_width": 4,
                    **{key: row[key] for key in ("intervention_id", "pair_id", "role", "target_domain", "layer", "expert_id")},
                    "domain": domain,
                    "baseline_nll": float(baseline_nll[domain_index].mean()),
                    "quantized_nll": float(quantized_nll[intervention_index, domain_index].mean()),
                    "delta_nll": float(domain_means[intervention_index, domain_index]),
                    "delta_nll_ci_low": low,
                    "delta_nll_ci_high": high,
                    "quantization_distortion": float(distortions[intervention_index]),
                    "risk_functional": functional * float(distortions[intervention_index]),
                    "risk_routing": routing * float(distortions[intervention_index]),
                }
            )
        for comparison_index in other_indices:
            pair_boot = boot[target_index] - boot[comparison_index]
            low, high = ci(pair_boot)
            pairwise_records.append(
                {
                    "bit_width": 4,
                    **{key: row[key] for key in ("intervention_id", "pair_id", "role", "target_domain", "layer", "expert_id")},
                    "comparison_domain": DOMAINS[comparison_index],
                    "target_delta_nll": float(domain_means[intervention_index, target_index]),
                    "comparison_delta_nll": float(domain_means[intervention_index, comparison_index]),
                    "target_minus_comparison_delta_nll": float(
                        domain_means[intervention_index, target_index]
                        - domain_means[intervention_index, comparison_index]
                    ),
                    "contrast_ci_low": low,
                    "contrast_ci_high": high,
                }
            )

    published_contrasts = published_by(
        analysis["intervention_contrasts"], ("bit_width", "intervention_id")
    )
    for computed in contrast_records:
        published = published_contrasts[(4, computed["intervention_id"])]
        for field in (
            "target_delta_nll",
            "target_delta_nll_ci_low",
            "target_delta_nll_ci_high",
            "mean_non_target_delta_nll",
            "mean_non_target_delta_nll_ci_low",
            "mean_non_target_delta_nll_ci_high",
            "target_minus_mean_other_contrast",
            "contrast_ci_low",
            "contrast_ci_high",
        ):
            checks.close(published[field], computed[field], f"contrast {computed['intervention_id']} {field}")

    published_results = published_by(
        analysis["quantization_pilot_results"], ("bit_width", "intervention_id", "domain")
    )
    for computed in result_records:
        published = published_results[(4, computed["intervention_id"], computed["domain"])]
        for field in (
            "baseline_nll",
            "quantized_nll",
            "delta_nll",
            "delta_nll_ci_low",
            "delta_nll_ci_high",
            "quantization_distortion",
            "risk_functional",
            "risk_routing",
        ):
            checks.close(
                published[field], computed[field],
                f"domain result {computed['intervention_id']}/{computed['domain']} {field}",
            )

    published_pairwise = published_by(
        analysis["quantization_pilot_pairwise"],
        ("bit_width", "intervention_id", "comparison_domain"),
    )
    for computed in pairwise_records:
        published = published_pairwise[(4, computed["intervention_id"], computed["comparison_domain"])]
        for field in (
            "target_delta_nll",
            "comparison_delta_nll",
            "target_minus_comparison_delta_nll",
            "contrast_ci_low",
            "contrast_ci_high",
        ):
            checks.close(published[field], computed[field], f"pairwise {computed['intervention_id']} {field}")

    contrast_by_id = {row["intervention_id"]: row for row in contrast_records}
    row_index = {row["intervention_id"]: index for index, row in enumerate(rows)}
    paired_records: list[dict[str, Any]] = []
    for pair in panel["pairs"]:
        specialist_id = f"specialist_{pair['pair_id']}"
        control = pair["matched_control"]
        control_id = f"control_{pair['pair_id']}_L{control['layer']}_E{control['expert_id']}"
        difference_boot = contrast_boot[row_index[specialist_id]] - contrast_boot[row_index[control_id]]
        low, high = ci(difference_boot)
        specialist = contrast_by_id[specialist_id]
        control_row = contrast_by_id[control_id]
        paired_records.append(
            {
                "bit_width": 4,
                "pair_id": pair["pair_id"],
                "target_domain": pair["target_domain"],
                "specialist_contrast": specialist["target_minus_mean_other_contrast"],
                "specialist_contrast_ci_low": specialist["contrast_ci_low"],
                "specialist_contrast_ci_high": specialist["contrast_ci_high"],
                "control_contrast": control_row["target_minus_mean_other_contrast"],
                "control_contrast_ci_low": control_row["contrast_ci_low"],
                "control_contrast_ci_high": control_row["contrast_ci_high"],
                "specialist_minus_control_difference": float(
                    specialist["target_minus_mean_other_contrast"]
                    - control_row["target_minus_mean_other_contrast"]
                ),
                "difference_ci_low": low,
                "difference_ci_high": high,
            }
        )
    published_paired = published_by(analysis["specialist_vs_control"], ("bit_width", "pair_id"))
    for computed in paired_records:
        published = published_paired[(4, computed["pair_id"])]
        for field in (
            "specialist_contrast",
            "specialist_contrast_ci_low",
            "specialist_contrast_ci_high",
            "control_contrast",
            "control_contrast_ci_low",
            "control_contrast_ci_high",
            "specialist_minus_control_difference",
            "difference_ci_low",
            "difference_ci_high",
        ):
            checks.close(published[field], computed[field], f"paired {computed['pair_id']} {field}")

    aggregate_records: list[dict[str, Any]] = []
    for scope, target in [("domain", domain) for domain in DOMAINS] + [("overall", "all")]:
        specialist_indices = [
            index for index, row in enumerate(rows)
            if row["role"] == "specialist" and (scope == "overall" or row["target_domain"] == target)
        ]
        control_indices = [
            index for index, row in enumerate(rows)
            if row["role"] == "control" and (scope == "overall" or row["target_domain"] == target)
        ]
        specialist_indices.sort(key=lambda index: rows[index]["pair_id"])
        control_indices.sort(key=lambda index: rows[index]["pair_id"])
        spec = np.asarray([
            contrast_by_id[rows[index]["intervention_id"]]["target_minus_mean_other_contrast"]
            for index in specialist_indices
        ])
        control = np.asarray([
            contrast_by_id[rows[index]["intervention_id"]]["target_minus_mean_other_contrast"]
            for index in control_indices
        ])
        spec_boot = contrast_boot[specialist_indices]
        control_boot = contrast_boot[control_indices]
        spec_ci = ci(spec_boot.mean(axis=0))
        control_ci = ci(control_boot.mean(axis=0))
        difference_ci = ci((spec_boot - control_boot).mean(axis=0))
        median_ci = ci(np.median(spec_boot, axis=0))
        aggregate_records.append(
            {
                "bit_width": 4,
                "scope": scope,
                "target_domain": target,
                "num_specialists": len(spec),
                "mean_specialist_contrast": float(spec.mean()),
                "mean_specialist_contrast_ci_low": spec_ci[0],
                "mean_specialist_contrast_ci_high": spec_ci[1],
                "median_specialist_contrast": float(np.median(spec)),
                "median_specialist_contrast_ci_low": median_ci[0],
                "median_specialist_contrast_ci_high": median_ci[1],
                "specialist_positive_proportion": float(np.mean(spec > 0)),
                "mean_control_contrast": float(control.mean()),
                "mean_control_contrast_ci_low": control_ci[0],
                "mean_control_contrast_ci_high": control_ci[1],
                "mean_specialist_minus_control_difference": float(np.mean(spec - control)),
                "mean_difference_ci_low": difference_ci[0],
                "mean_difference_ci_high": difference_ci[1],
                "pair_difference_positive_proportion": float(np.mean((spec - control) > 0)),
            }
        )
    published_aggregates = published_by(
        analysis["aggregate_results"], ("bit_width", "scope", "target_domain")
    )
    for computed in aggregate_records:
        published = published_aggregates[(4, computed["scope"], computed["target_domain"])]
        for field, value in computed.items():
            if field in ("bit_width", "scope", "target_domain", "num_specialists"):
                checks.require(published[field] == value, f"aggregate {computed['target_domain']} {field}")
            else:
                checks.close(published[field], value, f"aggregate {computed['target_domain']} {field}")

    with np.load(balanced_raw_path, allow_pickle=False) as data:
        mask_changes = data["per_example_loss_changes"].astype(np.float64)
        mask_ids = [str(value) for value in data["intervention_ids"]]
        mask_roles = ["specialist" if str(value) == "specialized" else str(value) for value in data["roles"]]
        mask_targets = [str(value) for value in data["target_domains"]]
        mask_layers = data["layers"].astype(int)
        mask_experts = data["expert_ids"].astype(int)
        mask_domains = [str(value) for value in data["domain_names"]]
    checks.require(mask_changes.shape == (24, 4, 100), "balanced masking NPZ shape")
    checks.require(mask_domains == list(DOMAINS), "balanced domain order")
    checks.require(np.all(np.isfinite(mask_changes)), "balanced masking contains non-finite values")
    masking_lookup: dict[tuple[str, int, int], float] = {}
    balanced_published = {
        row["intervention_id"]: row
        for row in balanced_results["balanced_analysis"]["intervention_contrasts"]
    }
    for index, intervention_id in enumerate(mask_ids):
        target_index = DOMAINS.index(mask_targets[index])
        other = [value for value in range(4) if value != target_index]
        value = float(mask_changes[index, target_index].mean() - mask_changes[index, other].mean())
        checks.close(
            balanced_published[intervention_id]["target_minus_mean_other_contrast"],
            value,
            f"balanced raw contrast {intervention_id}",
        )
        masking_lookup[(mask_roles[index], int(mask_layers[index]), int(mask_experts[index]))] = value

    masking_values = np.asarray([
        masking_lookup[(row["role"], row["layer"], row["expert_id"])] for row in rows
    ])
    quant_values = np.asarray([
        contrast_by_id[row["intervention_id"]]["target_minus_mean_other_contrast"] for row in rows
    ])
    mask_spearman_boot = np.asarray([
        safe_spearman(masking_values, contrast_boot[:, replicate])
        for replicate in range(BOOTSTRAP_REPLICATES)
    ])
    mask_kendall_boot = np.asarray([
        safe_kendall(masking_values, contrast_boot[:, replicate])
        for replicate in range(BOOTSTRAP_REPLICATES)
    ])
    sign_boot = np.asarray([
        np.mean(np.sign(masking_values) == np.sign(contrast_boot[:, replicate]))
        for replicate in range(BOOTSTRAP_REPLICATES)
    ])
    correlations: list[dict[str, Any]] = []
    mask_spearman_ci = finite_ci(mask_spearman_boot)
    mask_kendall_ci = finite_ci(mask_kendall_boot)
    sign_ci = ci(sign_boot)
    correlations.append(
        {
            "predictor": "masking_causal_contrast",
            "spearman": safe_spearman(masking_values, quant_values),
            "spearman_ci_low": mask_spearman_ci[0],
            "spearman_ci_high": mask_spearman_ci[1],
            "kendall_tau": safe_kendall(masking_values, quant_values),
            "kendall_tau_ci_low": mask_kendall_ci[0],
            "kendall_tau_ci_high": mask_kendall_ci[1],
            "sign_agreement": float(np.mean(np.sign(masking_values) == np.sign(quant_values))),
            "sign_agreement_ci_low": sign_ci[0],
            "sign_agreement_ci_high": sign_ci[1],
        }
    )
    predictor_fields = (
        ("functional_specialization", "specialization_margin"),
        ("routing_specialization", "routing_specialization_margin"),
        ("target_routing_frequency", "target_routing_frequency"),
    )
    for predictor_name, field in predictor_fields:
        predictor = np.asarray([float(row["record"][field]) for row in rows])
        values = np.asarray([
            safe_spearman(predictor, contrast_boot[:, replicate])
            for replicate in range(BOOTSTRAP_REPLICATES)
        ])
        low, high = finite_ci(values)
        correlations.append(
            {
                "predictor": predictor_name,
                "spearman": safe_spearman(predictor, quant_values),
                "spearman_ci_low": low,
                "spearman_ci_high": high,
                "kendall_tau": None,
                "kendall_tau_ci_low": None,
                "kendall_tau_ci_high": None,
                "sign_agreement": None,
                "sign_agreement_ci_low": None,
                "sign_agreement_ci_high": None,
            }
        )
    published_correlations = published_by(analysis["correlation_results"], ("bit_width", "predictor"))
    for computed in correlations:
        published = published_correlations[(4, computed["predictor"])]
        for field, value in computed.items():
            if field == "predictor":
                continue
            if value is None:
                checks.require(published[field] is None, f"correlation {computed['predictor']} {field}")
            else:
                checks.close(published[field], value, f"correlation {computed['predictor']} {field}")

    risk_correlations: list[dict[str, Any]] = []
    result_by = {(row["intervention_id"], row["domain"]): row for row in result_records}
    boot_flat = boot_domain.reshape(-1, BOOTSTRAP_REPLICATES)
    actual_flat = domain_means.reshape(-1)
    for proxy_name in ("risk_functional", "risk_routing"):
        proxy = np.asarray([
            result_by[(row["intervention_id"], domain)][proxy_name]
            for row in rows for domain in DOMAINS
        ])
        boot_values = np.asarray([
            safe_spearman(proxy, boot_flat[:, replicate])
            for replicate in range(BOOTSTRAP_REPLICATES)
        ])
        low, high = finite_ci(boot_values)
        risk_correlations.append(
            {
                "predictor": proxy_name,
                "spearman": safe_spearman(proxy, actual_flat),
                "spearman_ci_low": low,
                "spearman_ci_high": high,
            }
        )
    published_risk = published_by(analysis["risk_proxy_correlations"], ("bit_width", "predictor"))
    for computed in risk_correlations:
        published = published_risk[(4, computed["predictor"])]
        for field in ("spearman", "spearman_ci_low", "spearman_ci_high"):
            checks.close(published[field], computed[field], f"risk {computed['predictor']} {field}")

    overall = next(row for row in aggregate_records if row["scope"] == "overall")
    domains = [row for row in aggregate_records if row["scope"] == "domain"]
    positive_domains = [row["target_domain"] for row in domains if row["mean_specialist_contrast"] > 0]
    noise = max(
        float(row["max_absolute_per_token_nll_difference"])
        for row in baseline_reproduction["domains"].values()
    )
    threshold = max(1e-7, 10.0 * max(0.0, noise))
    median_absolute_delta = float(np.median(np.abs(domain_means)))
    gates = {
        "gate_a": overall["mean_specialist_contrast"] > 0
        and overall["mean_specialist_contrast_ci_low"] > 0,
        "gate_b": len(positive_domains) >= 3,
        "gate_c": overall["mean_specialist_minus_control_difference"] > 0,
        "gate_d": median_absolute_delta > threshold,
    }
    audit_decision = "GO" if all(gates.values()) else "NO_GO"
    published_gate = analysis["gates_by_bit_width"][0]
    for gate_name, value in gates.items():
        checks.require(published_gate[gate_name]["passed"] is value, f"{gate_name} mismatch")
    checks.close(published_gate["gate_d"]["observed"], median_absolute_delta, "Gate D statistic")
    checks.close(published_gate["gate_d"]["clear_noise_threshold"], threshold, "Gate D threshold")
    checks.require(analysis["stage1_decision"]["decision"] == audit_decision, "analysis decision mismatch")
    checks.require(decision_file == analysis["stage1_decision"], "stage1_decision.json mismatch")

    manifest_hashes = results["artifact_manifest"]["file_sha256"]
    for relative, expected_hash in manifest_hashes.items():
        checks.require(file_sha256(pilot / relative) == expected_hash, f"manifest hash mismatch: {relative}")
    csv_audit = {
        "quantization_pilot_results.csv": compare_csv(
            pilot / "quantization_pilot_results.csv", analysis["quantization_pilot_results"], checks
        ),
        "quantization_pilot_pairwise.csv": compare_csv(
            pilot / "quantization_pilot_pairwise.csv", analysis["quantization_pilot_pairwise"], checks
        ),
        "specialist_vs_control.csv": compare_csv(
            pilot / "specialist_vs_control.csv", analysis["specialist_vs_control"], checks
        ),
        "quantization_vs_masking.csv": compare_csv(
            pilot / "quantization_vs_masking.csv", analysis["quantization_vs_masking"], checks
        ),
        "quantization_distortion.csv": compare_csv(
            pilot / "quantization_distortion.csv", analysis["quantization_distortion"], checks
        ),
    }

    return {
        "schema_version": 1,
        "audit_method": "standalone_raw_checkpoint_recomputation_without_production_analysis_imports",
        "passed": not checks.errors,
        "decision_recomputed": audit_decision,
        "checks_performed": checks.comparisons,
        "maximum_absolute_numeric_difference": checks.maximum_absolute_numeric_difference,
        "errors": checks.errors,
        "provenance": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "runtime": results["run_config"].get("runtime_description"),
            "inference_fingerprint_recomputed": inference_hash,
            "pilot_panel_fingerprint_recomputed": panel_hash,
            "pilot_panel_file_sha256": file_sha256(pilot / "pilot_panel_preregistered.json"),
            "balanced_raw_masking_sha256": balanced_raw_hash,
            "controlled_input_sha256": input_hash_checks,
            "bootstrap_seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "derived_bootstrap_seeds": {
                domain: derived_seed(f"pilot-bootstrap-{domain}") for domain in DOMAINS
            },
        },
        "geometry": {
            "domains": list(DOMAINS),
            "examples_per_domain": EXAMPLES,
            "tokens_per_example": TOKENS_PER_EXAMPLE,
            "positions_per_domain": EXAMPLES * TOKENS_PER_EXAMPLE,
            "interventions": len(rows),
            "specialists": sum(row["role"] == "specialist" for row in rows),
            "controls": sum(row["role"] == "control" for row in rows),
            "evaluated_bits": [4],
            "three_bit_triggered": False,
            "per_example_quantization_shape": [1, 16, 4, 100],
            "balanced_masking_shape": [24, 4, 100],
        },
        "baseline_reproduction_noise_nll": noise,
        "restoration_and_checkpoint_audit": {
            "passed": not any("restoration" in error or "checkpoint" in error for error in checks.errors),
            "complete_intervention_domain_checkpoints": len(rows) * len(DOMAINS),
            "metadata_records": metadata_summary,
            "final_npz_reconstructed_bitwise_from_checkpoints": True,
        },
        "specialist_and_control_contrasts": contrast_records,
        "specialist_control_pairs": paired_records,
        "aggregate_results": aggregate_records,
        "correlations": correlations,
        "risk_proxy_correlations": risk_correlations,
        "gates": {
            **gates,
            "positive_domains": positive_domains,
            "gate_d_observed": median_absolute_delta,
            "gate_d_threshold": threshold,
        },
        "csv_consistency": csv_audit,
        "artifact_manifest_entries_verified": len(manifest_hashes),
    }


def main() -> int:
    args = parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Independent Stage-1 audit: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Decision recomputed: {report['decision_recomputed']}")
    print(f"Checks performed: {report['checks_performed']}")
    print(f"Report: {args.output.resolve()}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
