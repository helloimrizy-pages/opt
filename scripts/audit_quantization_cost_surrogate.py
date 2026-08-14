#!/usr/bin/env python3
"""Standalone Stage-2A audit; deliberately imports no production analysis code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import kendalltau, spearmanr


DOMAINS = ("general", "math", "coding", "reasoning")
BITS = (3, 4, 8, 16)
EPSILON = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit Stage-2A artifacts")
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("results/expert_quantization_pilot"),
    )
    parser.add_argument(
        "--surrogate-dir",
        type=Path,
        default=Path("results/quantization_cost_surrogate"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quantization_cost_surrogate/independent_audit.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


@dataclass
class Checks:
    performed: int = 0
    maximum_numeric_difference: float = 0.0
    errors: list[str] = field(default_factory=list)

    def check(self, condition: bool, message: str) -> None:
        self.performed += 1
        if not condition:
            self.errors.append(message)

    def close(self, observed: Any, expected: Any, message: str, atol: float = 1e-14) -> None:
        self.performed += 1
        observed_missing = observed is None or observed == ""
        expected_missing = expected is None or expected == ""
        try:
            observed_nan = math.isnan(float(observed))
        except (TypeError, ValueError):
            observed_nan = False
        try:
            expected_nan = math.isnan(float(expected))
        except (TypeError, ValueError):
            expected_nan = False
        if (observed_missing or observed_nan) and (expected_missing or expected_nan):
            return
        try:
            first = float(observed)
            second = float(expected)
        except (TypeError, ValueError):
            if observed != expected:
                self.errors.append(f"{message}: observed={observed!r}, expected={expected!r}")
            return
        if math.isnan(first) and math.isnan(second):
            return
        difference = abs(first - second)
        self.maximum_numeric_difference = max(self.maximum_numeric_difference, difference)
        if not math.isfinite(difference) or difference > atol:
            self.errors.append(
                f"{message}: observed={first:.17g}, expected={second:.17g}, "
                f"difference={difference:.3g}"
            )

    def array_equal(self, observed: np.ndarray, expected: np.ndarray, message: str) -> None:
        self.performed += 1
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            self.errors.append(message)

    def array_close(
        self, observed: np.ndarray, expected: np.ndarray, message: str, atol: float = 1e-14
    ) -> None:
        self.performed += 1
        if observed.shape != expected.shape:
            self.errors.append(f"{message}: shape mismatch")
            return
        if observed.size:
            difference = float(np.max(np.abs(observed.astype(float) - expected.astype(float))))
            self.maximum_numeric_difference = max(self.maximum_numeric_difference, difference)
        if not np.allclose(observed, expected, rtol=0, atol=atol, equal_nan=True):
            self.errors.append(message)


def audit(stage1_dir: Path, surrogate_dir: Path) -> dict[str, Any]:
    checks = Checks()
    decision = read_json(stage1_dir / "stage1_decision.json")
    stage1_audit = read_json(stage1_dir / "independent_audit.json")
    stage1_results = read_json(stage1_dir / "results.json")
    checks.check(decision.get("decision") == "GO", "Stage-1 decision is not GO")
    checks.check(decision.get("selected_bit_width") == 4, "Stage-1 selected bit is not 4")
    four_bit_gate_rows = [
        row
        for row in decision.get("gates_by_bit_width", [])
        if int(row.get("bit_width", -1)) == 4
    ]
    checks.check(
        len(four_bit_gate_rows) == 1
        and four_bit_gate_rows[0].get("all_required_gates_passed") is True,
        "Stage-1 4-bit gate record is not a valid GO",
    )
    checks.check(stage1_audit.get("passed") is True, "Stage-1 audit did not pass")
    stage1_raw_path = stage1_dir / "per_example_quantization_losses.npz"
    recorded = stage1_results["artifact_manifest"]["file_sha256"][
        "per_example_quantization_losses.npz"
    ]
    checks.check(sha256(stage1_raw_path) == recorded, "Stage-1 raw NPZ hash changed")
    with np.load(stage1_raw_path, allow_pickle=False) as data:
        stage1_ids = data["intervention_ids"].astype(str)
        stage1_targets = data["target_domains"].astype(str)
        stage1_layers = data["layers"].astype(np.int64)
        stage1_experts = data["expert_ids"].astype(np.int64)
        stage1_domains = data["domain_names"].astype(str)
        stage1_changes = data["per_example_loss_changes"].astype(np.float64)[0]
        stage1_actual = stage1_changes.mean(axis=-1)
        stage1_distortion = data["quantization_distortion"].astype(np.float64)[0]
    checks.array_equal(stage1_domains, np.asarray(DOMAINS), "Stage-1 domain order changed")
    checks.check(stage1_changes.shape == (16, 4, 100), "Stage-1 raw geometry changed")
    stage1_qdq: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted(
        (stage1_dir / "quantization" / "bit_4").glob(
            "layer_*_expert_*/quantization.metadata.json"
        )
    ):
        metadata = read_json(path)
        key = (int(metadata["layer"]), int(metadata["expert_id"]), int(metadata["bits"]))
        checks.check(key not in stage1_qdq, f"Duplicate Stage-1 QDQ metadata for {key}")
        checks.check(
            metadata.get("exact_restoration_verified") is True,
            f"Stage-1 QDQ restoration failed for {key}",
        )
        stage1_qdq[key] = metadata
    checks.check(len(stage1_qdq) == 16, "Stage-1 QDQ metadata set is incomplete")

    raw_path = surrogate_dir / "pilot_surrogate_raw.npz"
    with np.load(raw_path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    checks.array_equal(raw["intervention_ids"].astype(str), stage1_ids, "Pilot IDs differ")
    checks.array_equal(raw["target_domains"].astype(str), stage1_targets, "Targets differ")
    checks.array_equal(raw["layers"].astype(np.int64), stage1_layers, "Layers differ")
    checks.array_equal(raw["expert_ids"].astype(np.int64), stage1_experts, "Experts differ")
    checks.array_equal(raw["domain_names"].astype(str), stage1_domains, "Domains differ")
    checks.array_equal(
        raw["actual_per_example_delta_nll"], stage1_changes, "Raw ΔNLL observations differ"
    )
    checks.array_equal(raw["actual_delta_nll"], stage1_actual, "Point ΔNLL differs")
    checks.array_equal(
        raw["quantization_distortion"], stage1_distortion, "Weight distortion differs"
    )

    activation = {
        name: raw[f"activation_{name}"][0]
        for name in (
            "gated_delta_squared",
            "gated_baseline_squared",
            "ungated_delta_squared",
            "route_counts",
            "layer_energy",
            "domain_token_count",
            "aod",
            "reod",
            "apd",
            "uod",
            "unobserved",
        )
    }
    for name in (
        "gated_delta_squared",
        "gated_baseline_squared",
        "ungated_delta_squared",
        "layer_energy",
        "aod",
        "reod",
        "apd",
        "uod",
    ):
        checks.check(
            bool(np.all(np.isfinite(activation[name])) and np.all(activation[name] >= 0)),
            f"Activation raw array {name} is invalid",
        )
    recomputed_aod = activation["gated_delta_squared"] / (
        activation["layer_energy"] + EPSILON
    )
    recomputed_reod = activation["gated_delta_squared"] / (
        activation["gated_baseline_squared"] + EPSILON
    )
    recomputed_apd = activation["gated_delta_squared"] / activation[
        "domain_token_count"
    ]
    recomputed_uod = np.divide(
        activation["ungated_delta_squared"],
        activation["route_counts"],
        out=np.zeros_like(activation["ungated_delta_squared"]),
        where=activation["route_counts"] > 0,
    )
    checks.array_close(activation["aod"], recomputed_aod, "AOD raw formula mismatch")
    checks.array_close(activation["reod"], recomputed_reod, "REOD raw formula mismatch")
    checks.array_close(activation["apd"], recomputed_apd, "APD raw formula mismatch")
    checks.array_close(activation["uod"], recomputed_uod, "UOD raw formula mismatch")
    checks.array_equal(
        activation["unobserved"].astype(bool),
        activation["route_counts"] == 0,
        "Pilot unobserved flags differ from route counts",
    )
    checks.check(
        bool(np.all(activation["domain_token_count"] == 6400)),
        "Pilot APD denominators are not 6,400",
    )

    functional = raw["functional_importance"].astype(np.float64)
    routing = raw["routing_importance"].astype(np.float64)
    scores: dict[str, np.ndarray] = {
        "weight_risk_functional": functional * stage1_distortion[:, None],
        "weight_risk_routing": routing * stage1_distortion[:, None],
        "functional_importance": functional,
        "routing_importance": routing,
        "uod": recomputed_uod,
        "reod": recomputed_reod,
        "apd": recomputed_apd,
        "aod": recomputed_aod,
    }
    gradient_present = "gradient_gqs" in raw
    if gradient_present:
        signed = raw["gradient_signed_first_order_by_example"][0].astype(np.float64)
        gqs = np.mean(np.abs(signed), axis=-1)
        gqs2 = np.mean(np.square(signed), axis=-1)
        checks.array_close(raw["gradient_gqs"][0], gqs, "GQS raw formula mismatch")
        checks.array_close(raw["gradient_gqs2"][0], gqs2, "GQS2 raw formula mismatch")
        scores["gqs"] = gqs

    target_indices = np.asarray([DOMAINS.index(value) for value in stage1_targets])
    independently_analyzed = {
        name: analyze_score(values, stage1_actual, target_indices)
        for name, values in scores.items()
    }
    results = read_json(surrogate_dir / "results.json")
    aod_published = results["aod_analysis"]
    compare_analysis(checks, independently_analyzed, aod_published)
    compare_improvement_bootstrap(
        checks,
        scores["aod"],
        scores["weight_risk_functional"],
        stage1_actual,
        aod_published["weight_proxy_improvement_bootstrap"],
        "AOD",
    )
    aod_gates = evaluate_gates(independently_analyzed, "aod")
    compare_gates(checks, aod_gates, aod_published["primary_gates"], "AOD")
    aod_passed = bool(aod_gates["all_required_gates_passed"])
    gqs_passed = False
    if gradient_present:
        gqs_published = results["gradient_fallback"]["analysis"]
        compare_analysis(checks, independently_analyzed, gqs_published)
        compare_improvement_bootstrap(
            checks,
            scores["gqs"],
            scores["weight_risk_functional"],
            stage1_actual,
            gqs_published["weight_proxy_improvement_bootstrap"],
            "GQS",
        )
        gqs_gates = evaluate_gates(independently_analyzed, "gqs")
        compare_gates(checks, gqs_gates, gqs_published["primary_gates"], "GQS")
        gqs_passed = bool(gqs_gates["all_required_gates_passed"])
    expected_decision = (
        "AOD_GO"
        if aod_passed
        else ("SURROGATE_GO_GRADIENT" if gqs_passed else "SURROGATE_NO_GO")
    )
    published_decision = read_json(surrogate_dir / "surrogate_decision.json")
    checks.check(
        published_decision.get("provisional_metric_decision") == expected_decision,
        "Published provisional decision differs from independent gates",
    )
    checks.check(
        published_decision.get("decision") in (expected_decision, "SURROGATE_NO_GO"),
        "Published decision is inconsistent with independent gates",
    )

    audit_pilot_csv(checks, surrogate_dir / "pilot_surrogate_values.csv", stage1_ids, scores, stage1_actual)
    audit_comparison_csv(
        checks,
        surrogate_dir / "surrogate_comparison.csv",
        independently_analyzed,
    )
    replay_validation = read_json(surrogate_dir / "replay_validation.json")
    checks.check(replay_validation.get("passed") is True, "Replay validation did not pass")
    checks.check(
        int(replay_validation.get("validated_expert_count", 0)) >= 3,
        "Replay validation covered fewer than three experts",
    )
    replay_samples = replay_validation.get("samples", [])
    checks.check(
        len(replay_samples) == int(replay_validation.get("sample_count", -1)),
        "Replay validation sample count is inconsistent",
    )
    for row in replay_samples:
        checks.check(
            row.get("contribution_passed") is True
            and row.get("aggregate_passed") is True,
            "A saved replay-validation sample did not pass both comparisons",
        )
    qdq = results["qdq_reproduction"]
    checks.check(qdq.get("passed") is True, "Stage-1 QDQ reproduction did not pass")
    checks.check(
        qdq.get("exact_stage1_fingerprint_matches") == 16,
        "Not all 16 Stage-1 QDQ fingerprints matched",
    )
    qdq_rows = qdq.get("rows", [])
    checks.check(len(qdq_rows) == 16, "Pilot QDQ reproduction does not contain 16 rows")
    observed_qdq_keys: set[tuple[int, int, int]] = set()
    for row in qdq_rows:
        key = (int(row["layer"]), int(row["expert_id"]), int(row["bit_width"]))
        observed_qdq_keys.add(key)
        expected = stage1_qdq.get(key)
        checks.check(expected is not None, f"Pilot QDQ row is not in Stage 1: {key}")
        checks.check(
            row.get("exact_stage1_fingerprint_match") is True,
            f"QDQ fingerprint mismatch for L{row.get('layer')}/E{row.get('expert_id')}",
        )
        if expected is not None:
            checks.check(
                row.get("original_expert_fingerprint")
                == expected.get("original_expert_fingerprint"),
                f"Original QDQ fingerprint differs for {key}",
            )
            checks.check(
                row.get("quantized_expert_fingerprint")
                == expected.get("quantized_expert_fingerprint"),
                f"Quantized QDQ fingerprint differs for {key}",
            )
    checks.check(
        observed_qdq_keys == set(stage1_qdq),
        "Pilot QDQ reproduction does not cover the exact frozen Stage-1 panel",
    )

    full_matrix_audit = None
    if (surrogate_dir / "full_cost_matrix.npz").is_file():
        full_matrix_audit = audit_full_matrix(
            checks,
            surrogate_dir,
            scores,
            stage1_layers,
            stage1_experts,
            expected_decision,
        )
    required = [
        "capture_metadata.json",
        "replay_validation.json",
        "pilot_surrogate_values.csv",
        "surrogate_comparison.csv",
        "surrogate_specificity.csv",
        "within_expert_domain_rankings.csv",
        "domain_specific_correlations.csv",
        "bootstrap_results.json",
        "surrogate_decision.json",
        "results.json",
        "SUMMARY.md",
    ]
    for name in required:
        path = surrogate_dir / name
        checks.check(path.is_file() and path.stat().st_size > 0, f"Missing output {name}")
    return {
        "schema_version": 1,
        "audit_method": (
            "standalone_raw_surrogate_recomputation_without_production_analysis_imports"
        ),
        "passed": not checks.errors,
        "decision_recomputed": expected_decision,
        "checks_performed": checks.performed,
        "maximum_absolute_numeric_difference": checks.maximum_numeric_difference,
        "errors": checks.errors,
        "stage1": {
            "decision": decision.get("decision"),
            "raw_npz_sha256": sha256(stage1_raw_path),
            "observations": 64,
            "counterexamples_excluded": False,
        },
        "pilot_recomputation": {
            "surrogate_values_reconstructed": len(scores) * 64,
            "aod_formula_recomputed_from_raw_sums": True,
            "gqs_formula_recomputed_from_signed_per_example_sums": gradient_present,
            "grouped_bootstrap_recomputed": True,
            "gate_decision_recomputed": True,
        },
        "independent_metrics": independently_analyzed,
        "full_matrix_audit": full_matrix_audit,
    }


def analyze_score(
    predicted: np.ndarray, actual: np.ndarray, target_indices: np.ndarray
) -> dict[str, Any]:
    indices = bootstrap_indices(predicted.shape[0])
    predicted_specificity = specificity(predicted, target_indices)
    actual_specificity = specificity(actual, target_indices)
    overall_s = np.asarray(
        [safe_spearman(predicted[row].ravel(), actual[row].ravel()) for row in indices]
    )
    overall_k = np.asarray(
        [safe_kendall(predicted[row].ravel(), actual[row].ravel()) for row in indices]
    )
    spec_s = np.asarray(
        [safe_spearman(predicted_specificity[row], actual_specificity[row]) for row in indices]
    )
    spec_k = np.asarray(
        [safe_kendall(predicted_specificity[row], actual_specificity[row]) for row in indices]
    )
    top = np.asarray(
        [np.mean(np.argmax(predicted[row], axis=1) == np.argmax(actual[row], axis=1)) for row in indices]
    )
    sign = np.asarray(
        [
            np.mean(
                np.sign(predicted_specificity[row]) == np.sign(actual_specificity[row])
            )
            for row in indices
        ]
    )
    domain_rows = []
    for domain_index, domain in enumerate(DOMAINS):
        boot = np.asarray(
            [
                safe_spearman(predicted[row, domain_index], actual[row, domain_index])
                for row in indices
            ]
        )
        domain_rows.append(
            {
                "domain": domain,
                "spearman": safe_spearman(
                    predicted[:, domain_index], actual[:, domain_index]
                ),
                "spearman_ci_low": ci(boot)[0],
                "spearman_ci_high": ci(boot)[1],
            }
        )
    per_expert = np.asarray(
        [safe_spearman(predicted[index], actual[index]) for index in range(len(predicted))],
        dtype=np.float64,
    )
    finite_per_expert = per_expert[np.isfinite(per_expert)]
    return {
        "overall": {
            "spearman": safe_spearman(predicted.ravel(), actual.ravel()),
            "spearman_ci_low": ci(overall_s)[0],
            "spearman_ci_high": ci(overall_s)[1],
            "kendall_tau": safe_kendall(predicted.ravel(), actual.ravel()),
            "kendall_tau_ci_low": ci(overall_k)[0],
            "kendall_tau_ci_high": ci(overall_k)[1],
        },
        "specificity": {
            "spearman": safe_spearman(predicted_specificity, actual_specificity),
            "spearman_ci_low": ci(spec_s)[0],
            "spearman_ci_high": ci(spec_s)[1],
            "kendall_tau": safe_kendall(predicted_specificity, actual_specificity),
            "kendall_tau_ci_low": ci(spec_k)[0],
            "kendall_tau_ci_high": ci(spec_k)[1],
            "sign_agreement": float(
                np.mean(np.sign(predicted_specificity) == np.sign(actual_specificity))
            ),
            "sign_agreement_ci_low": ci(sign)[0],
            "sign_agreement_ci_high": ci(sign)[1],
        },
        "top_domain_accuracy": float(
            np.mean(np.argmax(predicted, axis=1) == np.argmax(actual, axis=1))
        ),
        "top_domain_accuracy_ci_low": ci(top)[0],
        "top_domain_accuracy_ci_high": ci(top)[1],
        "per_expert_spearman": per_expert,
        "mean_per_expert_spearman": (
            float(finite_per_expert.mean()) if finite_per_expert.size else None
        ),
        "median_per_expert_spearman": (
            float(np.median(finite_per_expert)) if finite_per_expert.size else None
        ),
        "finite_per_expert_count": int(finite_per_expert.size),
        "domain_correlations": domain_rows,
        "positive_domain_count": int(sum(row["spearman"] > 0 for row in domain_rows)),
    }


def evaluate_gates(all_metrics: Mapping[str, Any], primary: str) -> dict[str, Any]:
    metric = all_metrics[primary]
    baseline = all_metrics["weight_risk_functional"]
    improvement = metric["overall"]["spearman"] - baseline["overall"]["spearman"]
    gates = {
        "gate_a": {
            "passed": bool(metric["overall"]["spearman"] > 0.25 and improvement >= 0.15),
            "overall_spearman": metric["overall"]["spearman"],
            "weight_risk_functional_spearman": baseline["overall"]["spearman"],
            "improvement_over_weight_risk_functional": improvement,
        },
        "gate_b": {
            "passed": bool(metric["overall"]["spearman_ci_low"] > 0),
            "overall_spearman_ci_low": metric["overall"]["spearman_ci_low"],
        },
        "gate_c": {
            "passed": bool(metric["specificity"]["spearman"] > 0.30),
            "specificity_spearman": metric["specificity"]["spearman"],
        },
        "gate_d": {
            "passed": bool(metric["top_domain_accuracy"] > 0.40),
            "top_domain_accuracy": metric["top_domain_accuracy"],
        },
        "gate_e": {
            "passed": bool(metric["positive_domain_count"] >= 3),
            "positive_domain_count": metric["positive_domain_count"],
        },
    }
    gates["all_required_gates_passed"] = all(row["passed"] for row in gates.values())
    return gates


def compare_analysis(
    checks: Checks,
    independently_analyzed: Mapping[str, Any],
    published: Mapping[str, Any],
) -> None:
    for name, independent in independently_analyzed.items():
        if name not in published.get("surrogates", {}):
            continue
        observed = published["surrogates"][name]
        for field in (
            "spearman",
            "spearman_ci_low",
            "spearman_ci_high",
            "kendall_tau",
            "kendall_tau_ci_low",
            "kendall_tau_ci_high",
        ):
            checks.close(
                observed["overall"][field],
                independent["overall"][field],
                f"{name} overall {field}",
            )
        for field in (
            "spearman",
            "spearman_ci_low",
            "spearman_ci_high",
            "kendall_tau",
            "kendall_tau_ci_low",
            "kendall_tau_ci_high",
            "sign_agreement",
            "sign_agreement_ci_low",
            "sign_agreement_ci_high",
        ):
            checks.close(
                observed["specificity"][field],
                independent["specificity"][field],
                f"{name} specificity {field}",
            )
        ranking = observed["within_expert_domain_ranking"]
        checks.array_close(
            np.asarray(ranking["per_expert_spearman"], dtype=np.float64),
            independent["per_expert_spearman"],
            f"{name} per-expert Spearman values",
        )
        checks.close(
            ranking["mean_spearman"],
            independent["mean_per_expert_spearman"],
            f"{name} mean per-expert Spearman",
        )
        checks.close(
            ranking["median_spearman"],
            independent["median_per_expert_spearman"],
            f"{name} median per-expert Spearman",
        )
        checks.check(
            int(ranking["finite_expert_count"])
            == independent["finite_per_expert_count"],
            f"{name} finite per-expert correlation count differs",
        )
        checks.close(
            ranking["top_domain_accuracy"],
            independent["top_domain_accuracy"],
            f"{name} top-domain accuracy",
        )
        checks.close(
            ranking["top_domain_accuracy_ci_low"],
            independent["top_domain_accuracy_ci_low"],
            f"{name} top-domain CI low",
        )
        checks.close(
            ranking["top_domain_accuracy_ci_high"],
            independent["top_domain_accuracy_ci_high"],
            f"{name} top-domain CI high",
        )
        for observed_domain, independent_domain in zip(
            observed["domain_correlations"],
            independent["domain_correlations"],
            strict=True,
        ):
            checks.check(
                observed_domain["domain"] == independent_domain["domain"],
                f"{name} domain order mismatch",
            )
            for field in ("spearman", "spearman_ci_low", "spearman_ci_high"):
                checks.close(
                    observed_domain[field],
                    independent_domain[field],
                    f"{name}/{independent_domain['domain']} {field}",
                )


def compare_gates(
    checks: Checks, independent: Mapping[str, Any], published: Mapping[str, Any], label: str
) -> None:
    for gate_name in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"):
        checks.check(
            bool(independent[gate_name]["passed"])
            == bool(published[gate_name]["passed"]),
            f"{label} {gate_name} differs",
        )
    checks.check(
        bool(independent["all_required_gates_passed"])
        == bool(published["all_required_gates_passed"]),
        f"{label} all-gates decision differs",
    )


def compare_improvement_bootstrap(
    checks: Checks,
    primary: np.ndarray,
    baseline: np.ndarray,
    actual: np.ndarray,
    published: Mapping[str, Any],
    label: str,
) -> None:
    indices = bootstrap_indices(primary.shape[0])
    differences = np.asarray(
        [
            safe_spearman(primary[row].ravel(), actual[row].ravel())
            - safe_spearman(baseline[row].ravel(), actual[row].ravel())
            for row in indices
        ],
        dtype=np.float64,
    )
    interval = ci(differences)
    point = safe_spearman(primary.ravel(), actual.ravel()) - safe_spearman(
        baseline.ravel(), actual.ravel()
    )
    checks.close(
        published["point_estimate"], point, f"{label} weight-proxy improvement point"
    )
    checks.close(
        published["ci_low"], interval[0], f"{label} weight-proxy improvement CI low"
    )
    checks.close(
        published["ci_high"], interval[1], f"{label} weight-proxy improvement CI high"
    )


def audit_pilot_csv(
    checks: Checks,
    path: Path,
    intervention_ids: np.ndarray,
    scores: Mapping[str, np.ndarray],
    actual: np.ndarray,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks.check(len(rows) == 64, "Pilot surrogate CSV does not contain 64 rows")
    lookup = {str(value): index for index, value in enumerate(intervention_ids)}
    for row in rows:
        expert = lookup[row["intervention_id"]]
        domain = DOMAINS.index(row["domain"])
        checks.close(
            row["actual_delta_nll_4bit"], actual[expert, domain], "Pilot CSV actual ΔNLL"
        )
        for name, values in scores.items():
            if name in row:
                checks.close(row[name], values[expert, domain], f"Pilot CSV {name}")


def audit_comparison_csv(
    checks: Checks, path: Path, independent: Mapping[str, Any]
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        name = row["internal_name"]
        metric = independent[name]
        checks.close(row["overall_spearman"], metric["overall"]["spearman"], f"CSV {name}")
        checks.close(
            row["specificity_spearman"], metric["specificity"]["spearman"], f"CSV {name} specificity"
        )
        checks.close(
            row["top_domain_accuracy"], metric["top_domain_accuracy"], f"CSV {name} top"
        )


def audit_full_matrix(
    checks: Checks,
    root: Path,
    pilot_scores: Mapping[str, np.ndarray],
    pilot_layers: np.ndarray,
    pilot_experts: np.ndarray,
    expected_decision: str,
) -> dict[str, Any]:
    metadata = read_json(root / "full_matrix_metadata.json")
    checks.check(
        expected_decision in ("AOD_GO", "SURROGATE_GO_GRADIENT"),
        "A full cost matrix exists without an independently recomputed surrogate GO",
    )
    with np.load(root / "full_cost_matrix.npz", allow_pickle=False) as data:
        cost = data["cost"].astype(np.float64)
        layers = data["layer_indices"].astype(int)
        experts = data["expert_ids"].astype(int)
        domains = data["domain_names"].astype(str)
        bits = data["bit_widths"].astype(int)
        selected = str(data["selected_surrogate"].item())
    expected_selected = "AOD" if expected_decision == "AOD_GO" else "GQS"
    checks.check(
        selected == expected_selected,
        "Full matrix surrogate does not match the independently recomputed GO decision",
    )
    checks.check(
        metadata.get("selected_surrogate") == selected,
        "Full matrix metadata and NPZ disagree on the selected surrogate",
    )
    checks.check(cost.shape == (16, 64, 4, 4), "Full cost shape is not [16,64,4,4]")
    checks.array_equal(layers, np.arange(16), "Full matrix layer order changed")
    checks.array_equal(experts, np.arange(64), "Full matrix expert order changed")
    checks.array_equal(domains, np.asarray(DOMAINS), "Full matrix domain order changed")
    checks.array_equal(bits, np.asarray(BITS), "Full matrix bit order changed")
    checks.check(np.all(np.isfinite(cost)), "Full matrix has non-finite values")
    checks.check(np.all(cost >= 0), "Full matrix has negative values")
    checks.check(np.all(cost[..., 3] == 0), "Full matrix 16-bit cost is not zero")
    with np.load(root / "route_coverage_matrix.npz", allow_pickle=False) as data:
        route_counts = data["route_counts"].astype(np.int64)
        unobserved = data["unobserved"].astype(bool)
    checks.check(route_counts.shape == (16, 64, 4), "Route matrix shape is invalid")
    checks.array_equal(unobserved, route_counts == 0, "Full unobserved flags are invalid")
    checks.check(
        np.all(route_counts.sum(axis=1) == 6400 * 8),
        "Full route counts do not sum to 6,400×8 per layer/domain",
    )
    with np.load(root / "memory_matrix.npz", allow_pickle=False) as data:
        memory = {key: data[key] for key in data.files}
    weight_count = 6_291_456
    groups = 49_152
    expected_bytes = []
    expected_effective = []
    for bit in BITS:
        if bit == 16:
            projected = weight_count * 2
        else:
            projected = math.ceil(weight_count * bit / 8) + groups * 2
        expected_bytes.append(projected)
        expected_effective.append(projected * 8.0 / weight_count)
    for bit_index, bit in enumerate(BITS):
        checks.check(
            np.all(memory["projected_bytes"][..., bit_index] == expected_bytes[bit_index]),
            f"Memory bytes differ at {bit}-bit",
        )
        checks.check(
            np.allclose(
                memory["effective_bits_per_weight"][..., bit_index],
                expected_effective[bit_index],
                rtol=0,
                atol=0,
            ),
            f"Effective bits/weight differ at {bit}-bit",
        )
        expected_groups = 0 if bit == 16 else groups
        checks.check(
            np.all(memory["number_of_groups"][..., bit_index] == expected_groups),
            f"Memory group count differs at {bit}-bit",
        )
        checks.check(
            np.all(memory["other_required_metadata_bits"][..., bit_index] == 0),
            f"Unexpected per-expert metadata bits at {bit}-bit",
        )
    pilot_key = "aod" if selected == "AOD" else "gqs"
    pilot = pilot_scores[pilot_key]
    extracted = np.empty_like(pilot)
    for index, (layer, expert) in enumerate(zip(pilot_layers, pilot_experts, strict=True)):
        extracted[index] = cost[int(layer), int(expert), :, BITS.index(4)]
    checks.array_close(extracted, pilot, "Full matrix does not reproduce pilot values", atol=1e-12)
    reproduction = metadata.get("pilot_reproduction", {})
    checks.check(reproduction.get("passed") is True, "Published pilot reproduction failed")
    for name, expected_hash in metadata.get("file_sha256", {}).items():
        path = root / name
        checks.check(path.is_file(), f"Full matrix manifest file missing: {name}")
        if path.is_file():
            checks.check(sha256(path) == expected_hash, f"Full matrix hash mismatch: {name}")
    return {
        "passed": True,
        "shape": list(cost.shape),
        "selected_surrogate": selected,
        "bit_widths": bits.tolist(),
        "route_coverage_shape": list(route_counts.shape),
        "unobserved_count": int(unobserved.sum()),
        "pilot_maximum_absolute_difference": float(np.max(np.abs(extracted - pilot))),
        "memory_accounting_recomputed": True,
        "file_hashes_verified": len(metadata.get("file_sha256", {})),
    }


def bootstrap_indices(num_experts: int) -> np.ndarray:
    digest = hashlib.sha256(
        b"olmoe-stage2a-grouped-bootstrap-v1\x0042"
    ).hexdigest()
    seed = int(digest[:16], 16) % (2**63 - 1)
    return np.random.default_rng(seed).integers(0, num_experts, size=(1000, num_experts))


def specificity(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    selected = values[np.arange(values.shape[0]), targets]
    return selected - (values.sum(axis=1) - selected) / 3.0


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.all(first == first.flat[0]) or np.all(second == second.flat[0]):
        return math.nan
    return float(spearmanr(first, second).statistic)


def safe_kendall(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.all(first == first.flat[0]) or np.all(second == second.flat[0]):
        return math.nan
    return float(kendalltau(first, second).statistic)


def ci(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    try:
        result = audit(args.stage1_dir.resolve(), args.surrogate_dir.resolve())
    except BaseException as exc:
        result = {
            "schema_version": 1,
            "audit_method": (
                "standalone_raw_surrogate_recomputation_without_production_analysis_imports"
            ),
            "passed": False,
            "decision_recomputed": None,
            "checks_performed": 0,
            "maximum_absolute_numeric_difference": None,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    atomic_write(args.output.resolve(), result)
    print(json.dumps(json_safe(result), indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
