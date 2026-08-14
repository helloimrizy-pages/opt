from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .balanced import BALANCED_DOMAINS, file_sha256
from .io_utils import (
    atomic_save_npz,
    atomic_write_json,
    read_json,
    write_csv,
)
from .quantization_pilot import (
    pilot_intervention_panel,
    validate_pilot_preregistration,
)


SURROGATE_VALIDATION_SCHEMA_VERSION = 1
SURROGATE_ANALYSIS_VERSION = "fixed_activation_surrogates_grouped_expert_bootstrap_v1"
DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_SEED = 42

SURROGATE_LABELS = {
    "weight_risk_functional": "WeightRiskFunctional",
    "weight_risk_routing": "WeightRiskRouting",
    "functional_importance": "Functional specialization alone",
    "routing_importance": "Routing specialization alone",
    "uod": "UOD",
    "reod": "REOD",
    "apd": "APD",
    "aod": "AOD",
    "gqs": "GQS",
    "gqs2": "GQS2",
}


@dataclass(frozen=True)
class PilotValidationData:
    intervention_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]
    roles: tuple[str, ...]
    target_domains: tuple[str, ...]
    layers: np.ndarray
    expert_ids: np.ndarray
    domains: tuple[str, ...]
    actual_delta_nll: np.ndarray
    per_example_delta_nll: np.ndarray
    functional_importance: np.ndarray
    routing_importance: np.ndarray
    quantization_distortion: np.ndarray
    weight_risk_functional: np.ndarray
    weight_risk_routing: np.ndarray
    stage1_metadata: dict[str, Any]

    @property
    def num_experts(self) -> int:
        return len(self.intervention_ids)

    @property
    def num_domains(self) -> int:
        return len(self.domains)

    @property
    def target_domain_indices(self) -> np.ndarray:
        return np.asarray(
            [self.domains.index(domain) for domain in self.target_domains],
            dtype=np.int64,
        )

    def validate(self) -> None:
        expected = (self.num_experts, self.num_domains)
        if self.num_experts != 16 or self.num_domains != 4:
            raise RuntimeError("Stage-2 validation requires exactly 16 experts × 4 domains")
        for name, values in (
            ("actual_delta_nll", self.actual_delta_nll),
            ("functional_importance", self.functional_importance),
            ("routing_importance", self.routing_importance),
            ("weight_risk_functional", self.weight_risk_functional),
            ("weight_risk_routing", self.weight_risk_routing),
        ):
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Pilot validation array {name} is invalid")
        if self.per_example_delta_nll.shape[:2] != expected or self.per_example_delta_nll.shape[
            2
        ] != 100:
            raise RuntimeError("Stage-1 per-example loss array has unexpected geometry")
        if not np.all(np.isfinite(self.per_example_delta_nll)):
            raise RuntimeError("Stage-1 per-example loss changes contain non-finite values")
        if self.quantization_distortion.shape != (self.num_experts,) or np.any(
            self.quantization_distortion < 0
        ):
            raise RuntimeError("Stage-1 weight distortion array is invalid")
        if tuple(self.domains) != tuple(BALANCED_DOMAINS):
            raise RuntimeError("Stage-1 domain ordering changed")
        if not np.allclose(
            self.actual_delta_nll,
            self.per_example_delta_nll.mean(axis=-1),
            rtol=0,
            atol=1e-15,
        ):
            raise RuntimeError("Stage-1 point effects do not equal raw per-example means")


def load_stage1_validation_data(stage1_dir: Path) -> PilotValidationData:
    """Load all 64 immutable observations from authoritative Stage-1 artifacts."""

    decision_path = stage1_dir / "stage1_decision.json"
    results_path = stage1_dir / "results.json"
    panel_path = stage1_dir / "pilot_panel_preregistered.json"
    raw_path = stage1_dir / "per_example_quantization_losses.npz"
    audit_path = stage1_dir / "independent_audit.json"
    for path in (decision_path, results_path, panel_path, raw_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage-1 artifact is missing: {path}")
    decision = read_json(decision_path)
    four_bit_gate_rows = [
        row
        for row in decision.get("gates_by_bit_width", [])
        if int(row.get("bit_width", -1)) == 4
    ]
    if (
        decision.get("decision") != "GO"
        or decision.get("selected_bit_width") != 4
        or len(four_bit_gate_rows) != 1
        or four_bit_gate_rows[0].get("all_required_gates_passed") is not True
    ):
        raise RuntimeError("Stage 1 does not contain a valid audited 4-bit GO decision")
    audit = read_json(audit_path)
    if audit.get("passed") is not True or audit.get("decision_recomputed") != "GO":
        raise RuntimeError("Stage-1 independent audit did not reproduce GO")
    results = read_json(results_path)
    if results.get("artifact_manifest", {}).get("output_audit_passed") is not True:
        raise RuntimeError("Stage-1 output manifest did not pass")
    recorded_hash = results["artifact_manifest"].get("file_sha256", {}).get(
        "per_example_quantization_losses.npz"
    )
    observed_hash = file_sha256(raw_path)
    if recorded_hash != observed_hash:
        raise RuntimeError("Stage-1 per-example loss artifact hash changed")
    panel_payload = read_json(panel_path)
    validate_pilot_preregistration(panel_payload)
    panel = pilot_intervention_panel(panel_payload)

    with np.load(raw_path, allow_pickle=False) as data:
        required = {
            "bit_widths",
            "intervention_ids",
            "pair_ids",
            "roles",
            "target_domains",
            "layers",
            "expert_ids",
            "domain_names",
            "baseline_per_example_nll",
            "quantized_per_example_nll",
            "per_example_loss_changes",
            "token_counts",
            "quantization_distortion",
        }
        if set(data.files) != required:
            raise RuntimeError("Stage-1 authoritative NPZ fields changed")
        bit_widths = data["bit_widths"].astype(int)
        if bit_widths.tolist() != [4]:
            raise RuntimeError("Stage-1 validation NPZ is not the frozen 4-bit run")
        intervention_ids = tuple(str(value) for value in data["intervention_ids"])
        pair_ids = tuple(str(value) for value in data["pair_ids"])
        roles = tuple(str(value) for value in data["roles"])
        targets = tuple(str(value) for value in data["target_domains"])
        layers = data["layers"].astype(np.int64)
        expert_ids = data["expert_ids"].astype(np.int64)
        domains = tuple(str(value) for value in data["domain_names"])
        baseline = data["baseline_per_example_nll"].astype(np.float64)
        quantized = data["quantized_per_example_nll"].astype(np.float64)[0]
        changes = data["per_example_loss_changes"].astype(np.float64)[0]
        token_counts = data["token_counts"].astype(np.uint32)
        distortion = data["quantization_distortion"].astype(np.float64)[0]
    if quantized.shape != (16, 4, 100) or baseline.shape != (4, 100):
        raise RuntimeError("Stage-1 loss arrays have unexpected dimensions")
    if not np.array_equal(quantized - baseline[None, :, :], changes):
        raise RuntimeError("Stage-1 saved loss-change identity failed")
    if not np.all(token_counts == 64):
        raise RuntimeError("Stage-1 measured-position budget changed")
    expected_identity = [
        (
            row["intervention_id"],
            row["pair_id"],
            row["role"],
            row["target_domain"],
            int(row["layer"]),
            int(row["expert_id"]),
        )
        for row in panel
    ]
    observed_identity = list(
        zip(
            intervention_ids,
            pair_ids,
            roles,
            targets,
            layers.tolist(),
            expert_ids.tolist(),
            strict=True,
        )
    )
    if observed_identity != expected_identity:
        raise RuntimeError("Stage-1 panel identity/order differs from frozen preregistration")

    functional = np.asarray(
        [
            [
                float(row["baseline_record"]["normalized_contribution_by_domain"][domain])
                for domain in domains
            ]
            for row in panel
        ],
        dtype=np.float64,
    )
    routing = np.asarray(
        [
            [
                float(row["baseline_record"]["routing_frequency_by_domain"][domain])
                for domain in domains
            ]
            for row in panel
        ],
        dtype=np.float64,
    )
    risk_functional = functional * distortion[:, None]
    risk_routing = routing * distortion[:, None]
    actual = changes.mean(axis=-1)
    stage1_analysis = results["quantization_analysis"]
    published_rows = stage1_analysis["quantization_pilot_results"]
    published = {
        (row["intervention_id"], row["domain"]): row for row in published_rows
    }
    for expert_index, intervention_id in enumerate(intervention_ids):
        for domain_index, domain in enumerate(domains):
            row = published[(intervention_id, domain)]
            checks = {
                "delta_nll": actual[expert_index, domain_index],
                "functional_importance": functional[expert_index, domain_index],
                "routing_frequency": routing[expert_index, domain_index],
                "risk_functional": risk_functional[expert_index, domain_index],
                "risk_routing": risk_routing[expert_index, domain_index],
            }
            for key, expected in checks.items():
                if not math.isclose(
                    float(row[key]), float(expected), rel_tol=0, abs_tol=1e-15
                ):
                    raise RuntimeError(
                        f"Stage-1 published {key} failed raw reconstruction for "
                        f"{intervention_id}/{domain}"
                    )
    observed_functional = _safe_spearman(risk_functional.ravel(), actual.ravel())
    observed_routing = _safe_spearman(risk_routing.ravel(), actual.ravel())
    published_risk = {
        row["predictor"]: row
        for row in stage1_analysis["risk_proxy_correlations"]
    }
    if not math.isclose(
        observed_functional,
        float(published_risk["risk_functional"]["spearman"]),
        rel_tol=0,
        abs_tol=1e-15,
    ) or not math.isclose(
        observed_routing,
        float(published_risk["risk_routing"]["spearman"]),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Stage-1 failed weight-proxy correlations were not reproduced")
    value = PilotValidationData(
        intervention_ids=intervention_ids,
        pair_ids=pair_ids,
        roles=roles,
        target_domains=targets,
        layers=layers,
        expert_ids=expert_ids,
        domains=domains,
        actual_delta_nll=actual,
        per_example_delta_nll=changes,
        functional_importance=functional,
        routing_importance=routing,
        quantization_distortion=distortion,
        weight_risk_functional=risk_functional,
        weight_risk_routing=risk_routing,
        stage1_metadata={
            "decision_sha256": file_sha256(decision_path),
            "results_sha256": file_sha256(results_path),
            "panel_sha256": file_sha256(panel_path),
            "per_example_npz_sha256": observed_hash,
            "independent_audit_sha256": file_sha256(audit_path),
            "pilot_panel_fingerprint": panel_payload["pilot_panel_fingerprint"],
            "source_input_fingerprint": results["run_config"][
                "source_input_fingerprint"
            ],
            "existing_weight_risk_functional_spearman": observed_functional,
            "existing_weight_risk_routing_spearman": observed_routing,
            "all_64_observations_retained": True,
        },
    )
    value.validate()
    return value


def load_stage1_qdq_fingerprints(
    stage1_dir: Path,
) -> dict[tuple[int, int, int], dict[str, str]]:
    output: dict[tuple[int, int, int], dict[str, str]] = {}
    for path in sorted(
        (stage1_dir / "quantization" / "bit_4").glob(
            "layer_*_expert_*/quantization.metadata.json"
        )
    ):
        metadata = read_json(path)
        key = (
            int(metadata["layer"]),
            int(metadata["expert_id"]),
            int(metadata["bits"]),
        )
        if metadata.get("exact_restoration_verified") is not True:
            raise RuntimeError(f"Stage-1 QDQ restoration failed in {path}")
        output[key] = {
            "original": str(metadata["original_expert_fingerprint"]),
            "quantized": str(metadata["quantized_expert_fingerprint"]),
        }
    if len(output) != 16 or {key[2] for key in output} != {4}:
        raise RuntimeError("Stage-1 expert QDQ fingerprint set is incomplete")
    return output


def grouped_bootstrap_indices(
    num_experts: int,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    if num_experts < 2 or replicates < 1:
        raise ValueError("Grouped bootstrap requires experts and replicates")
    digest = hashlib.sha256(
        f"olmoe-stage2a-grouped-bootstrap-v1\0{seed}".encode("utf-8")
    ).hexdigest()
    derived = int(digest[:16], 16) % (2**63 - 1)
    return np.random.default_rng(derived).integers(
        0, num_experts, size=(replicates, num_experts)
    )


def specificity_contrast(
    values: np.ndarray, target_domain_indices: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    targets = np.asarray(target_domain_indices, dtype=np.int64)
    if matrix.ndim != 2 or targets.shape != (matrix.shape[0],):
        raise ValueError("Specificity inputs are misaligned")
    if np.any(targets < 0) or np.any(targets >= matrix.shape[1]):
        raise ValueError("Specificity target-domain index is invalid")
    target_values = matrix[np.arange(matrix.shape[0]), targets]
    other_mean = (matrix.sum(axis=1) - target_values) / float(matrix.shape[1] - 1)
    return target_values - other_mean


def top_domain_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    first = np.asarray(predicted, dtype=np.float64)
    second = np.asarray(actual, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Top-domain arrays must have matching [expert, domain] shape")
    return float(np.mean(np.argmax(first, axis=1) == np.argmax(second, axis=1)))


def analyze_fixed_surrogates(
    data: PilotValidationData,
    surrogate_scores: Mapping[str, np.ndarray],
    *,
    primary_name: str,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    data.validate()
    if bootstrap_replicates != 1000 or seed != 42:
        raise ValueError("Stage 2A requires exactly 1,000 grouped replicates and seed 42")
    scores: dict[str, np.ndarray] = {
        "weight_risk_functional": data.weight_risk_functional,
        "weight_risk_routing": data.weight_risk_routing,
        "functional_importance": data.functional_importance,
        "routing_importance": data.routing_importance,
    }
    for name, values in surrogate_scores.items():
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != data.actual_delta_nll.shape:
            raise ValueError(
                f"Surrogate {name} has shape {matrix.shape}, expected "
                f"{data.actual_delta_nll.shape}"
            )
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError(f"Surrogate {name} contains invalid values")
        scores[name] = matrix
    if primary_name not in scores:
        raise ValueError(f"Primary surrogate {primary_name!r} is unavailable")
    indices = grouped_bootstrap_indices(
        data.num_experts, replicates=bootstrap_replicates, seed=seed
    )
    target_indices = data.target_domain_indices
    actual_specificity = specificity_contrast(data.actual_delta_nll, target_indices)
    metric_results: dict[str, Any] = {}
    bootstrap_values: dict[str, dict[str, np.ndarray]] = {}
    for name, predicted in scores.items():
        predicted_specificity = specificity_contrast(predicted, target_indices)
        overall_spearman_boot = np.empty(bootstrap_replicates, dtype=np.float64)
        overall_kendall_boot = np.empty(bootstrap_replicates, dtype=np.float64)
        specificity_spearman_boot = np.empty(bootstrap_replicates, dtype=np.float64)
        specificity_kendall_boot = np.empty(bootstrap_replicates, dtype=np.float64)
        sign_boot = np.empty(bootstrap_replicates, dtype=np.float64)
        top_boot = np.empty(bootstrap_replicates, dtype=np.float64)
        domain_boot = np.empty(
            (data.num_domains, bootstrap_replicates), dtype=np.float64
        )
        for replicate, sampled in enumerate(indices):
            overall_spearman_boot[replicate] = _safe_spearman(
                predicted[sampled].reshape(-1),
                data.actual_delta_nll[sampled].reshape(-1),
            )
            overall_kendall_boot[replicate] = _safe_kendall(
                predicted[sampled].reshape(-1),
                data.actual_delta_nll[sampled].reshape(-1),
            )
            specificity_spearman_boot[replicate] = _safe_spearman(
                predicted_specificity[sampled], actual_specificity[sampled]
            )
            specificity_kendall_boot[replicate] = _safe_kendall(
                predicted_specificity[sampled], actual_specificity[sampled]
            )
            sign_boot[replicate] = np.mean(
                np.sign(predicted_specificity[sampled])
                == np.sign(actual_specificity[sampled])
            )
            top_boot[replicate] = top_domain_accuracy(
                predicted[sampled], data.actual_delta_nll[sampled]
            )
            for domain_index in range(data.num_domains):
                domain_boot[domain_index, replicate] = _safe_spearman(
                    predicted[sampled, domain_index],
                    data.actual_delta_nll[sampled, domain_index],
                )

        per_expert = np.asarray(
            [
                _safe_spearman(predicted[index], data.actual_delta_nll[index])
                for index in range(data.num_experts)
            ],
            dtype=np.float64,
        )
        predicted_top = np.argmax(predicted, axis=1)
        actual_top = np.argmax(data.actual_delta_nll, axis=1)
        domain_rows = []
        for domain_index, domain in enumerate(data.domains):
            domain_rows.append(
                {
                    "domain": domain,
                    "spearman": _safe_spearman(
                        predicted[:, domain_index],
                        data.actual_delta_nll[:, domain_index],
                    ),
                    "spearman_ci_low": _finite_ci(domain_boot[domain_index])[0],
                    "spearman_ci_high": _finite_ci(domain_boot[domain_index])[1],
                }
            )
        overall_ci = _finite_ci(overall_spearman_boot)
        overall_kendall_ci = _finite_ci(overall_kendall_boot)
        specificity_ci = _finite_ci(specificity_spearman_boot)
        specificity_kendall_ci = _finite_ci(specificity_kendall_boot)
        sign_ci = _finite_ci(sign_boot)
        top_ci = _finite_ci(top_boot)
        finite_per_expert = per_expert[np.isfinite(per_expert)]
        metric_results[name] = {
            "surrogate": name,
            "label": SURROGATE_LABELS.get(name, name),
            "overall": {
                "observations": data.num_experts * data.num_domains,
                "spearman": _safe_spearman(
                    predicted.reshape(-1), data.actual_delta_nll.reshape(-1)
                ),
                "spearman_ci_low": overall_ci[0],
                "spearman_ci_high": overall_ci[1],
                "kendall_tau": _safe_kendall(
                    predicted.reshape(-1), data.actual_delta_nll.reshape(-1)
                ),
                "kendall_tau_ci_low": overall_kendall_ci[0],
                "kendall_tau_ci_high": overall_kendall_ci[1],
                "bootstrap_unit": "expert_with_four_domains_grouped",
            },
            "specificity": {
                "experts": data.num_experts,
                "definition": "target minus arithmetic mean of three other domains",
                "spearman": _safe_spearman(
                    predicted_specificity, actual_specificity
                ),
                "spearman_ci_low": specificity_ci[0],
                "spearman_ci_high": specificity_ci[1],
                "kendall_tau": _safe_kendall(
                    predicted_specificity, actual_specificity
                ),
                "kendall_tau_ci_low": specificity_kendall_ci[0],
                "kendall_tau_ci_high": specificity_kendall_ci[1],
                "sign_agreement": float(
                    np.mean(
                        np.sign(predicted_specificity)
                        == np.sign(actual_specificity)
                    )
                ),
                "sign_agreement_ci_low": sign_ci[0],
                "sign_agreement_ci_high": sign_ci[1],
            },
            "within_expert_domain_ranking": {
                "per_expert_spearman": per_expert.tolist(),
                "mean_spearman": (
                    float(finite_per_expert.mean()) if finite_per_expert.size else None
                ),
                "median_spearman": (
                    float(np.median(finite_per_expert))
                    if finite_per_expert.size
                    else None
                ),
                "finite_expert_count": int(finite_per_expert.size),
                "top_domain_accuracy": float(np.mean(predicted_top == actual_top)),
                "top_domain_accuracy_ci_low": top_ci[0],
                "top_domain_accuracy_ci_high": top_ci[1],
                "stable_argmax_domain_order": list(data.domains),
                "predicted_top_tie_count": int(
                    sum(
                        np.count_nonzero(row == row.max()) > 1 for row in predicted
                    )
                ),
            },
            "domain_correlations": domain_rows,
            "positive_domain_correlation_count": int(
                sum(
                    row["spearman"] is not None
                    and np.isfinite(row["spearman"])
                    and row["spearman"] > 0
                    for row in domain_rows
                )
            ),
            "formula_tuned_on_pilot_outcomes": False,
        }
        bootstrap_values[name] = {
            "overall_spearman": overall_spearman_boot,
            "overall_kendall": overall_kendall_boot,
            "specificity_spearman": specificity_spearman_boot,
            "specificity_kendall": specificity_kendall_boot,
            "sign_agreement": sign_boot,
            "top_domain_accuracy": top_boot,
            "domain_spearman": domain_boot,
        }

    primary = metric_results[primary_name]
    baseline = metric_results["weight_risk_functional"]
    improvement_boot = (
        bootstrap_values[primary_name]["overall_spearman"]
        - bootstrap_values["weight_risk_functional"]["overall_spearman"]
    )
    improvement_ci = _finite_ci(improvement_boot)
    improvement = float(
        primary["overall"]["spearman"] - baseline["overall"]["spearman"]
    )
    gates = {
        "gate_a": {
            "passed": bool(
                _strictly_greater(primary["overall"]["spearman"], 0.25)
                and _at_least(improvement, 0.15)
            ),
            "overall_spearman": primary["overall"]["spearman"],
            "required_overall_spearman_strictly_greater_than": 0.25,
            "weight_risk_functional_spearman": baseline["overall"]["spearman"],
            "improvement_over_weight_risk_functional": improvement,
            "required_improvement_at_least": 0.15,
            "improvement_bootstrap_ci_low": improvement_ci[0],
            "improvement_bootstrap_ci_high": improvement_ci[1],
        },
        "gate_b": {
            "passed": _strictly_greater(
                primary["overall"]["spearman_ci_low"], 0.0
            ),
            "overall_spearman_ci_low": primary["overall"]["spearman_ci_low"],
            "required_ci_lower_bound_strictly_greater_than": 0.0,
        },
        "gate_c": {
            "passed": _strictly_greater(
                primary["specificity"]["spearman"], 0.30
            ),
            "specificity_spearman": primary["specificity"]["spearman"],
            "specificity_spearman_ci_low": primary["specificity"][
                "spearman_ci_low"
            ],
            "specificity_spearman_ci_high": primary["specificity"][
                "spearman_ci_high"
            ],
            "required_strictly_greater_than": 0.30,
            "ci_excludes_zero_preferred_not_required": bool(
                _strictly_greater(
                    primary["specificity"]["spearman_ci_low"], 0.0
                )
            ),
        },
        "gate_d": {
            "passed": bool(
                primary["within_expert_domain_ranking"]["top_domain_accuracy"]
                > 0.40
            ),
            "top_domain_accuracy": primary["within_expert_domain_ranking"][
                "top_domain_accuracy"
            ],
            "required_strictly_greater_than": 0.40,
            "random_chance": 0.25,
        },
        "gate_e": {
            "passed": bool(primary["positive_domain_correlation_count"] >= 3),
            "positive_domain_count": primary[
                "positive_domain_correlation_count"
            ],
            "required_positive_domains": 3,
            "domain_spearman": {
                row["domain"]: row["spearman"]
                for row in primary["domain_correlations"]
            },
        },
    }
    all_passed = all(bool(value["passed"]) for value in gates.values())
    gates["all_required_gates_passed"] = all_passed

    comparison = []
    ordered_names = [
        "weight_risk_functional",
        "weight_risk_routing",
        "functional_importance",
        "routing_importance",
        "uod",
        "reod",
        "apd",
        "aod",
    ]
    if "gqs" in metric_results:
        ordered_names.append("gqs")
    for name in ordered_names:
        if name not in metric_results:
            continue
        result = metric_results[name]
        comparison.append(
            {
                "surrogate": result["label"],
                "internal_name": name,
                "overall_spearman": result["overall"]["spearman"],
                "overall_spearman_ci_low": result["overall"]["spearman_ci_low"],
                "overall_spearman_ci_high": result["overall"]["spearman_ci_high"],
                "specificity_spearman": result["specificity"]["spearman"],
                "specificity_spearman_ci_low": result["specificity"][
                    "spearman_ci_low"
                ],
                "specificity_spearman_ci_high": result["specificity"][
                    "spearman_ci_high"
                ],
                "top_domain_accuracy": result[
                    "within_expert_domain_ranking"
                ]["top_domain_accuracy"],
                "notes": _surrogate_note(name, primary_name),
            }
        )
    return {
        "schema_version": SURROGATE_VALIDATION_SCHEMA_VERSION,
        "analysis_version": SURROGATE_ANALYSIS_VERSION,
        "primary_surrogate": primary_name,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "expert",
            "grouped_domains_per_expert": True,
            "percentile_interval": [0.025, 0.975],
        },
        "observations": {
            "experts": data.num_experts,
            "domains": data.num_domains,
            "total": data.num_experts * data.num_domains,
            "all_stage1_observations_retained": True,
        },
        "surrogates": metric_results,
        "comparison": comparison,
        "primary_gates": gates,
        "primary_passed": all_passed,
        "weight_proxy_improvement_bootstrap": {
            "point_estimate": improvement,
            "ci_low": improvement_ci[0],
            "ci_high": improvement_ci[1],
        },
        "actual_specificity": actual_specificity.tolist(),
        "bootstrap_summaries": {
            name: {
                metric: {
                    "finite_replicates": int(np.isfinite(values).sum()),
                    "ci_low": _finite_ci(values)[0],
                    "ci_high": _finite_ci(values)[1],
                }
                for metric, values in bootstrap.items()
                if values.ndim == 1
            }
            for name, bootstrap in bootstrap_values.items()
        },
    }


def decision_from_analyses(
    aod_analysis: Mapping[str, Any],
    gqs_analysis: Mapping[str, Any] | None,
    *,
    audit_passed: bool | None = None,
) -> dict[str, Any]:
    if aod_analysis.get("primary_surrogate") != "aod":
        raise ValueError("AOD analysis must be evaluated first")
    aod_passed = bool(aod_analysis.get("primary_passed"))
    gqs_passed = bool(gqs_analysis and gqs_analysis.get("primary_passed"))
    if aod_passed:
        decision = "AOD_GO"
        selected = "AOD"
        fallback_triggered = False
    elif gqs_passed:
        decision = "SURROGATE_GO_GRADIENT"
        selected = "GQS"
        fallback_triggered = True
    elif gqs_analysis is None:
        raise RuntimeError("AOD_NO_GO requires the preregistered GQS fallback")
    else:
        decision = "SURROGATE_NO_GO"
        selected = None
        fallback_triggered = True
    provisional = decision
    if audit_passed is False and decision != "SURROGATE_NO_GO":
        decision = "SURROGATE_NO_GO"
        selected = None
    return {
        "decision": decision,
        "provisional_metric_decision": provisional,
        "selected_surrogate": selected,
        "aod_gates": aod_analysis["primary_gates"],
        "aod_passed": aod_passed,
        "gradient_fallback_triggered": fallback_triggered,
        "gqs_gates": gqs_analysis["primary_gates"] if gqs_analysis else None,
        "gqs_passed": gqs_passed if gqs_analysis else None,
        "independent_audit_passed": audit_passed,
        "full_cost_matrix_authorized": bool(
            decision in ("AOD_GO", "SURROGATE_GO_GRADIENT")
            and audit_passed is True
        ),
        "mixed_precision_optimizer_implemented": False,
        "rationale": (
            "AOD passed every preregistered Stage-2A gate."
            if decision == "AOD_GO"
            else (
                "AOD failed at least one gate; the predefined primary GQS fallback "
                "passed every gate."
                if decision == "SURROGATE_GO_GRADIENT"
                else (
                    "Independent audit failure blocks GO."
                    if audit_passed is False and provisional != "SURROGATE_NO_GO"
                    else "Neither AOD nor the predefined GQS fallback passed every gate."
                )
            )
        ),
    }


def build_pilot_output_rows(
    data: PilotValidationData, scores: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for expert_index, intervention_id in enumerate(data.intervention_ids):
        for domain_index, domain in enumerate(data.domains):
            row: dict[str, Any] = {
                "intervention_id": intervention_id,
                "pair_id": data.pair_ids[expert_index],
                "role": data.roles[expert_index],
                "target_domain": data.target_domains[expert_index],
                "layer": int(data.layers[expert_index]),
                "expert_id": int(data.expert_ids[expert_index]),
                "domain": domain,
                "is_target_domain": domain == data.target_domains[expert_index],
                "actual_delta_nll_4bit": data.actual_delta_nll[
                    expert_index, domain_index
                ],
            }
            for name, values in scores.items():
                row[name] = float(values[expert_index, domain_index])
            rows.append(row)
    return rows


def write_validation_tables(
    output_dir: Path,
    data: PilotValidationData,
    scores: Mapping[str, np.ndarray],
    analysis: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_rows = build_pilot_output_rows(data, scores)
    pilot_fields = [
        "intervention_id",
        "pair_id",
        "role",
        "target_domain",
        "layer",
        "expert_id",
        "domain",
        "is_target_domain",
        "actual_delta_nll_4bit",
        *scores.keys(),
    ]
    write_csv(output_dir / "pilot_surrogate_values.csv", pilot_rows, pilot_fields)
    comparison_fields = [
        "surrogate",
        "internal_name",
        "overall_spearman",
        "overall_spearman_ci_low",
        "overall_spearman_ci_high",
        "specificity_spearman",
        "specificity_spearman_ci_low",
        "specificity_spearman_ci_high",
        "top_domain_accuracy",
        "notes",
    ]
    write_csv(
        output_dir / "surrogate_comparison.csv",
        analysis["comparison"],
        comparison_fields,
    )
    specificity_rows = []
    ranking_rows = []
    target_indices = data.target_domain_indices
    actual_specificity = specificity_contrast(data.actual_delta_nll, target_indices)
    actual_ranks = np.argsort(
        np.argsort(-data.actual_delta_nll, axis=1, kind="stable"),
        axis=1,
        kind="stable",
    ) + 1
    for name, predicted in scores.items():
        if name not in analysis["surrogates"]:
            continue
        predicted_specificity = specificity_contrast(predicted, target_indices)
        predicted_ranks = np.argsort(
            np.argsort(-predicted, axis=1, kind="stable"),
            axis=1,
            kind="stable",
        ) + 1
        per_expert = analysis["surrogates"][name][
            "within_expert_domain_ranking"
        ]["per_expert_spearman"]
        for expert_index, intervention_id in enumerate(data.intervention_ids):
            specificity_rows.append(
                {
                    "surrogate": SURROGATE_LABELS.get(name, name),
                    "internal_name": name,
                    "intervention_id": intervention_id,
                    "target_domain": data.target_domains[expert_index],
                    "layer": int(data.layers[expert_index]),
                    "expert_id": int(data.expert_ids[expert_index]),
                    "predicted_contrast": predicted_specificity[expert_index],
                    "actual_contrast": actual_specificity[expert_index],
                    "sign_agreement": np.sign(predicted_specificity[expert_index])
                    == np.sign(actual_specificity[expert_index]),
                }
            )
            predicted_top = int(np.argmax(predicted[expert_index]))
            actual_top = int(np.argmax(data.actual_delta_nll[expert_index]))
            ranking_rows.append(
                {
                    "surrogate": SURROGATE_LABELS.get(name, name),
                    "internal_name": name,
                    "intervention_id": intervention_id,
                    "layer": int(data.layers[expert_index]),
                    "expert_id": int(data.expert_ids[expert_index]),
                    "target_domain": data.target_domains[expert_index],
                    "per_expert_spearman": per_expert[expert_index],
                    "predicted_top_domain": data.domains[predicted_top],
                    "actual_top_domain": data.domains[actual_top],
                    "top_domain_correct": predicted_top == actual_top,
                    **{
                        f"predicted_rank_{domain}": int(
                            predicted_ranks[expert_index, domain_index]
                        )
                        for domain_index, domain in enumerate(data.domains)
                    },
                    **{
                        f"actual_rank_{domain}": int(
                            actual_ranks[expert_index, domain_index]
                        )
                        for domain_index, domain in enumerate(data.domains)
                    },
                }
            )
    write_csv(
        output_dir / "surrogate_specificity.csv",
        specificity_rows,
        [
            "surrogate",
            "internal_name",
            "intervention_id",
            "target_domain",
            "layer",
            "expert_id",
            "predicted_contrast",
            "actual_contrast",
            "sign_agreement",
        ],
    )
    rank_fields = [
        "surrogate",
        "internal_name",
        "intervention_id",
        "layer",
        "expert_id",
        "target_domain",
        "per_expert_spearman",
        "predicted_top_domain",
        "actual_top_domain",
        "top_domain_correct",
        *[f"predicted_rank_{domain}" for domain in data.domains],
        *[f"actual_rank_{domain}" for domain in data.domains],
    ]
    write_csv(
        output_dir / "within_expert_domain_rankings.csv", ranking_rows, rank_fields
    )
    domain_rows = []
    for name, metric in analysis["surrogates"].items():
        for row in metric["domain_correlations"]:
            domain_rows.append(
                {
                    "surrogate": metric["label"],
                    "internal_name": name,
                    **row,
                }
            )
    write_csv(
        output_dir / "domain_specific_correlations.csv",
        domain_rows,
        [
            "surrogate",
            "internal_name",
            "domain",
            "spearman",
            "spearman_ci_low",
            "spearman_ci_high",
        ],
    )


def write_pilot_raw_npz(
    path: Path,
    data: PilotValidationData,
    activation_raw: Mapping[str, np.ndarray],
    *,
    gradient_raw: Mapping[str, np.ndarray] | None = None,
) -> None:
    arrays: dict[str, Any] = {
        "intervention_ids": np.asarray(data.intervention_ids, dtype=np.str_),
        "target_domains": np.asarray(data.target_domains, dtype=np.str_),
        "layers": data.layers.astype(np.int16),
        "expert_ids": data.expert_ids.astype(np.int16),
        "domain_names": np.asarray(data.domains, dtype=np.str_),
        "actual_per_example_delta_nll": data.per_example_delta_nll,
        "actual_delta_nll": data.actual_delta_nll,
        "functional_importance": data.functional_importance,
        "routing_importance": data.routing_importance,
        "quantization_distortion": data.quantization_distortion,
    }
    for name, values in activation_raw.items():
        arrays[f"activation_{name}"] = np.asarray(values)
    if gradient_raw is not None:
        for name, values in gradient_raw.items():
            arrays[f"gradient_{name}"] = np.asarray(values)
    atomic_save_npz(path, **arrays)


def create_surrogate_figures(
    output_dir: Path,
    data: PilotValidationData,
    scores: Mapping[str, np.ndarray],
    analysis: Mapping[str, Any],
    *,
    primary_name: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "general": "#4C78A8",
        "math": "#F58518",
        "coding": "#54A24B",
        "reasoning": "#B279A2",
    }
    outputs: list[Path] = []
    primary = scores[primary_name]

    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    for domain_index, domain in enumerate(data.domains):
        axis.scatter(
            primary[:, domain_index],
            data.actual_delta_nll[:, domain_index],
            label=domain.title(),
            color=colors[domain],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    axis.set_xlabel(SURROGATE_LABELS.get(primary_name, primary_name))
    axis.set_ylabel("Actual 4-bit ΔNLL")
    axis.set_title("Actual quantization damage vs fixed surrogate")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    outputs.extend(_save_figure(figure, figure_dir / "figure_1_actual_vs_surrogate", plt))

    target_indices = data.target_domain_indices
    predicted_specificity = specificity_contrast(primary, target_indices)
    actual_specificity = specificity_contrast(data.actual_delta_nll, target_indices)
    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    for index, domain in enumerate(data.target_domains):
        axis.scatter(
            predicted_specificity[index],
            actual_specificity[index],
            color=colors[domain],
            s=48,
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Predicted target-minus-other contrast")
    axis.set_ylabel("Actual target-minus-other ΔNLL contrast")
    axis.set_title("Actual vs predicted quantization specificity")
    axis.grid(alpha=0.18)
    outputs.extend(
        _save_figure(figure, figure_dir / "figure_2_actual_vs_specificity", plt)
    )

    comparison_names = [
        name
        for name in (
            "weight_risk_functional",
            "uod",
            "reod",
            "apd",
            "aod",
            "gqs",
        )
        if name in analysis["surrogates"]
    ]
    estimates = np.asarray(
        [
            _finite_or_nan(
                analysis["surrogates"][name]["overall"]["spearman"]
            )
            for name in comparison_names
        ],
        dtype=np.float64,
    )
    lows = np.asarray(
        [
            _finite_or_default(
                analysis["surrogates"][name]["overall"]["spearman_ci_low"],
                estimates[index],
            )
            for index, name in enumerate(comparison_names)
        ],
        dtype=np.float64,
    )
    highs = np.asarray(
        [
            _finite_or_default(
                analysis["surrogates"][name]["overall"]["spearman_ci_high"],
                estimates[index],
            )
            for index, name in enumerate(comparison_names)
        ],
        dtype=np.float64,
    )
    y = np.arange(len(comparison_names))
    figure, axis = plt.subplots(figsize=(7.4, max(4.4, 0.62 * len(comparison_names))))
    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="o",
        color="#4C78A8",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_yticks(y, [SURROGATE_LABELS[name] for name in comparison_names])
    axis.set_xlabel("Overall Spearman (95% grouped-bootstrap CI)")
    axis.set_title("Fixed surrogate comparison")
    axis.grid(axis="x", alpha=0.18)
    figure.tight_layout()
    outputs.extend(_save_figure(figure, figure_dir / "figure_3_surrogate_comparison", plt))

    predicted_top = np.argmax(primary, axis=1)
    actual_top = np.argmax(data.actual_delta_nll, axis=1)
    figure, axis = plt.subplots(figsize=(8.8, 6.0))
    for index in range(data.num_experts):
        axis.plot(
            [0, 1],
            [predicted_top[index], actual_top[index]],
            color=("#54A24B" if predicted_top[index] == actual_top[index] else "#E45756"),
            alpha=0.75,
        )
        axis.scatter([0], [predicted_top[index]], color="#4C78A8", s=25)
        axis.scatter([1], [actual_top[index]], color="#F58518", s=25)
    axis.set_xlim(-0.25, 1.25)
    axis.set_xticks([0, 1], ["Predicted worst", "Actual worst"])
    axis.set_yticks(range(data.num_domains), [domain.title() for domain in data.domains])
    axis.set_title("Worst-domain identification for each pilot expert")
    axis.grid(axis="y", alpha=0.18)
    outputs.extend(_save_figure(figure, figure_dir / "figure_4_domain_prediction", plt))
    return outputs


def write_surrogate_summary(
    path: Path,
    results: Mapping[str, Any],
) -> None:
    decision = results["surrogate_decision"]
    aod_analysis = results["aod_analysis"]
    aod = aod_analysis["surrogates"]["aod"]
    gradient = results.get("gradient_fallback", {})
    gradient_analysis = gradient.get("analysis") if isinstance(gradient, dict) else None
    comparison = (
        gradient_analysis["comparison"]
        if gradient_analysis
        else results["aod_analysis"]["comparison"]
    )
    lines = [
        "# OLMoE Stage 2A Quantization-Cost Surrogate",
        "",
        "## Scope",
        "",
        "This stage validates fixed activation-aware replay scores against all 64 frozen "
        "Stage-1 expert-domain 4-bit ΔNLL observations. No regressor, coefficient fit, "
        "post-hoc formula search, mixed-precision allocation, or model update is used.",
        "",
        "## Primary formula",
        "",
        "`AOD(l,e,d,b) = Σ_t ||g(l,t,e) · (f_q(h)-f(h))||² / "
        "(Σ_t ||y_moe(l,t)||² + 1e-30)`",
        "",
        "## AOD validation",
        "",
        f"- Overall Spearman: {_format_number(aod['overall']['spearman'])} "
        f"[{_format_number(aod['overall']['spearman_ci_low'])}, "
        f"{_format_number(aod['overall']['spearman_ci_high'])}]",
        f"- Overall Kendall tau: {_format_number(aod['overall']['kendall_tau'])} "
        f"[{_format_number(aod['overall']['kendall_tau_ci_low'])}, "
        f"{_format_number(aod['overall']['kendall_tau_ci_high'])}]",
        f"- Specificity Spearman: {_format_number(aod['specificity']['spearman'])} "
        f"[{_format_number(aod['specificity']['spearman_ci_low'])}, "
        f"{_format_number(aod['specificity']['spearman_ci_high'])}]",
        f"- Specificity Kendall tau: "
        f"{_format_number(aod['specificity']['kendall_tau'])}",
        f"- Specificity sign agreement: "
        f"{_format_number(aod['specificity']['sign_agreement'], digits=4, signed=False)}",
        f"- Top-domain accuracy: "
        f"{aod['within_expert_domain_ranking']['top_domain_accuracy']:.4f} "
        f"[{_format_number(aod['within_expert_domain_ranking']['top_domain_accuracy_ci_low'], digits=4, signed=False)}, "
        f"{_format_number(aod['within_expert_domain_ranking']['top_domain_accuracy_ci_high'], digits=4, signed=False)}]",
        f"- Mean/median within-expert domain Spearman: "
        f"{_format_number(aod['within_expert_domain_ranking']['mean_spearman'])} / "
        f"{_format_number(aod['within_expert_domain_ranking']['median_spearman'])}",
        f"- Positive per-domain correlations: {aod['positive_domain_correlation_count']}/4",
        f"- Improvement over WeightRiskFunctional: "
        f"{_format_number(aod_analysis['weight_proxy_improvement_bootstrap']['point_estimate'])} "
        f"[{_format_number(aod_analysis['weight_proxy_improvement_bootstrap']['ci_low'])}, "
        f"{_format_number(aod_analysis['weight_proxy_improvement_bootstrap']['ci_high'])}]",
        "",
        "| Domain | AOD Spearman | 95% grouped-bootstrap CI |",
        "|---|---:|---:|",
        *[
            f"| {row['domain'].title()} | {_format_number(row['spearman'])} | "
            f"[{_format_number(row['spearman_ci_low'])}, "
            f"{_format_number(row['spearman_ci_high'])}] |"
            for row in aod["domain_correlations"]
        ],
        "",
        "| AOD gate | Outcome |",
        "|---|---:|",
        *[
            f"| {name.replace('_', ' ').title()} | "
            f"{'PASS' if aod_analysis['primary_gates'][name]['passed'] else 'FAIL'} |"
            for name in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e")
        ],
        "",
        "## Fixed surrogate comparison",
        "",
        "| Surrogate | Overall Spearman | Specificity Spearman | Top-domain accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['surrogate']} | {_format_number(row['overall_spearman'])} | "
            f"{_format_number(row['specificity_spearman'])} | "
            f"{_format_number(row['top_domain_accuracy'], digits=4, signed=False)} |"
        )
    if gradient_analysis:
        gqs = gradient_analysis["surrogates"]["gqs"]
        lines.extend(
            [
                "",
                "## Pre-registered gradient fallback",
                "",
                "AOD failed at least one gate, so GQS was activated exactly as "
                "pre-registered. GQS is primary; GQS2 remains diagnostic only.",
                "",
                f"- Overall GQS Spearman: "
                f"{_format_number(gqs['overall']['spearman'])} "
                f"[{_format_number(gqs['overall']['spearman_ci_low'])}, "
                f"{_format_number(gqs['overall']['spearman_ci_high'])}]",
                f"- GQS specificity Spearman: "
                f"{_format_number(gqs['specificity']['spearman'])}",
                f"- GQS top-domain accuracy: "
                f"{_format_number(gqs['within_expert_domain_ranking']['top_domain_accuracy'], digits=4, signed=False)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['decision']}**",
            "",
            decision["rationale"],
            "",
            "The distributionally robust mixed-precision optimizer is not implemented by "
            "this stage.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _surrogate_note(name: str, primary_name: str) -> str:
    notes = {
        "weight_risk_functional": "Frozen failed Stage-1 negative-control proxy",
        "weight_risk_routing": "Frozen failed Stage-1 negative-control proxy",
        "functional_importance": "No weight-distortion multiplier",
        "routing_importance": "No weight-distortion multiplier",
        "uod": "Diagnostic; omits router coefficient",
        "reod": "Secondary; conditional on the expert's own routed contribution",
        "apd": "Secondary absolute perturbation per measured domain token",
        "aod": "PRIMARY fixed activation-aware score",
        "gqs": "Pre-registered gradient fallback",
    }
    value = notes.get(name, "Fixed diagnostic")
    if name == primary_name:
        value += "; gate-evaluated metric"
    return value


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.size < 2 or np.all(x == x.flat[0]) or np.all(y == y.flat[0]):
        return math.nan
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else math.nan


def _safe_kendall(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.size < 2 or np.all(x == x.flat[0]) or np.all(y == y.flat[0]):
        return math.nan
    value = float(kendalltau(x, y).statistic)
    return value if math.isfinite(value) else math.nan


def _finite_ci(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _strictly_greater(value: Any, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > threshold


def _at_least(value: Any, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= threshold


def _format_number(value: Any, *, digits: int = 6, signed: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite_or_default(value: Any, default: float) -> float:
    number = _finite_or_nan(value)
    return number if math.isfinite(number) else default


def _save_figure(figure: Any, base: Path, plt: Any) -> list[Path]:
    figure.tight_layout()
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths
