"""Stage 2 variant specifications and their frozen identifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .advisers import pool_size, uniform_weights, validate_simplex


WEIGHT_SOURCES = ("uniform", "static_global", "static_per_layer")
LOSSES = ("rank", "cost")
SCOPES = ("per_layer", "global")


@dataclass(frozen=True)
class RaceVariant:
    """One fully specified RACE configuration.

    The eviction mechanism is identical for every variant; only the adviser pool,
    the adviser weight vector and whether it adapts differ.
    """

    name: str
    weight_source: str = "uniform"
    adaptive: bool = False
    loss: str | None = None
    eta: float | None = None
    scope: str = "per_layer"
    pool: str = "primary"
    static_weights: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.weight_source not in WEIGHT_SOURCES:
            raise ValueError(f"Unknown weight source {self.weight_source!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"Unknown weight scope {self.scope!r}")
        size = pool_size(self.pool)
        if self.adaptive:
            if self.loss not in LOSSES:
                raise ValueError("An adaptive variant needs a rank or cost loss")
            if self.eta is None or not (float(self.eta) > 0.0):
                raise ValueError("An adaptive variant needs a positive learning rate")
        else:
            if self.loss is not None or self.eta is not None:
                raise ValueError("A non-adaptive variant must not carry a loss or eta")
        if self.weight_source == "uniform":
            if self.static_weights is not None:
                raise ValueError("Uniform variants must not carry static weights")
        else:
            if self.static_weights is None:
                raise ValueError(f"{self.weight_source} requires static weights")
            values = np.asarray(self.static_weights, dtype=np.float64)
            if self.weight_source == "static_global":
                validate_simplex(values, size)
            else:
                if values.ndim != 2 or values.shape[1] != size:
                    raise ValueError("Per-layer static weights must be [layer, adviser]")
                for row in values:
                    validate_simplex(row, size)
        if self.weight_source == "static_per_layer" and self.scope != "per_layer":
            raise ValueError("Per-layer static weights require the per-layer scope")

    @property
    def size(self) -> int:
        return pool_size(self.pool)

    @property
    def variant_id(self) -> str:
        parts = [self.name.lower()]
        if self.weight_source != "uniform":
            parts.append(self.weight_source)
        if self.adaptive:
            parts.append(f"{self.loss}loss")
            parts.append(f"eta{format(float(self.eta), '.12g')}")
            parts.append(
                f"init_{'uniform' if self.weight_source == 'uniform' else 'static'}"
            )
            parts.append(f"scope_{self.scope}")
        if self.pool != "primary":
            parts.append(f"pool_{self.pool}")
        return "_".join(parts)

    def parameters(self) -> dict[str, Any]:
        return {
            "weight_source": self.weight_source,
            "adaptive": bool(self.adaptive),
            "loss": self.loss,
            "eta": None if self.eta is None else float(self.eta),
            "scope": self.scope,
            "pool": self.pool,
        }

    def initial_weight_matrix(self, streams: int) -> np.ndarray:
        """Initial weights for every learning stream of one capacity instance."""

        size = pool_size(self.pool)
        if self.weight_source == "uniform":
            return np.tile(uniform_weights(size), (streams, 1))
        values = np.asarray(self.static_weights, dtype=np.float64)
        if self.weight_source == "static_global":
            return np.tile(values, (streams, 1))
        if values.shape[0] != streams:
            raise ValueError("Per-layer static weights do not match the stream count")
        return values.copy()


def uniform_variant(pool: str = "primary") -> RaceVariant:
    name = "RACE_UNIFORM" if pool == "primary" else f"RACE_UNIFORM_{pool.upper()}"
    return RaceVariant(name=name, pool=pool)


def static_variant(weights: np.ndarray, pool: str = "primary") -> RaceVariant:
    name = "RACE_STATIC" if pool == "primary" else f"RACE_STATIC_{pool.upper()}"
    return RaceVariant(
        name=name, weight_source="static_global", static_weights=weights, pool=pool
    )


def static_per_layer_variant(weights: np.ndarray, pool: str = "primary") -> RaceVariant:
    return RaceVariant(
        name="RACE_STATIC_PERLAYER",
        weight_source="static_per_layer",
        static_weights=weights,
        pool=pool,
    )


def online_variant(
    *,
    loss: str,
    eta: float,
    initialization: str,
    static_weights: np.ndarray | None = None,
    scope: str = "per_layer",
    pool: str = "primary",
) -> RaceVariant:
    if initialization not in {"uniform", "static"}:
        raise ValueError("Initialization must be 'uniform' or 'static'")
    weight_source = "uniform" if initialization == "uniform" else "static_global"
    name = "RACE_ONLINE" if loss == "rank" else "RACE_COST"
    if scope == "global":
        name = f"{name}_GLOBAL"
    if pool != "primary":
        name = f"{name}_{pool.upper()}"
    return RaceVariant(
        name=name,
        weight_source=weight_source,
        adaptive=True,
        loss=loss,
        eta=float(eta),
        scope=scope,
        pool=pool,
        static_weights=None if weight_source == "uniform" else static_weights,
    )


def variant_from_spec(spec: Mapping[str, Any]) -> RaceVariant:
    """Rebuild a variant from a serializable specification."""

    weights = spec.get("static_weights")
    return RaceVariant(
        name=str(spec["name"]),
        weight_source=str(spec.get("weight_source", "uniform")),
        adaptive=bool(spec.get("adaptive", False)),
        loss=spec.get("loss"),
        eta=spec.get("eta"),
        scope=str(spec.get("scope", "per_layer")),
        pool=str(spec.get("pool", "primary")),
        static_weights=None if weights is None else np.asarray(weights, dtype=np.float64),
    )


def variant_to_spec(variant: RaceVariant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "weight_source": variant.weight_source,
        "adaptive": bool(variant.adaptive),
        "loss": variant.loss,
        "eta": None if variant.eta is None else float(variant.eta),
        "scope": variant.scope,
        "pool": variant.pool,
        "static_weights": (
            None
            if variant.static_weights is None
            else np.asarray(variant.static_weights, dtype=np.float64).tolist()
        ),
    }
