"""Causal per-expert feature state. Every feature uses only events already observed."""
from __future__ import annotations
import numpy as np

HORIZONS = (1, 2, 4, 8, 16, 32)
CAP = 33

NAMES = (
    "markov_h1", "markov_h2", "markov_h4", "markov_h8", "markov_h16", "markov_h32",
    "nor_h1", "nor_h2", "nor_h4",
    "log_elapsed", "recip_elapsed", "never_seen",
    "freq_a90", "freq_a95", "freq_a99",
    "gate_ewma", "last_gate", "in_prev_request",
    "static_pop",
)


class FeatureState:
    def __init__(self, models, num_layers, num_experts, static_pop):
        self.M = np.stack([np.stack([models.matrix(h, l) for h in HORIZONS])
                           for l in range(num_layers)])
        self.M.flags.writeable = False
        self.L, self.E = num_layers, num_experts
        self.static = static_pop / np.maximum(static_pop.sum(axis=1, keepdims=True), 1.0)
        self.reset()

    def reset(self):
        L, E = self.L, self.E
        self.last = np.full((L, E), -1, dtype=np.int64)
        self.f90 = np.zeros((L, E)); self.f95 = np.zeros((L, E)); self.f99 = np.zeros((L, E))
        self.gate = np.zeros((L, E)); self.lastgate = np.zeros((L, E))
        self.prev = [None] * L

    def features(self, layer, request, gates, sorted_request, position):
        """Feature block for every expert, computed BEFORE this event is absorbed."""
        block = self.M[layer][:, sorted_request, :]                  # (6, k, E)
        markov = block.mean(axis=1)                                  # (6, E)
        nor = 1.0 - np.prod(1.0 - block[:3], axis=1)                 # (3, E)
        elapsed = position - self.last[layer]
        seen = self.last[layer] >= 0
        tau = np.where(seen, np.maximum(elapsed, 1), 4096)
        prev = np.zeros(self.E)
        if self.prev[layer] is not None:
            prev[self.prev[layer]] = 1.0
        return np.stack([
            *markov, *nor,
            np.log1p(tau), 1.0 / tau, (~seen).astype(np.float64),
            self.f90[layer], self.f95[layer], self.f99[layer],
            self.gate[layer], self.lastgate[layer], prev,
            self.static[layer],
        ])                                                            # (F, E)

    def absorb(self, layer, request, gates, position):
        self.f90[layer] *= 0.90; self.f90[layer][request] += 1.0
        self.f95[layer] *= 0.95; self.f95[layer][request] += 1.0
        self.f99[layer] *= 0.99; self.f99[layer][request] += 1.0
        self.gate[layer] *= 0.95; self.gate[layer][request] += 0.05 * gates
        self.lastgate[layer][request] = gates
        self.prev[layer] = request
        self.last[layer, request] = position


class LinearScorer:
    """Deploys a fitted linear model as a retention score (higher = retain)."""

    def __init__(self, state, weights, bias, name="linear"):
        self.state, self.w, self.b, self.name = state, np.asarray(weights), float(bias), name

    def reset(self):
        self.state.reset()

    def step(self, layer, request, gates, sorted_request, position):
        feats = self.state.features(layer, request, gates, sorted_request, position)
        predicted = self.w @ feats + self.b
        self.state.absorb(layer, request, gates, position)
        return -predicted
