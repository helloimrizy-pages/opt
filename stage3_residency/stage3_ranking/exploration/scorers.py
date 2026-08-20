"""Candidate causal scorers. Larger returned value = retain more strongly."""
from __future__ import annotations
import numpy as np

HMAX = 32
CAP = HMAX + 1
HORIZONS = (1, 2, 4, 8, 16, 32)

# E[min(d,33)] = 33 - sum_{h=1..32} S_h, with S piecewise-linear between the fitted
# horizons. Collecting terms gives fixed coefficients on the six survival points.
# Segment [a,b] contributes S_a*(b-a-1)/2 + S_b*(b-a+1)/2.
EXPDIST_COEFF = np.array([1.0, 1.5, 3.0, 6.0, 12.0, 8.5])
assert EXPDIST_COEFF.sum() == 32.0


class ExpDistMarkov:
    """-E[min(d,33)] from the six frozen Markov survival probabilities.

    Because the functional is linear in S_h and S_h is a mean over the current
    request's rows, the whole estimator collapses to one precomputed matrix per
    layer: score(e) = mean_{e' in R} V[e', e], with V = sum_h c_h * M_h.
    """
    name = "expdist_markov"

    def __init__(self, models, num_layers, num_experts):
        self.V = np.stack([
            sum(EXPDIST_COEFF[i] * models.matrix(h, l) for i, h in enumerate(HORIZONS))
            for l in range(num_layers)])
        self.V.flags.writeable = False

    def reset(self): pass

    def step(self, layer, request, gates, sorted_request, position):
        return self.V[layer][sorted_request].mean(axis=0)


class ExpDistMarkovNoisyOr:
    """Same target, but combine the request's rows by noisy-OR instead of averaging."""
    name = "expdist_markov_noisyor"

    def __init__(self, models, num_layers, num_experts):
        self.M = np.stack([np.stack([models.matrix(h, l) for h in HORIZONS])
                           for l in range(num_layers)])
        self.M.flags.writeable = False
        self.c = EXPDIST_COEFF[:, None]

    def reset(self): pass

    def step(self, layer, request, gates, sorted_request, position):
        block = self.M[layer][:, sorted_request, :]              # (6, k, E)
        surv = 1.0 - np.prod(1.0 - block, axis=1)                # (6, E)
        return (self.c * surv).sum(axis=0)


class Hazard:
    """-E[min(d,33) | elapsed] from a calibration-fitted per-expert gap survival.

    Uses the elapsed same-layer time since an expert was last requested, which no
    Stage 1 or Stage 2 adviser conditions on in a calibrated way.
    """
    name = "hazard"
    SUPPORT = 4096

    def __init__(self, survival, cumsum, marginal, num_layers, num_experts):
        self.G, self.CG, self.marginal = survival, cumsum, marginal
        self.L, self.E = num_layers, num_experts
        self.reset()

    def reset(self):
        self.last = np.full((self.L, self.E), -1, dtype=np.int64)

    def expected(self, layer, position):
        elapsed = position - self.last[layer]
        seen = self.last[layer] >= 0
        tau = np.clip(elapsed, 1, self.SUPPORT - HMAX - 1)
        g = self.G[layer][np.arange(self.E), tau]
        num = (self.CG[layer][np.arange(self.E), tau + HMAX]
               - self.CG[layer][np.arange(self.E), tau - 1])
        value = np.where(g > 1e-12, num / np.maximum(g, 1e-12), float(CAP))
        return np.where(seen, np.minimum(value, CAP), self.marginal[layer])

    def step(self, layer, request, gates, sorted_request, position):
        out = -self.expected(layer, position)
        self.last[layer, request] = position
        return out


def fit_hazard(trace, workload, layer_streams, support=Hazard.SUPPORT):
    """Empirical per-(layer, expert) inter-arrival survival from calibration only."""
    L, E = trace.num_layers, trace.num_experts
    counts = np.zeros((L, E, support + HMAX + 2), dtype=np.float64)
    for l, idx in enumerate(layer_streams):
        reqs = trace.requested_expert_ids[idx].astype(np.int64)
        last = np.full(E, -1, dtype=np.int64)
        for p in range(reqs.shape[0]):
            r = reqs[p]
            gap = p - last[r]
            prev = last[r] >= 0
            for e, g, ok in zip(r, gap, prev):
                if ok:
                    counts[l, e, min(int(g), support)] += 1.0
            last[r] = p
    # Laplace smoothing keeps every survival strictly positive.
    counts += 1e-3
    total = counts.sum(axis=2, keepdims=True)
    pmf = counts / total
    survival = 1.0 - np.cumsum(pmf, axis=2)          # G(t) = P(gap > t)
    survival = np.clip(survival, 1e-12, 1.0)
    cumsum = np.cumsum(survival, axis=2)
    marginal = np.zeros((L, E))
    for l in range(L):
        marginal[l] = np.minimum(CAP, cumsum[l][:, HMAX])
    return survival, cumsum, marginal


class Blend:
    """Convex blend of two expected-distance estimators, on the distance scale."""

    def __init__(self, markov, hazard, beta, name=None):
        self.m, self.h, self.beta = markov, hazard, float(beta)
        self.name = name or f"blend_beta{beta:g}"

    def reset(self):
        self.m.reset(); self.h.reset()

    def step(self, layer, request, gates, sorted_request, position):
        em = CAP - self.m.step(layer, request, gates, sorted_request, position)
        eh = -self.h.step(layer, request, gates, sorted_request, position)
        return -(self.beta * em + (1.0 - self.beta) * eh)


class Stage1Ref:
    """The frozen Stage 1 winner, reimplemented here purely as a harness check."""
    name = "stage1_hybrid_ref"

    def __init__(self, models, num_layers, num_experts, beta=0.5, alpha=0.95):
        self.M = np.stack([models.matrix(2, l) for l in range(num_layers)])
        self.beta, self.alpha = beta, alpha
        self.L, self.E = num_layers, num_experts
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.L, self.E))

    def step(self, layer, request, gates, sorted_request, position):
        cond = self.M[layer][sorted_request].mean(axis=0)
        self.hist[layer] *= self.alpha
        self.hist[layer][request] += 1.0 - self.alpha
        return self.beta * cond + (1.0 - self.beta) * self.hist[layer]
