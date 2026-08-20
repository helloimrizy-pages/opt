"""Reversed split: fit on calB, measure on calA. Confirms the gain is not a calB artifact."""
import numpy as np
from harness import *
from features3 import FeatureState3
from collect2 import sub
from fit import group_slices, standardize, build_pairs, fit_pairwise
from run4 import simulate_multi
from residency_headroom.workloads import calibration_frequency_scores

CAPS4 = (12, 16, 24, 32)

def fit_on(tr, fitwl, models, static, seed_w):
    st = FeatureState3(models, tr.num_layers, tr.num_experts, static)
    buckets = [[[], [], [], 0] for _ in CAPS4]
    simulate_multi(tr, fitwl, CAPS4, st, [lambda f: seed_w @ f]*4, collect_to=buckets)
    ws = []
    for ci in range(4):
        X = np.concatenate(buckets[ci][0]).astype(np.float64)
        Y = np.concatenate(buckets[ci][1]).astype(np.float64)
        G = np.concatenate(buckets[ci][2])
        mu, sd = standardize(X)
        D, wp = build_pairs((X-mu)/sd, Y, G, max_pairs=30)
        th, _ = fit_pairwise(D, wp, 3e-3)
        ws.append(th/sd)
    return ws, st

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA'); calB = sub(inputs.calibration, 40, 80, 'calB')
    from committed import seed_weights
    seed = seed_weights()
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    for fitwl, testwl, tag in ((calA, calB, "fit calA -> test calB"),
                               (calB, calA, "fit calB -> test calA")):
        static = calibration_frequency_scores(tr, fitwl).astype(np.float64)
        refs = references(inputs, testwl, CAPS4)
        s1 = {int(x.capacity): int(x.misses)
              for x in simulate_causal_capacities(tr, testwl, CAPS4, spec, models)}
        ws, st = fit_on(tr, fitwl, models, static, seed)
        c = simulate_multi(tr, testwl, CAPS4, st, [(lambda w: (lambda f: w @ f))(w) for w in ws])
        print(f"--- {tag} ---")
        report("  STAGE1 winner", s1, refs, CAPS4)
        report("  learned ranking scorer", c, refs, CAPS4)
        imp = {k: 100*(s1[k]-c[k])/s1[k] for k in CAPS4}
        res = {k: 100*(s1[k]-c[k])/(s1[k]-refs['oracle'][k]) for k in CAPS4}
        print("    improvement vs Stage 1 :", " ".join(f"{k}:{v:+.2f}%" for k, v in imp.items()))
        print("    Stage 1 residual taken :", " ".join(f"{k}:{v:.1f}%" for k, v in res.items()))
