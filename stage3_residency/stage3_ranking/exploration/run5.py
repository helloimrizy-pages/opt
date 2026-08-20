"""Does truncating the learning target to the slot-residence horizon help?"""
import time, numpy as np
from harness import *
from features3 import FeatureState3
from collect2 import sub
from fit import group_slices, standardize, build_pairs, fit_pairwise
from run4 import simulate_multi
from residency_headroom.workloads import calibration_frequency_scores

CAPS4 = (12, 16, 24, 32)

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA'); calB = sub(inputs.calibration, 40, 80, 'calB')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    refs = references(inputs, calB, CAPS)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses) for x in simulate_causal_capacities(tr, calB, CAPS, spec, models)}
    report("STAGE1 winner", s1, refs, CAPS4)
    print(f"  residence horizon ~ B/8 events: " + ", ".join(f"B={c}->{c//8}" for c in CAPS4))
    print()
    from committed import seed_weights
    w0 = seed_weights()   # round-1 pooled model, from the committed calibration selection
    st = FeatureState3(models, tr.num_layers, tr.num_experts, static)
    buckets = [[[], [], [], 0] for _ in CAPS4]
    simulate_multi(tr, calA, CAPS4, st, [lambda f: w0 @ f]*4, collect_to=buckets)
    data = []
    for ci in range(4):
        data.append((np.concatenate(buckets[ci][0]).astype(np.float64),
                     np.concatenate(buckets[ci][1]).astype(np.float64),
                     np.concatenate(buckets[ci][2])))
    best = None
    for T in (2, 3, 4, 6, 8, 12, 33):
        ws = []
        for ci in range(4):
            X, Y, G = data[ci]
            Yt = np.minimum(Y, T + 1)
            mu, sd = standardize(X)
            D, wp = build_pairs((X - mu)/sd, Yt, G, max_pairs=30)
            th, _ = fit_pairwise(D, wp, 3e-3)
            ws.append(th / sd)
        c = simulate_multi(tr, calB, CAPS4, st, [(lambda w: (lambda f: w @ f))(w) for w in ws])
        g = report(f"target truncated at T={T:<2d}", c, refs, CAPS4)
        imp = {k: 100*(s1[k]-c[k])/s1[k] for k in CAPS4}
        print(f"      vs Stage 1: " + " ".join(f"{k}:{v:+.2f}%" for k, v in imp.items()))
        if best is None or g > best[0]:
            best = (g, T, ws)
    print(f"\nbest truncation T={best[1]} mean_gap={best[0]*100:.2f}%")
    np.savez('fit5.npz', T=best[1], **{f"w{c}": w for c, w in zip(CAPS4, best[2])})
