import time, numpy as np
from harness import *
from features import FeatureState, NAMES
from collect2 import sub
from residency_headroom.workloads import calibration_frequency_scores

class Deployed:
    def __init__(self, state, weights, name, sign=+1.0):
        self.state, self.w, self.name, self.sign = state, np.asarray(weights, float), name, sign
    def reset(self): self.state.reset()
    def step(self, layer, request, gates, sorted_request, position):
        f = self.state.features(layer, request, gates, sorted_request, position)
        out = self.sign * (self.w @ f)
        self.state.absorb(layer, request, gates, position)
        return out

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA')
    calB = sub(inputs.calibration, 40, 80, 'calB')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    refs = references(inputs, calB, CAPS)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses)
          for x in simulate_causal_capacities(tr, calB, CAPS, spec, models)}
    print("calB references — simple:", refs['simple'], "oracle:", refs['oracle'])
    report("STAGE1 winner (target to beat)", s1, refs, CAPS)
    print()
    import fit as _fit  # builds fit_linear.npz on demand
    if not __import__('pathlib').Path('fit_linear.npz').exists():
        __import__('runpy').run_path('fit.py', run_name='__main__')
    z = np.load('fit_linear.npz')
    st = FeatureState(models, tr.num_layers, tr.num_experts, static)
    for tag, theta, sign in (("pairwise-logistic linear", z['theta'], +1.0),
                             ("least-squares linear", z['ls'], +1.0)):
        w = np.asarray(theta) / z['sd']
        t0 = time.perf_counter()
        c, _ = simulate_scorer(tr, calB, CAPS, Deployed(st, w, tag, sign))
        report(tag, c, refs, CAPS, time.perf_counter() - t0)
    print("\nImprovement over the Stage 1 winner (the binding Condition A metric):")
    w = np.asarray(z['theta']) / z['sd']
    c, _ = simulate_scorer(tr, calB, CAPS, Deployed(st, w, 'x'))
    for cap in CAPS:
        print(f"  capacity {cap:>2d}: {100*(s1[cap]-c[cap])/s1[cap]:+6.2f}%   "
              f"(need +10% at 3 of 4)")
