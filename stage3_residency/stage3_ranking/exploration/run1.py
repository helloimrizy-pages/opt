import time, numpy as np
from harness import *
from scorers import *
from race_stage1.models import same_layer_indices
from race_stage2.frozen import truncated_workload
from residency_headroom.workloads import Workload, WorkloadSequence

inputs, models = load()
tr = inputs.trace
cal = inputs.calibration

def sub(workload, lo, hi, name):
    seqs = tuple(WorkloadSequence(s.source_sequence_id, i, s.segment_index, s.segment_label, s.domain)
                 for i, s in enumerate(workload.sequences[lo:hi]))
    return Workload(name=name, regime=workload.regime, sequences=seqs)

calA = sub(cal, 0, 40, 'calA')     # fit any new component here
calB = sub(cal, 40, 80, 'calB')    # measure here
print(f"calA={len(calA.sequences)} seqs  calB={len(calB.sequences)} seqs")

refs = references(inputs, calB, CAPS)
spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
s1 = {int(x.capacity): int(x.misses) for x in simulate_causal_capacities(tr, calB, CAPS, spec, models)}
print("calB simple:", refs['simple'], " oracle:", refs['oracle'])
base = report("STAGE1 winner (frozen)", s1, refs, CAPS)
print()

t0=time.perf_counter(); c,_ = simulate_scorer(tr, calB, CAPS, Stage1Ref(models, tr.num_layers, tr.num_experts))
report("  harness check: stage1 reimpl", c, refs, CAPS, time.perf_counter()-t0)
print("  (must match the frozen row above:", c == s1, ")\n")

for S in (ExpDistMarkov(models, tr.num_layers, tr.num_experts),
          ExpDistMarkovNoisyOr(models, tr.num_layers, tr.num_experts)):
    t0=time.perf_counter(); c,_ = simulate_scorer(tr, calB, CAPS, S)
    report(S.name, c, refs, CAPS, time.perf_counter()-t0)

t0=time.perf_counter()
surv, cum, marg = fit_hazard(tr, calA, same_layer_indices(tr, calA))
print(f"\nhazard fitted on calA in {time.perf_counter()-t0:.0f}s")
hz = Hazard(surv, cum, marg, tr.num_layers, tr.num_experts)
t0=time.perf_counter(); c,_ = simulate_scorer(tr, calB, CAPS, hz)
report("hazard (fit calA)", c, refs, CAPS, time.perf_counter()-t0)

mk = ExpDistMarkov(models, tr.num_layers, tr.num_experts)
for beta in (0.25, 0.5, 0.75):
    b = Blend(mk, Hazard(surv, cum, marg, tr.num_layers, tr.num_experts), beta)
    t0=time.perf_counter(); c,_ = simulate_scorer(tr, calB, CAPS, b)
    report(b.name, c, refs, CAPS, time.perf_counter()-t0)
