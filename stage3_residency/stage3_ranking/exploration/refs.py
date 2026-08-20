import time, json, numpy as np
from harness import *
from race_stage2.frozen import truncated_workload

inputs, models = load()
# Fast iteration slice: first 20 calibration sequences. Calibration data only.
slice20 = truncated_workload(inputs.calibration, 20, 'cal20')
for label, wl in (('cal20', slice20), ('calFULL', inputs.calibration)):
    t0 = time.perf_counter()
    r = references(inputs, wl, CAPS)
    spec = dict(inputs.preregistration['stage1_reference']['winner_spec'])
    s1 = {int(x.capacity): int(x.misses)
          for x in simulate_causal_capacities(inputs.trace, wl, CAPS, spec, models)}
    print(f"--- {label} ({time.perf_counter()-t0:.0f}s) ---")
    print("  simple  :", r['simple'], r['simple_which'])
    print("  oracle  :", r['oracle'])
    report("  STAGE1 winner", s1, r, CAPS)
    json.dump({'refs': {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
                        for k, v in r.items() if k != 'simple_all'},
               'stage1': {str(k): v for k, v in s1.items()}},
              open(f'refs_{label}.json', 'w'), indent=1, default=str)
