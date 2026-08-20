import time, numpy as np
from harness import *
from features import FeatureState, NAMES, CAP
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences
from residency_headroom.workloads import Workload, WorkloadSequence, calibration_frequency_scores

def sub(w, lo, hi, n):
    return Workload(n, w.regime, tuple(WorkloadSequence(s.source_sequence_id, i, s.segment_index,
        s.segment_label, s.domain) for i, s in enumerate(w.sequences[lo:hi])))

def collect(trace, workload, models, static, stride=17, seed=0):
    """Sampled (feature, capped-next-use-distance) rows over non-requested experts."""
    state = FeatureState(models, trace.num_layers, trace.num_experts, static)
    fut = build_future_occurrences(trace, workload, same_layer_indices(trace, workload))
    req_all = trace.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    pos = np.zeros(trace.num_layers, dtype=np.int64)
    X, Y, W = [], [], []
    n = 0
    for _s, view in workload.iter_slices(trace):
        for idx in range(view.start, view.stop):
            o = int(trace.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
            true_d = fut.advance(o, p, r)
            if n % stride == 0 and p >= 40:
                f = state.features(o, r, gate_all[idx], srt[idx], p)
                keep = np.ones(trace.num_experts, bool); keep[r] = False
                X.append(f[:, keep].T.astype(np.float32))
                Y.append(np.minimum(true_d[keep], CAP).astype(np.float32))
                W.append(np.full(keep.sum(), o, np.int16))
            state.absorb(o, r, gate_all[idx], p)
            pos[o] += 1; n += 1
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W)

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    t0 = time.perf_counter()
    X, Y, Lay = collect(tr, calA, models, static, stride=17)
    print(f"collected {X.shape[0]:,} rows x {X.shape[1]} features in {time.perf_counter()-t0:.0f}s")
    np.savez_compressed('calA_dataset.npz', X=X, Y=Y, L=Lay, names=np.array(NAMES))
    print("target quantiles:", [int(np.quantile(Y, q)) for q in (.1,.25,.5,.75,.9)], "mean", Y.mean())
