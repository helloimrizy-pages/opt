"""Collect within-candidate-set rows under the frozen Stage 1 policy, calibration only."""
import time, numpy as np
from harness import *
from features import FeatureState, NAMES, CAP
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences
from residency_headroom.workloads import Workload, WorkloadSequence, calibration_frequency_scores

CAPS4 = (12, 16, 24, 32)

def sub(w, lo, hi, n):
    return Workload(n, w.regime, tuple(WorkloadSequence(s.source_sequence_id, i, s.segment_index,
        s.segment_label, s.domain) for i, s in enumerate(w.sequences[lo:hi])))

def collect_groups(trace, workload, models, static, stride=13, capacities=CAPS4):
    state = FeatureState(models, trace.num_layers, trace.num_experts, static)
    fut = build_future_occurrences(trace, workload, same_layer_indices(trace, workload))
    req_all = trace.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
    gate_all = trace.router_weights.astype(np.float64)
    L, E, C = trace.num_layers, trace.num_experts, len(capacities)
    res = np.zeros((C, L, E), bool); lu = np.full((L, E), -1, np.int64)
    pos = np.zeros(L, np.int64); mask = np.zeros(E, bool)
    hist = np.zeros((L, E)); M2 = np.stack([models.matrix(2, l) for l in range(L)])
    spares = [c - trace.top_k for c in capacities]
    X, Y, G, CAPC, LAY = [], [], [], [], []
    gid = 0; n = 0
    for _s, view in workload.iter_slices(trace):
        for idx in range(view.start, view.stop):
            o = int(trace.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
            true_d = fut.advance(o, p, r)
            sample = (n % stride == 0) and p >= 40
            feats = state.features(o, r, gate_all[idx], srt[idx], p) if sample else None
            cond = M2[o][srt[idx]].mean(axis=0)
            hist[o] *= 0.95; hist[o][r] += 0.05
            score = 0.5*cond + 0.5*hist[o]
            state.absorb(o, r, gate_all[idx], p)
            mask[r] = True; lu[o, r] = p
            for ci, sp in enumerate(spares):
                row = res[ci, o]; cand = np.flatnonzero(row & ~mask)
                if cand.size > sp:
                    if sample and cand.size >= 4:
                        X.append(feats[:, cand].T.astype(np.float32))
                        Y.append(np.minimum(true_d[cand], CAP).astype(np.float32))
                        G.append(np.full(cand.size, gid, np.int32))
                        CAPC.append(np.full(cand.size, capacities[ci], np.int16))
                        LAY.append(np.full(cand.size, o, np.int16))
                        gid += 1
                    ordr = np.lexsort((cand, -lu[o, cand], -score[cand]))
                    row.fill(False); row[r] = True; row[cand[ordr[:sp]]] = True
                else:
                    row[r] = True
            mask[r] = False; pos[o] += 1; n += 1
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(G),
            np.concatenate(CAPC), np.concatenate(LAY))

if __name__ == "__main__":
    inputs, models = load(); tr = inputs.trace
    calA = sub(inputs.calibration, 0, 40, 'calA')
    static = calibration_frequency_scores(tr, calA).astype(np.float64)
    t0 = time.perf_counter()
    X, Y, G, C, Lay = collect_groups(tr, calA, models, static)
    print(f"{X.shape[0]:,} candidate rows in {G.max()+1:,} groups, {X.shape[1]} features, "
          f"{time.perf_counter()-t0:.0f}s")
    print("target quantiles 10/25/50/75/90:", [int(np.quantile(Y,q)) for q in (.1,.25,.5,.75,.9)],
          f" mean {Y.mean():.2f}  saturated {100*(Y==CAP).mean():.1f}%")
    np.savez_compressed('calA_groups.npz', X=X, Y=Y, G=G, C=C, L=Lay, names=np.array(NAMES))
