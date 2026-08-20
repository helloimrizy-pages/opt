import numpy as np
from harness import *
from scorers import *
from race_stage1.models import same_layer_indices
from race_stage2.diagnostics import build_future_occurrences
from residency_headroom.workloads import Workload, WorkloadSequence

inputs, models = load(); tr = inputs.trace
def sub(w, lo, hi, n):
    return Workload(n, w.regime, tuple(WorkloadSequence(s.source_sequence_id, i, s.segment_index,
        s.segment_label, s.domain) for i, s in enumerate(w.sequences[lo:hi])))
calB = sub(inputs.calibration, 40, 80, 'calB')

streams = same_layer_indices(tr, calB)
fut = build_future_occurrences(tr, calB, streams)
V = np.stack([sum(EXPDIST_COEFF[i]*models.matrix(h,l) for i,h in enumerate(HORIZONS))
              for l in range(tr.num_layers)])
M = np.stack([np.stack([models.matrix(h,l) for h in HORIZONS]) for l in range(tr.num_layers)])

# Replay the Stage 1 winner and inspect the candidate sets it actually faces at cap 24.
CAPACITY, SP = 24, 16
req_all = tr.requested_expert_ids.astype(np.int64); srt = np.sort(req_all, axis=1)
E, L = tr.num_experts, tr.num_layers
res = np.zeros((L, E), bool); lu = np.full((L, E), -1, np.int64); pos = np.zeros(L, np.int64)
hist = np.zeros((L, E)); mask = np.zeros(E, bool)
dists, nsurv, spreads = [], [], []
markov_acc = np.zeros(6); expd_acc = 0.0; pairs = 0.0
for _s, view in calB.iter_slices(tr):
    for idx in range(view.start, view.stop):
        o = int(tr.layer_index[idx]); r = req_all[idx]; p = int(pos[o])
        true_d = fut.advance(o, p, r)
        cond = M[o][1][srt[idx]].mean(axis=0)
        hist[o] *= 0.95; hist[o][r] += 0.05
        score = 0.5*cond + 0.5*hist[o]
        mask[r] = True; lu[o, r] = p
        row = res[o]; cand = np.flatnonzero(row & ~mask)
        if cand.size > SP:
            d = np.minimum(true_d[cand], 33)
            dists.append(d)
            spreads.append(np.unique(d).size)
            better = d[:, None] < d[None, :]
            n = better.sum()
            if n:
                pairs += n
                sub_m = M[o][:, srt[idx], :].mean(axis=1)[:, cand]     # (6, k)
                for j in range(6):
                    df = sub_m[j][:, None] - sub_m[j][None, :]
                    markov_acc[j] += ((df > 0) & better).sum() + 0.5*((df == 0) & better).sum()
                v = V[o][srt[idx]].mean(axis=0)[cand]
                df = v[:, None] - v[None, :]
                expd_acc += ((df > 0) & better).sum() + 0.5*((df == 0) & better).sum()
            ordr = np.lexsort((cand, -lu[o, cand], -score[cand]))
            row.fill(False); row[r] = True; row[cand[ordr[:SP]]] = True
        else:
            row[r] = True
        mask[r] = False; pos[o] += 1
d = np.concatenate(dists)
print(f"candidate true capped next-use distance, {d.size:,} candidate observations at capacity 24")
print("  quantiles 1/5/10/25/50/75/90/95/99 :",
      [int(np.quantile(d, q)) for q in (.01,.05,.10,.25,.50,.75,.90,.95,.99)])
print(f"  mean {d.mean():.2f}   fraction saturated at 33: {100*(d==33).mean():.2f}%")
print(f"  distinct distances per candidate set: mean {np.mean(spreads):.1f} of ~24")
print("\n  survival S_h averaged over candidates (how much each horizon can discriminate):")
for j, h in enumerate(HORIZONS):
    print(f"    S_{h:<2d} mean={np.mean([np.mean(np.minimum(x,33)<=h) for x in dists]):.4f}"
          f"   pairwise ordering accuracy of MARKOV_H{h} = {100*markov_acc[j]/pairs:.2f}%")
print(f"    expdist_markov (the E[min(d,33)] functional)  = {100*expd_acc/pairs:.2f}%")
