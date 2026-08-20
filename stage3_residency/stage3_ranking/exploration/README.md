# Stage 3 exploration — calibration-only

This directory holds the exploration that produced the Stage 3 design. It is
**supporting material, not part of the frozen evaluation path**: nothing here is
imported by `src/race_stage3/`, nothing here ran during the frozen evaluation, and
none of it is covered by `reports/final_archive_manifest.json`, which seals the frozen
path only.

It is committed because `reports/race_stage3_report.md` section H makes a load-bearing
claim — that pairwise ranking accuracy over causal routing history saturates near 69%
regardless of model class, training-set size or model capacity — and that claim has to
be reproducible.

## Data discipline

Every script reads the frozen Stage 0 routing trace and the frozen Stage 0 **calibration**
workload only. To keep the exploration itself honest, the 80 calibration sequences were
split again:

- `calA` = calibration sequences 0–39, used to fit anything new;
- `calB` = calibration sequences 40–79, used to measure.

`reverse.py` repeats the headline measurement with the split direction swapped. No
evaluation sequence is read by any file in this directory.

## What each script does

| Script | Question |
| --- | --- |
| `harness.py` | Shared loader, a generic scorer-driven replay, and the calibration reference costs |
| `refs.py` | Strongest-simple, oracle and Stage 1 winner costs on the calibration paths |
| `scorers.py` | The four principled estimators that were tried first and lost |
| `run1.py` | Deploys those four: expected-capped-distance, noisy-OR, hazard, blends |
| `diag.py` | Candidate next-use distance distribution and per-horizon discriminative power |
| `features.py`, `features2.py`, `features3.py` | The 19-, 37- and 45-feature causal sets |
| `collect.py`, `collect2.py` | Feature/target collection, over all experts vs over real candidate sets |
| `fit.py` | Pairwise-logistic ranking fit and pairwise-accuracy scoring |
| `deploy.py`, `run2.py`, `run3.py` | Deploying the learned rankers as capacity policies |
| `run4.py` | Per-capacity models retrained on the deployed policy's own trajectory |
| `run5.py` | Does truncating the target to the slot-residence horizon help? (no) |
| `run6.py` | Does boundary-weighted training help? (no, monotonically worse) |
| `ceiling.py` | Is the plateau a feature limit or a linear-form limit? |
| `wall.py` | Does 13x data and 8x model capacity break the plateau? (no) |
| `frontier.py` | Non-causal: what ranking accuracy would each gap level require? |
| `reverse.py` | Same result with the calA/calB split direction swapped |

## Running

```bash
stage3_residency/stage3_ranking/exploration/run_exploration.sh wall.py
stage3_residency/stage3_ranking/exploration/run_exploration.sh frontier.py
```

`wall.py`, `ceiling.py` and `run4.py` need scikit-learn; the frozen Stage 3 path does
not, because the shipped model is linear.

Recorded outputs and their interpretation are in
`../reports/race_stage3_exploration_notes.md`.
