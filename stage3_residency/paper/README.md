# Cross-stage artifacts for the write-up

Everything here is **derived** from the sealed Stage 0/1/2/3 archives. It measures no
new policy behaviour and changes no frozen artifact. It exists because a paper needs
things that no single stage produces: one joined results table, one figure covering the
whole arc, head-to-head comparisons between stages, and an accounting of what the
policy itself costs to run.

Rebuild with:

```bash
stage3_residency/paper/build_paper_pack.sh
```

## Contents

| File | What it is |
| --- | --- |
| `headline_results.json` | Every stage's suite cost and gap closure at all five capacities, plus verdicts and archive hashes |
| `results_all_stages.csv` | Suite and per-regime costs, gap closure and normalized cost for Stage 0/1/2/3 and the like-for-like Stage 3 variant |
| `results_by_workload.csv` | The same broken out over all ten frozen workloads |
| `stage_comparisons.json` | Paired conditional bootstrap for Stage 3 vs Stage 1, **Stage 3 vs Stage 2**, and Stage 2 vs Stage 1 |
| `policy_overhead.json` | Arithmetic and memory cost of running the Stage 3 scorer |
| `stage3_per_layer_misses.csv` | Per-MoE-layer miss counts, recorded during evaluation but never analyzed |
| `figure_cross_stage.png` / `.pdf` | Two-panel main figure: normalized cost, and oracle-gap recovery vs spare residency |
| `requirements-lock.txt` | The exact 56-package environment, pinned |

## The three numbers a reviewer will ask for first

**Stage 3 beats Stage 2 head-to-head**, which no stage report established because each
stage only compared itself to Stage 1:

| Capacity | improvement over Stage 2 | 95% CI | sequences improved |
| ---: | ---: | --- | ---: |
| 12 | +3.87% | [3.69, 4.06] | 1344 / 1380 |
| 16 | +5.46% | [5.20, 5.74] | 1339 / 1380 |
| 24 | +6.73% | [6.37, 7.14] | 1307 / 1380 |
| 32 | +7.03% | [6.54, 7.53] | 1256 / 1380 |

**The policy is nearly free.** No stage report addressed this, and it is the obvious
objection to any caching policy: does scoring cost more than the transfers it saves?

- 13,248 arithmetic operations per same-layer event, 211,968 per generated token across
  16 MoE layers;
- that is `8.15e-5` of one decode forward pass, taking OLMoE-1B-7B at roughly 1.3B
  active parameters and two FLOPs per active parameter;
- policy state is under 4 MB (3.1 MB of it the Markov tables) against 12.9 GB of expert
  weights;
- the policy avoids **61 to 105 bytes of expert transfer per FLOP it spends**, depending
  on capacity.

The measured Python replay throughput of ~145 microseconds per same-layer event is
interpreter-dominated and is **not** the deployable cost. It is reported in
`policy_overhead.json` only as an upper bound on a research implementation.

**Where the gain comes from.** Panel (b) of the figure plots Stage 3 restricted to the
information Stage 1 had (dashed) beside the full model, so the reader can see the split
between better use of the same signals and the one new signal (decode-request
boundaries) without hunting through an ablation table.

---

## What is still missing, and why

These are genuine gaps. None of them can be closed from the sealed archives.

### 1. No hardware anchor — the study is in transfer counts, not time

`stage3_residency/reports/hardware_transfer_calibration_status.json` records
`available: false`: CUDA was not reachable when Stage 0 ran, so the host-device expert
transfer cost was never measured. Every number in every stage is a **count** of expert
transfers. `policy_overhead.json` converts counts to bytes, but nothing converts bytes
to milliseconds.

To close it, run `stage3_residency/scripts/measure_transfer.py` on the lab GPU host.
Until then the paper must be framed as a simulation study of transfer counts, which is
what every stage report already says, and must not report a speedup.

### 2. No related work

Nothing in this repository positions the work against prior MoE offloading and expert
caching systems. I did not write one, because doing so from memory means inventing
citations, and a fabricated reference is worse than an absent section.

The online-learning citations in
`stage3_residency/stage2_race/reports/race_stage2_theory_notes.md`
(Littlestone–Warmuth, Freund–Schapire, Vovk, Cesa-Bianchi–Lugosi, Herbster–Warmuth,
Weinberger–Ordentlich, Joulani et al., Belady) are ones I am confident in and can be
reused directly. The systems side needs a real literature search.

### 3. One model, one trace — the top threat to validity

All four stages use a single frozen trace: OLMoE-1B-7B-0924, greedy decoding, batch
size 1, at most 128 generated tokens, 100 prompts each from wikitext / MBPP / GSM8K /
ARC-Challenge. There is no evidence about another checkpoint, another decoding
strategy, or another domain mix. Generating a second trace needs the GPU host.

### 4. Batch size 1 is baked into the cache model

The whole formulation assumes exactly one atomic top-8 request per layer per decode
step. Under batching, a layer step requests the union of several tokens' experts, which
changes both the request size and the reuse structure. This is a scope boundary the
paper has to state explicitly; it is not a limitation any stage measured around.

### 5. No literature baseline

Stage 3 is compared against LRU, LFU, decayed LFU, a static hot set, the Stage 1
predictor and the Belady oracle — all implemented here. There is no comparison against
a published expert-caching or MoE-offloading policy. Whether that matters depends on
what section 2 ends up containing.

### 6. Calibration-size sensitivity is unmeasured

Stage 3 fits on all 80 calibration sequences. Nobody measured how performance degrades
with 40, 20 or 10, which a reviewer may reasonably ask since it bears on deployment
cost. This one is cheap to add — roughly half an hour of compute on the existing
calibration path, no new data — and it was simply never run.

### 7. Naming wart

The virtualenv used for Stage 1, 2 **and** 3 is called `.venv-race-stage2`. Harmless,
but confusing if a reader follows the commands literally. `requirements-lock.txt` is the
authoritative record of what was installed.
