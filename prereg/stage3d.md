# Stage 3D preregistration: selection-headroom diagnostics

Frozen 2026-08-16, before any Stage 3D loss was computed. Every threshold in
Step 5 below was written into `src/expert_analysis/stage3d_diagnostics.py`
before the first evaluation ran. Nothing in this file may be edited after
results exist.

## 1. What this stage is and why it is a new stage

Stages 2A, 2B and 2C each tried a different way to predict which experts to
keep at 8 bits under a 20% memory budget. All three failed to beat baseline.
Two explanations remain:

1. The predictors were bad, and per-expert sensitivity really is heterogeneous.
2. Per-expert sensitivity is near-uniform in this model, so the objective is
   flat and no selection rule can win.

Stage 3D measures which. It replaces predicted per-expert damage with directly
measured sensitivity at three granularities: expert sets (Sweep A), whole
layers (Sweep B), and routers (Sweep C).

**This is a new stage, not a Stage 2C variant.** Stage 2C searched for a better
weighting of expert scores. Stage 3D does not reweight, refit, or rescore
anything. It changes the decision variable: instead of asking which experts a
rule should pick, it asks whether the choice of experts moves the loss at all
relative to the spread produced by choosing at random. No Stage 2A, 2B or 2C
outcome enters any Stage 3D quantity, and every frozen negative decision from
those stages is preserved unchanged.

Stage 3D is diagnosis only. It does not implement a bit-allocation optimizer,
does not run the 1024-expert leave-one-in sweep, does not add domains, does not
change domain weightings, and does not change the budget.

## 2. Naming, and its relationship to the unrun measured-damage Stage 3

The repository already contains a committed but never executed stage called
Stage 3, *measured expert-damage preservation*
(`src/expert_analysis/measured_damage*.py`, commit 69fc93e). To avoid two
things named Stage 3, this stage is **Stage 3D**, with artifacts under
`results/stage3d_diagnostics/` and `prereg/stage3d.md`.

The measured-damage Stage 3 assumes per-expert damage is heterogeneous enough
to be worth measuring 1024 times. Sweep A tests that assumption directly, so
Stage 3D is a gate on it, not a stage sitting alongside it. Recorded here and
in `EXPERIMENT_STATUS.md`:

- The measured-damage Stage 3 is **on hold pending Stage 3D Sweep A**. It is
  not superseded; that word would imply the answer is already known.
- Its go condition is Sweep A returning `HEADROOM` under the Step 5 rule below.
- If Sweep A returns `FLAT`, the negative result is the paper, not a blocker on
  the way to one.
- Stage 3D does **not** consume split seed 46. It builds no data split. Seed 46
  remains available to the measured-damage Stage 3, subject to point 4 below.

## 3. Development seed and seed hygiene

**Stage 3D uses seeds 46 through 65 to select protection sets.** These twenty
seeds have not been used anywhere in Stages 1, 2A, 2B or 2C. The complete prior
seed inventory, verified against the source and the frozen artifacts:

| seed | prior use |
|---|---|
| 42 | global run seed; the seed-42 candidate pool inspected during controlled selection |
| 43 | Stage 2B development split |
| 44 | final split, built and deliberately never evaluated |
| 45 | Stage 2C development split |
| 46 | claimed by the unrun measured-damage Stage 3 as its split seed; never used to produce any evaluation |
| 1001–1005 | Stage 2B random protection allocations |
| 20260815 | calibration subset selection and bootstrap replicates |
| 11, 23, 37, 53 | fixed per-domain offsets combined with the seeds above |
| ≥ 61,997,031 | assignment-verification sample seeds derived from allocation hashes |

Seed 46 appears twice in that table and here. The roles do not collide: in the
measured-damage Stage 3 it seeds a dataset shuffle, and in Stage 3D it seeds
`numpy.random.default_rng(46).permutation(1024)` over the expert grid. Stage 3D
builds no split, so it cannot consume seed 46 in the split sense.

## 4. Evaluation set

**A fresh disjoint split is not buildable, and this is recorded as a limitation
rather than worked around silently.** Running the frozen Stage 3 split builder
aborts on the coding domain. Remaining eligible examples per domain, after
excluding the frozen controlled 100, seed 43, seed 44 and seed 45:

| domain | dataset | rows loaded | eligible at ≥65 content tokens | already used | free |
|---|---|---|---|---|---|
| general | Salesforce/wikitext, wikitext-103-raw-v1 test | 2880 | 1601 | 300 | 1301 |
| math | openai/gsm8k, main test | 1319 | 404 | 300 | 104 |
| coding | google-research-datasets/mbpp, full test | 500 | 332 | 300 | **32** |
| reasoning | allenai/ai2_arc, ARC-Challenge test | 1172 | 498 | 300 | 198 |

A balanced fresh split now caps at 32 examples per domain, against the 50 every
prior stage used. The measured-damage Stage 3 would fail at the same line.

**Stage 3D therefore evaluates on the frozen Stage 2B seed-43 development split
concatenated with the frozen Stage 2C seed-45 development split**, seed-43 rows
first: 100 examples per domain, 400 examples, 25,600 measured tokens. Both
source splits verify against their own manifests as they load, and the build
proves they are mutually disjoint and contain no seed-44 row.

Frozen input hashes, `results/stage3d_diagnostics/evaluation_set/evaluation_set_manifest.json`:

| domain | examples | sequence length | measured tokens each | `input_ids` SHA-256 prefix |
|---|---|---|---|---|
| general | 100 | 68 | 64 | `ef3ddf2f98dddbc4` |
| math | 100 | 68 | 64 | `b2f530cc52f0b7db` |
| coding | 100 | 68 | 64 | `5a9982563e226b3f` |
| reasoning | 100 | 68 | 64 | `0ce356ed1ec9a9ae` |

Seeds 43 and 45 were observed by Stage 2B and Stage 2C. That matters for
selection and tuning, and Stage 3D does neither. Its random sets come from
seeded permutations over the expert grid; its two deliberate sets come from
Stage 2B calibration routing counts measured on the frozen controlled
100-per-domain set, which is disjoint from both. No Stage 3D quantity depends
on any seed-43 or seed-45 outcome.

**The seed-44 final reserve is not touched by this stage.**

### Calibration versus evaluation

There is no fitting on data anywhere in this pipeline. Quantization is
round-to-nearest: every scale is the per-group absolute maximum of the weight
tensor itself, so no scale is fit on calibration data or on anything else.

The calibration set is used only to rank experts by routing frequency for
Sweep A's two deliberate sets. It is the frozen Stage 2B subset: 25 examples
per domain, 100 total, selected with seed 20260815 from the frozen controlled
100-per-domain set. It is disjoint from the evaluation set defined above.

## 5. Model and quantization configuration

| property | value |
|---|---|
| checkpoint | `allenai/OLMoE-1B-7B-0924` |
| revision | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| weight dtype | **bfloat16**, not fp16 |
| MoE layers | 16 |
| experts per layer | 64, 1024 total |
| routing | top-8, `norm_topk_prob` false |
| device | CUDA, NVIDIA A40, matching every prior stage |
| batch size | 1 |
| global seed | 42 |

Quantizer: `symmetric_groupwise_qdq` in `src/expert_analysis/quantization.py`,
unchanged from Stage 1 and reused by Stages 2A through 2C.

| property | value |
|---|---|
| scheme | symmetric, no zero point, `qmax = 2^(bits-1) - 1` |
| rounding | round-to-nearest; not GPTQ, no Hessian, no data |
| granularity | group-wise along the last dimension, the input-feature dimension |
| group size | 128 |
| scale storage | computed in fp32, rounded to fp16, expanded to fp32 for the round-and-multiply |
| form | fake quantization: quantize then dequantize back into bfloat16 storage, not packed weights |
| bit widths used | 3, 4, 8, and 16 as an exact identity |
| scope | expert FFN weight matrices only |

Per expert that is `experts.gate_up_proj` sliced to `[2048, 2048]` and
`experts.down_proj` sliced to `[2048, 1024]`: 6,291,456 weights per expert,
6,442,450,944 across all 1024.

**Left at bfloat16 throughout, except where Sweep C says otherwise:** MoE router
weights, attention projections, embeddings, `lm_head`, and all normalization
parameters. `MixedPrecisionExpertManager` SHA-256 fingerprints every non-expert
parameter and re-verifies them after every apply and restore, so any accidental
change is a hard error.

## 6. Loss

Teacher-forced next-token cross entropy, computed on logits cast to fp32, at
the 64 measured positions of each 68-token example. Positions 0 through 2 are
the neutral prefix `"Input:\n"` (token ids 8982, 27, 187) and are not measured.

Padding is not involved: every example is exactly 68 tokens, the attention mask
is all ones, and every example contributes exactly 64 measured tokens.

Per-domain aggregation is the mean over examples of each example's mean token
cross entropy, identical to Stages 2B and 2C. Because token counts are uniform,
this equals the token-weighted mean.

### Two worst-domain definitions, both recorded, one deciding

Every table and every JSONL record carries both:

- **`worst_domain_relative`** — the maximum over the four domains of
  `(loss_domain - bf16_loss_domain) / bf16_loss_domain`. This is the Stage 1
  through 2C definition.
- **`worst_domain_raw`** — the maximum over the four domains of the raw mean
  token loss.

**Step 5 is applied to `worst_domain_relative` only.** The raw definition is
reported because the request named it, and it is recorded here so that choice
cannot be made after seeing numbers. The four domains differ in baseline loss
by more than a factor of two, so the raw maximum is pinned to whichever domain
is hardest in absolute terms and barely moves across protection sets.

## 7. Sweeps

### Sweep A, expert-level flatness — 36 evaluations

The Stage 2C budget: 20% of the bytes it would cost to raise every expert from
base precision to 8 bits. Both regimes protect exactly 204 of 1024 experts,
because all experts are the same size.

| regime | base bits | protected bits | budget bytes | total increment | protected experts | random sets |
|---|---|---|---|---|---|---|
| 4to8 | 4 | 8 | 644,245,094 | 3,221,225,472 | 204 | 20 (seeds 46–65) |
| 3to8 | 3 | 8 | 805,306,368 | 4,026,531,840 | 204 | 10 (seeds 46–55) |

Plus, in each regime: the 204 most frequently routed experts, the 204 least
frequently routed experts, and one run with no protection at all.

**The twenty random sets are selected once and reused by both arms.** Seeds 46
through 65 pick expert indices under the 4to8 budget; each set is then checked
against the 3to8 budget and reused unchanged, so the two arms compare the same
experts rather than two independent draws. The 3to8 arm evaluates the first ten
of those same sets. The build verifies set equality by SHA-256 and refuses to
proceed otherwise.

Routing frequency is the total routing count per `(layer, expert)` over the
frozen Stage 2B calibration set, summed across all four domains. Ties break on
`(layer, expert)` ascending in both directions. On the frozen counts the top
204 experts carry 38.0% of all routes and the bottom 204 carry 8.1%.

### Sweep B, layer sensitivity — 16 evaluations

For each layer 0 through 15: all 64 experts of that layer at 4 bits, every
other parameter at bfloat16. The bfloat16 baseline comes from the Step 2
harness and is not re-measured.

### Sweep C, router protection — 1 new evaluation

Step 0 established that the pipeline already excludes routers, so as specified
only the quantized-router state is new, and it is labelled a diagnostic on
baseline strength rather than a protection comparison.

The run is every expert at 4 bits with every MoE router weight also at 4 bits,
same quantizer, same group size. Its routers-at-bfloat16 comparison point is
Sweep A's `a_4to8_no_protection` run, which is the identical expert assignment;
the reporter verifies the two share a bit-matrix hash before comparing them.

Router tensors are the 16 `model.layers.*.mlp.gate.weight` parameters, shape
`[64, 2048]`, no bias: 2,097,152 parameters, about 0.030% of the model. The
expert container's `gate_up_proj` also contains the word gate but is expert
weight and is quantized separately. The 3-bit router variant is not run.

Sweep A runs first and is reported before Sweep B starts.

## 8. Correctness harness, before any sweep

All four checks must pass. `scripts/run_stage3d_harness.py`.

1. Model in eval mode, no dropout, `torch.use_deterministic_algorithms(True)`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled, fixed seed, fixed data
   order.
2. The bfloat16 baseline evaluated twice must agree **bitwise**; per-domain
   losses agreeing to fewer than 6 decimal places stops the run. Reports
   per-domain losses and token counts.
3. Layer 0 expert 0 quantized to 4 bits, evaluated, restored from a cached copy
   of its original tensors, evaluated again. The restored losses must equal the
   baseline bitwise. The model is never reloaded from disk between runs.
4. Two Stage 1 single-expert measurements reproduced on the frozen controlled
   100-per-domain set that Stage 1 used: layer 11 expert 27 and layer 12 expert
   43 at 4 bits, the two largest measured Stage 1 effects. Default tolerance is
   exact agreement. Any disagreement stops the run and reports its size.

Peak device memory and wall clock for one evaluation pass are reported.

## 9. Step 5 decision thresholds

Copied verbatim from `DECISION_THRESHOLDS` in
`src/expert_analysis/stage3d_diagnostics.py`, where they were fixed before any
result existed. They are applied mechanically and are not edited afterwards.

### Sweep A

Let `sd_random` be the **sample standard deviation, ddof = 1**, of
`worst_domain_relative` across that arm's random sets, and let `gap` be the
`worst_domain_relative` of the least-routed set minus that of the most-routed
set.

- `gap` greater than **4 × `sd_random`** → `HEADROOM`. Expert-level selection
  has headroom, and the full 1024-expert leave-one-in sweep becomes worth
  running.
- `gap` less than **2 × `sd_random`** → `FLAT`. The objective is flat at expert
  granularity; stop pursuing expert selection.
- Between 2 and 4 → `INCONCLUSIVE`. Report and ask.

**The 4to8 arm carries the decision.** The 3to8 arm is a secondary check on
whether the picture looks qualitatively different where damage is larger. It
can escalate a `FLAT` primary outcome to `INCONCLUSIVE` when it returns
`HEADROOM`, which then requires running its full 20 random sets before any
conclusion is drawn. It can never on its own authorize the 1024-expert sweep.
Precedence, applied in order:

| 4to8 arm | 3to8 arm | Sweep A outcome |
|---|---|---|
| `HEADROOM` | any | `HEADROOM` |
| `FLAT` | `HEADROOM` | `INCONCLUSIVE`, run the full 20 at 3 bits |
| `FLAT` | `FLAT` or `INCONCLUSIVE` | `FLAT` |
| `INCONCLUSIVE` | any | `INCONCLUSIVE` |

A negative `gap`, meaning the least-routed set degrades *less* than the
most-routed set, satisfies the `FLAT` rule as written and is decided that way.
It is flagged in the record and reported, not investigated further.

### Sweep B

Let `ratio` be the largest per-layer `worst_domain_relative` increase divided
by the smallest.

- `ratio` at or above **3.0** → `HEADROOM`. Layer-wise bit allocation has
  headroom.
- `ratio` at or below **1.2** → `DROP`. Drop layer-wise allocation.
- Strictly between → `INCONCLUSIVE`. Report and ask.
- If the smallest per-layer increase is not strictly positive the ratio is
  undefined; the outcome is `INCONCLUSIVE` and the raw values are reported.

### Sweep C

No threshold. Record the `worst_domain_relative` difference between quantized
and bfloat16 routers alongside the router parameter count and its share of
deployed memory.

## 10. Deliverables

- `prereg/stage3d.md`, this file, committed before the first sweep runs.
- `results/stage3d_diagnostics/stage3d_a.jsonl`, `stage3d_b.jsonl`,
  `stage3d_c.jsonl`. One record per run, appended and fsynced individually, so
  a crash loses at most the run in flight. Each record carries the git commit
  hash, the seed, the configuration hash, the run description, per-domain loss,
  both worst-domain definitions, and wall clock.
- `stage3d_a.csv`, `stage3d_b.csv`, `stage3d_c.csv`, one table per sweep in
  plain text and CSV.
- `stage3d_decision.json` and `SUMMARY.md`, applying Step 5 and stating which
  outcome each sweep landed in, with no interpretation beyond the thresholds.

## 11. What this stage will not do

- Implement the bit-allocation optimizer.
- Run the 1024-expert leave-one-in sweep. That decision depends on Sweep A.
- Add domains, change domain weightings, or change the budget.
- Adjust any threshold after seeing numbers.
- Reuse any seed from Stages 1, 2A, 2B or 2C.
- Evaluate the seed-44 final split.
- Investigate a surprising result further without asking.
