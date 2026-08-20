# Optimizer-state memory under test-time adaptation — novelty / overlap audit

Status: **complete for the searchable literature reachable from this host on
2026-08-20**; internet access was available and every claim below was checked
against primary text (paper PDF/HTML or reference implementation source), not
against a search-result snippet. Where a source could not be verified in full
text, this is stated explicitly.

This audit precedes the Stage 1 experiments and is deliberately conservative:
absence of a matching paper in these searches is **not** treated as evidence of
novelty.

## The precise novelty question

> Has prior work systematically isolated Adam first- and second-moment carryover
> across continual test-time distribution shifts, using **matched model weights
> at the domain boundary**?

"Matched model weights" is the load-bearing clause. Any method that resets
weights *and* optimizer state together answers a different question, because the
weight change alone can explain the outcome.

## Search terms used

`test-time adaptation optimizer state` · `continual test-time adaptation Adam
moments` · `TTA momentum reset` · `CTTA optimizer reset` · `Adam state
distribution shift` · `optimizer state non-stationary adaptation` · `first
moment second moment test-time adaptation` · `momentum stale distribution
shift` · `"optimizer state" reset Tent entropy minimization "exp_avg"` ·
`continual test-time adaptation reset optimizer state momentum buffers restore`

## A. Model-parameter resets (weights, not optimizer state)

| Work | What is manipulated | Optimizer state? |
|---|---|---|
| **CoTTA** (Wang et al., CVPR 2022) — [code](https://github.com/qinenergy/cotta) | *Stochastic restoration*: a Bernoulli mask over **model parameters** `p` restores a fraction `rst_m=0.1` of weights to source each step; plus a teacher EMA and augmentation averaging | Only inside `reset()`, which restores model **and** optimizer state jointly. `exp_avg`/`exp_avg_sq` are never manipulated on their own. Verified in `cifar/cotta.py` lines 118–128, 154–168. |
| **RDumb** (Press et al., NeurIPS 2023, arXiv:2306.05401) | Periodic full reset `Θ_t → Θ_0` every `T = 1000` steps | Not discussed. RDumb/Tent/ETA/EATA in that paper use **SGD**, so there is no Adam second moment to carry. Verified in the paper PDF (2 occurrences of "optimizer", both about the update rule and the LR). |
| **ASR — "When and Where to Reset Matters for Long-Term Test-Time Adaptation"** (Lim, Hwang, Lee, ICLR 2026, arXiv:2603.03796) | *Adaptive and Selective Reset* of **model parameters**, triggered by predicted-class concentration, scoped to layers nearer the output; plus a Fisher-importance regulariser and an on-the-fly adaptation adjustment | **None.** Full-text search of the camera-ready PDF returns 0 hits for `optimizer`, `exp_avg`, `first moment`, `second moment`, `SGD`; the only "Adam" hits are the Paszke et al. PyTorch citation and an author name. The "momentum" hits are EMA coefficients for the concentration statistic and the Fisher/parameter EMAs — **not** Adam momentum. |
| **EBaR**, snapshot/buffer resets (ACM MM 2025) | Restores previously *optimized model snapshots* | Snapshot-level; not an isolated moment manipulation (not verified in full text). |

**Correction worth recording.** An automated summariser initially reported that
the ASR paper "focuses on resetting optimizer states … exp_avg and exp_avg_sq
and step counters … at domain boundaries". Reading the actual PDF shows this is
false. This is exactly the kind of hallucinated overlap that would have wrongly
killed or wrongly justified this stage, and it is the reason every claim here
was checked against primary text.

## B. Teacher-model momentum (not Adam momentum)

Teacher/student CTTA methods carry a momentum **coefficient** for an exponential
moving average over *network weights*. This is a different object from Adam's
`exp_avg`, which is an EMA over *gradients*.

- **CoTTA** teacher EMA, `mt_alpha = 0.99`.
- **Continual Momentum Filtering** (Lee & Chang, ICLR 2024) — Kalman filtering in
  *parameter space*; the "momentum" is over weights.
- **SloMo-Fast**, **RoTTA**, and similar slow/fast teacher schemes.

None of these manipulate optimizer first/second moments.

## C. Optimizer-state manipulation

| Work | First moment `m` | Second moment `v` | Step counter | Setting | Trigger |
|---|---|---|---|---|---|
| **Asadi, Fakoor & Sabach, "Resetting the Optimizer in Deep RL" (NeurIPS 2023, arXiv:2306.17833)** | **zeroed** | **zeroed** | reset (Adam's debiasing handles it) | **deep RL value-function optimization** (Rainbow/DQN, Atari) | at each new optimization problem, i.e. every target-network update. Network weights are *not* changed by the reset. |
| **Nikishin et al., "The Primacy Bias in Deep RL" (ICML 2022)** | zeroed for the reset subset | zeroed for the reset subset | reset | deep RL | resets a subset of **network weights** *and the corresponding Adam statistics* — i.e. optimizer state is reset jointly with weights |
| **Ellis et al., "Adam on Local Time / Adam-Rel" (arXiv:2412.17113)** | kept | kept | **reset to 0 per epoch (local timestep)** | deep RL, nonstationary targets | after target changes; motivated by the global timestep making post-target-change updates too large |
| **Official Tent example** (`DequanWang/tent`, `tent.py`) | restored | restored | restored | TTA, CIFAR-10-C | `Tent.reset()` restores `model.state_dict()` **and** `optimizer.state_dict()` together at every corruption boundary of the standard benchmark protocol |
| **SPAM** (Huang et al., ICLR 2025, arXiv:2501.06842) | zeroed | zeroed | not reset (a cosine LR warmup of `N=150` steps is applied instead) | **LLM pre-training** | periodic, every `ΔT = 500` steps; motivated by *gradient spikes*, not distribution shift |
| **Optimizer-state quantization staleness** (arXiv:2603.16731) | zeroed | zeroed (staleness *more pronounced* for `v`) | not manipulated | **LLM pre-training** (LLaMA 60M–350M on C4) | periodic (~1000/300/200 steps for BF16/FP8/FP4); staleness arises from *low-precision storage stalling updates*; the paper states it contains **no** distribution-shift or OOD analysis |
| **PALM** (Maharana et al., AAAI 2025, arXiv:2403.10650) | not touched | not touched | not touched | CTTA, CIFAR-10-C/100-C, ImageNet-C, Adam base optimizer | computes a *gradient*-magnitude sensitivity (KL to uniform) to set **per-layer learning rates**. Verified: 0 full-text hits for `exp_avg`, `optimizer state`, `reset`, `first moment`, `second moment`. It reads gradients, never Adam's internal buffers. |
| Practitioner folklore | zeroed | zeroed | usually zeroed | RL / LR-schedule jumps | "clear the Adam state when you make a large LR jump"; widely repeated, not systematically measured across distribution shifts |

### The closest prior work, stated plainly

**Asadi, Fakoor & Sabach (NeurIPS 2023) is the nearest neighbour of this study
and it is close.** Their framing — "information obtained in previous iterations
… can contaminate the moment estimates because the optimization landscape can
change arbitrarily from one iteration to the next" — is the same hypothesis this
stage tests. Their intervention (zero `m` and `v` at the landscape-change
boundary, leaving network weights untouched) is our `RESET_MV`/`FRESH_ADAM`.
And their mechanism diagnostic is *literally our H3*: they compute the cosine
similarity between Adam's first-moment estimate and the current gradient
immediately before and after the boundary, reporting a drop from **0.389 before
the target update to −0.002 after**, averaged over 12 Atari games, and conclude
that the carried moment "just contaminates Adam's moment accumulation
procedure".

Nothing in this Stage 1 study should be described as inventing that idea or that
diagnostic. What differs:

- **Their boundary is a target-network update**, an *internally generated* change
  in the loss function with the data distribution held fixed. Ours is an
  *external* change in the input distribution with the loss functional fixed
  (entropy). Whether the two behave alike is an empirical question, not a given.
- They reset `m` and `v` together. They do not separate first moment, second
  moment and bias-correction counter, so they cannot say which component
  carries the effect.
- They have **no stationary control**: nothing in that paper distinguishes
  "resetting helps because the landscape changed" from "resetting helps anyway".
- Deep RL value iteration is heavily non-stationary by construction and involves
  full-network optimization; Tent adapts 17,952 BatchNorm affine scalars with a
  fixed entropy objective. The two regimes are not interchangeable.

## The overlap that matters most

The most important overlap is **not** a paper but the **standard benchmark
protocol itself**: the official Tent CIFAR-10-C example calls `model.reset()`
between corruption types, which restores model weights *and* Adam state. That
means the canonical 18.6%-error Tent number is already produced by an
optimizer-state reset — but a *confounded* one, because the weights are reset in
the same operation. No published work we could find separates the two.

## What appears genuinely unaddressed

Combining the three categories:

1. Model-parameter resets (A) change weights, so they cannot attribute anything
   to optimizer state.
2. Teacher momentum (B) is a weight EMA and is unrelated to Adam's buffers.
3. Optimizer-state resets (C) exist and, in deep RL, the core hypothesis and the
   `cos(m, g)` diagnostic have **already been published** (Asadi et al. 2023).
   Within *test-time adaptation*, however, every instance we verified either
   resets state jointly with weights (Tent's `reset()`, Nikishin et al.), targets
   LLM pre-training pathologies — gradient spikes (SPAM) or low-precision
   stalling (arXiv:2603.16731) — with no distribution-shift analysis, or reads
   gradients rather than optimizer state (PALM).

So the honest statement of the gap is narrow and specific. We found no work that,
**under a genuine input-distribution shift**, holds model parameters
bitwise-identical at the boundary and varies `exp_avg`, `exp_avg_sq` and the Adam
step counter *separately*, across a set of continual test-time transitions, with
a **matched stationary control** that distinguishes shift-induced staleness from
a generic benefit of resetting. Even that framing is a refinement of Asadi et
al.'s question rather than a new question, and any future write-up must position
against them explicitly rather than against the TTA literature alone.

Measuring it is worthwhile *whichever way the result comes out*, including the
reverse result that carried state helps — and a null result here would be
informative precisely because the RL result is positive.

## Caveats

- Literature search is inherently incomplete; several 2026 preprints were only
  reachable as abstracts.
- Workshop papers, non-English venues and unreleased code were not searchable.
- Some CTTA repositories re-instantiate the optimizer as an implementation detail
  of a weight reset without documenting it. Such silent optimizer-state resets
  would be additional undocumented overlap, and cannot be excluded by literature
  search alone.
- Novelty is **not** claimed on the basis of this audit. The audit's purpose is
  to make sure Stage 1 measures something not already measured, and to name the
  works that a later write-up must position against.

## Sources

- Tent — https://openreview.net/forum?id=uXl3bZLkr3c · code https://github.com/DequanWang/tent
- CoTTA — https://github.com/qinenergy/cotta
- RDumb / CCC — https://arxiv.org/abs/2306.05401
- ASR, "When and Where to Reset Matters for Long-Term TTA" — https://arxiv.org/pdf/2603.03796
- PALM — https://arxiv.org/abs/2403.10650
- SPAM — https://arxiv.org/abs/2501.06842
- Optimizer-state quantization / state staleness — https://arxiv.org/html/2603.16731
- Continual Momentum Filtering — https://openreview.net/forum?id=BllUWdpIOA
- Resetting the Optimizer in Deep RL (Asadi, Fakoor, Sabach) — https://arxiv.org/abs/2306.17833
- Adam on Local Time / Adam-Rel (Ellis et al.) — https://arxiv.org/abs/2412.17113
- The Primacy Bias in Deep RL (Nikishin et al., ICML 2022) — https://arxiv.org/abs/2205.07802
- EBaR — https://dl.acm.org/doi/10.1145/3746027.3754907
