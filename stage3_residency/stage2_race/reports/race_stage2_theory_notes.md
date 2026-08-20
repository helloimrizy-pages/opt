# RACE Stage 2 theory notes

These notes state precisely what is and is not proved about the Stage 2 algorithm as
implemented. Section 4 contains a complete proof for the exact delayed update used in
the code. Section 5 lists the gaps that would have to be closed before any statement
about expert-transfer cost could be called a theorem.

**No formal regret theorem is claimed for the ranking loss of the combined RACE score,
and none is claimed for expert-transfer cost.** The only proved statement is Theorem 1,
which concerns the linear mixture loss against the best fixed adviser on the realized
loss sequence.

---

## 1. Setting

Fix one RACE policy instance: one cache capacity `B` and one MoE layer `l`. Write
`K` for the adviser-pool size (9 in the primary pool) and `H = 32` for the capped
future-reuse horizon in same-layer events.

The instance produces a chronological sequence of *learning examples*. Example `t`
is created at same-layer decision position `p_t`, at an event where an eviction was
actually required and at least one retention slot remained. Its loss vector
`ℓ_t ∈ [0, 1]^K` is the pairwise future-use ranking loss of section 15 of the Stage 2
specification (or the cost-sensitive variant of section 16). Both are bounded in
`[0, 1]` by construction, and examples with no comparable candidate pair are dropped
from the sequence entirely rather than being assigned an arbitrary loss.

Decision positions are strictly increasing, `p_1 < p_2 < ... < p_T`, and at most one
example exists per (capacity, layer, same-layer event).

## 2. The implemented update

`ℓ_t` becomes observable only once the `H` same-layer events after `p_t` have been
seen. The implementation resolves example `t` at same-layer position `p_t + H`,
immediately after that event's atomic request has been observed and *before* that
event's own retention decision. So the weights used to act at position `p` are

```
w(p) ∝ exp( -η · Σ_{ t : p_t + H ≤ p } ℓ_t )
```

normalized to the simplex, which is exactly `w_j ← w_j exp(-η ℓ_j)` applied once per
resolved example, carried in log space. Write `w_t := w(p_t)` for the weight vector
deployed when example `t` was created.

Define the not-yet-resolved set at time `t`:

```
D_t = { s < t : p_s > p_t - H }.
```

Since decision positions are distinct integers and there are at most `H - 1` integers
strictly between `p_t - H` and `p_t`,

```
|D_t| ≤ H - 1.                                                            (1)
```

This is the only structural property of the delay that the proof uses. In particular
the bound holds even though the examples are a *subsequence* of events (eviction-free
events produce no example) and even though the number of intervening events varies.

## 3. Prediction with expert advice

The construction is Hedge / multiplicative weights (Littlestone–Warmuth,
Freund–Schapire, Vovk) applied to advisers rather than to actions: RACE does not learn
the future-use label, it learns which causal adviser to trust. With `ℓ_t ∈ [0, 1]^K`
and no delay, the classical bound is

```
Σ_{t=1}^T ⟨ŵ_t, ℓ_t⟩ - min_j Σ_{t=1}^T ℓ_{t,j} ≤ ln K / η + η T / 8         (2)
```

where `ŵ_t ∝ exp(-η Σ_{s<t} ℓ_s)` is the *undelayed* iterate. (2) is the standard
exponential-potential argument with Hoeffding's lemma; it holds pathwise, hence also
against an adaptive adversary that chooses `ℓ_t` after seeing `w_1, …, w_t`.

## 4. Theorem 1 — delayed-Hedge regret for the implemented update

> **Theorem 1.** Let `ℓ_t ∈ [0,1]^K` for `t = 1, …, T` be the realized loss vectors of
> one RACE policy instance, let `w_t` be the weight vector the implementation deployed
> at example `t`, and let `M := η (H - 1)`. Then
>
> ```
> Σ_{t=1}^T ⟨w_t, ℓ_t⟩ - min_j Σ_{t=1}^T ℓ_{t,j} ≤ ln K / η + η T / 8 + T (e^{M} - 1).
> ```
>
> Consequently, if `M ≤ 1` then `e^M - 1 ≤ e·M`, and choosing
> `η = sqrt( ln K / (T (1/8 + e(H-1))) )` gives
> `Regret ≤ 2 sqrt( ln K · T · (1/8 + e(H-1)) ) = O( sqrt(H T ln K) )`.

**Proof.** Write `A_t := Σ_{t' : p_{t'} + H ≤ p_t} ℓ_{t'}` for the cumulative loss the
implementation had actually resolved when it acted at example `t`, and
`B_t := Σ_{s<t} ℓ_s` for the cumulative loss an undelayed Hedge would have used. By
construction `B_t = A_t + Δ_t` with `Δ_t := Σ_{s ∈ D_t} ℓ_s`, so componentwise
`0 ≤ Δ_{t,i} ≤ |D_t| ≤ H - 1` by (1). Then `w_t ∝ exp(-η A_t)` and
`ŵ_t ∝ exp(-η B_t) = exp(-η A_t - η Δ_t)`.

*Perturbation lemma.* Let `p ∝ exp(-c)` and `q ∝ exp(-c - δ)` on `{1,…,K}` with
`0 ≤ δ_i ≤ M` for every `i`. Then `‖p - q‖_1 ≤ e^{M} - 1`.

  Indeed `q_i = p_i e^{-δ_i} / Z` with `Z = Σ_j p_j e^{-δ_j} ∈ [e^{-M}, 1]`, hence
  `q_i / p_i = e^{-δ_i}/Z ∈ [e^{-M}, e^{M}]` and
  `‖p - q‖_1 = Σ_i p_i |1 - q_i/p_i| ≤ max_i |1 - q_i/p_i| ≤ max(1 - e^{-M}, e^{M} - 1)
  = e^{M} - 1`, using `Σ_i p_i = 1`. ∎

Applying the lemma with `δ = η Δ_t` and `M = η(H-1)` gives `‖w_t - ŵ_t‖_1 ≤ e^{M} - 1`,
and since `ℓ_t ∈ [0,1]^K`, Hölder gives

```
⟨w_t, ℓ_t⟩ - ⟨ŵ_t, ℓ_t⟩ ≤ ‖w_t - ŵ_t‖_1 · ‖ℓ_t‖_∞ ≤ e^{M} - 1.
```

Summing over `t` and adding (2):

```
Σ_t ⟨w_t, ℓ_t⟩ - min_j Σ_t ℓ_{t,j}
  = Σ_t ⟨w_t - ŵ_t, ℓ_t⟩ + ( Σ_t ⟨ŵ_t, ℓ_t⟩ - min_j Σ_t ℓ_{t,j} )
  ≤ T (e^{M} - 1) + ln K / η + η T / 8.
```

For the corollary, `e^{M} - 1 ≤ e·M` for `M ∈ [0,1]`, so the bound becomes
`ln K/η + ηT/8 + e η (H-1) T`; minimizing over `η` gives the stated square-root form. ∎

**Scope of Theorem 1.** It bounds the *linear mixture* loss `Σ_t ⟨w_t, ℓ_t⟩` against
the best single adviser evaluated on the same realized loss sequence. It applies per
(capacity, layer) instance, since those instances share no state. It holds pathwise
and therefore survives the fact that the loss sequence is generated by the algorithm's
own trajectory.

**Numerical reality check.** With `H = 32` and `K = 9` the bound is only informative
when `η(H-1) = 31 η` is small. At the calibration-selected `η = 0.1` the delay term is
`e^{3.1} - 1 ≈ 21.2` per round, which is vacuous for losses bounded by 1. The bound
becomes non-trivial around `η ≈ 8·10⁻⁴` for `T ≈ 4·10⁴` resolved examples — roughly two
orders of magnitude below the smallest value in the preregistered grid `{0.1, 0.3, 1.0}`.
The grid was frozen to minimize measured calibration cache cost, not to satisfy this
bound, and the two criteria point in different directions. The measured empirical
regret reported in the Stage 2 report is therefore the operative evidence; Theorem 1 is
a structural guarantee about the mechanism, not a description of the regime the frozen
configuration runs in.

**Known better dependence.** Joulani, György and Szepesvári (2013) and
Weinberger–Ordentlich (2002) obtain `O(sqrt(H T ln K))` by running `H + 1` interleaved
undelayed instances and routing each round to one of them, which also removes the
`e^{ηH}` sensitivity at fixed `η`. That reduction is *not* what Stage 2 implements: the
implementation runs a single instance per (capacity, layer) with a delayed update. The
rate above matches after optimizing `η`, but the single-instance form is much more
sensitive to `η` being too large.

## 5. What is not proved

1. **The combined RACE score is not the mixture loss.** RACE evicts by
   `S_e = Σ_j w_j z_{j,e}`, and the pairwise ranking loss of `S` is *not* linear in `w`,
   so Theorem 1 does not bound it. The gap is real, not merely technical: for one
   comparable pair `(a,b)` write `δ_j = z_{j,a} - z_{j,b}`. The mixture loss charges
   `Σ_j w_j 1[δ_j < 0]`, whereas the combined score errs iff `Σ_j w_j δ_j < 0`. A single
   adviser with weight `0.01` and `δ = -1` outvotes eight advisers with total weight
   `0.99` and `δ = +0.005`: the combined score's loss is 1 while the mixture loss is
   0.01. No inequality in either direction holds in general.

2. **Ranking loss is not transfer cost.** Cache cost is a non-additive functional of a
   stateful trajectory: an eviction changes which candidates exist at every later event.
   Only inversions that cross the retention cutoff can change a miss at all, and one
   inversion can propagate for many events. The empirical relationship between ranking
   quality and transfer cost is measured in the report; it is not derived.

3. **The comparator is not a counterfactual policy.** `min_j Σ_t ℓ_{t,j}` is the best
   fixed adviser *on the loss sequence RACE actually generated*. Running that adviser
   alone would have produced different candidate sets and hence a different loss
   sequence. The empirical adviser regret reported in the Stage 2 report inherits this
   caveat and must not be read as "RACE is within X of the best single adviser policy".

4. **Static weights are a different object.** `RACE_STATIC` minimizes a convex logistic
   surrogate of the combined score's pairwise loss on calibration examples. Its
   generalization to the evaluation split is not analyzed here; no uniform-convergence
   argument is offered, and the examples are neither independent nor identically
   distributed.

5. **Plain multiplicative weights is not a tracking algorithm.** Theorem 1 compares
   against a *single fixed* adviser. Against a comparator sequence that switches
   advisers `m` times, plain Hedge has no such guarantee, and the Stage 2 test suite
   contains an explicit characterization of the failure: after one adviser accumulates
   a large cumulative advantage its competitors hold numerically negligible weight and a
   late regime switch is not tracked within a comparable number of rounds. Fixed-share
   (Herbster–Warmuth) is the standard remedy and carries an
   `O(sqrt(T (m ln K + m ln(T/m))))` shifting-regret bound, but it is deliberately not
   part of the preregistered Stage 2 algorithm and was not run.

## 6. What would have to be proven

A statement of the form "RACE's expert-transfer cost is within `f(T)` of the best fixed
adviser's expert-transfer cost" would need, at minimum:

1. a surrogate bound linking the pairwise ranking loss of a *weighted rank
   aggregation* to the mixture of its components' pairwise losses, under an explicit
   margin or score-gap condition on the adviser rank differences `δ_j` (item 1 above
   shows an unconditional bound is impossible);
2. a stability argument bounding how many additional misses a bounded number of
   boundary inversions can induce in the frozen atomic mandatory-admission cache — that
   is, a Lipschitz property of the cache cost with respect to ranking error at the
   retention cutoff, not over all pairs;
3. a policy-regret argument that handles the fact that the loss sequence is generated
   by the learner's own trajectory, so that the comparator becomes a counterfactual
   policy rather than a fixed sequence.

Items 1 and 2 look tractable under explicit assumptions; item 3 is the hard one, and it
is the same obstacle that separates external regret from policy regret in adaptive
online learning generally.

## 7. References

- N. Littlestone and M. K. Warmuth. *The weighted majority algorithm.* Information and
  Computation, 1994.
- Y. Freund and R. E. Schapire. *A decision-theoretic generalization of on-line learning
  and an application to boosting.* JCSS, 1997.
- V. Vovk. *A game of prediction with expert advice.* JCSS, 1998.
- N. Cesa-Bianchi and G. Lugosi. *Prediction, Learning, and Games.* Cambridge, 2006
  (Theorem 2.2 gives the form of (2) used here).
- M. Herbster and M. K. Warmuth. *Tracking the best expert.* Machine Learning, 1998.
- M. J. Weinberger and E. Ordentlich. *On delayed prediction of individual sequences.*
  IEEE Transactions on Information Theory, 2002.
- P. Joulani, A. György and C. Szepesvári. *Online learning under delayed feedback.*
  ICML, 2013.
- L. A. Belady. *A study of replacement algorithms for a virtual-storage computer.*
  IBM Systems Journal, 1966 (the offline farthest-future comparator Stage 0 validates).
