# BP+OSD decoder — how it works, and how to push past 0.035

Ported from branch `lut-bp-osd-decoder` (see `README.md` for provenance).
Their measured numbers: **data/train LER 0.03522** (X 0.03574 / Z 0.03471),
**fresh seed-7 0.03473, seed-8 0.03520**, median runtime 7.6 s on an 8-core
laptop. Verification on this branch: §6.

## 1. The problem, precisely

Hidden error vector `e ∈ {0,1}^n` (n ≈ 2470 mechanisms, independent priors
`p_i`). We observe detector events `D = H·e (mod 2)` (70 bits) and must predict
observable flips `O = L·e (mod 2)` (4 bits). The map `e → D` is many-to-one
(degenerate code), so the per-bit optimal prediction is the **marginal**
`P(O_j = 1 | D)` — an average over *all* error patterns consistent with `D`,
weighted by priors. Everything in this decoder is an approximation of that
marginal. (The ML decoder in `../models/` approximates the same object by
regression; this decoder computes it from the exact `H, L, p` structure.)

## 2. Stage 1 — exact-match lookup table (the easy shots)

`build_lut.py` samples 10M fresh Stim shots per basis (seeds 20261113/14 —
deliberately different from train's seed 42, and using no data files at
all) and stores, per observed detector pattern, the **majority vote** of each
logical bit. That is the empirical Bayes decision for that pattern: with 10M
shots, frequent patterns have well-sampled `P(O|D)`.

`decode_bposd` trusts the table **only for shots with ≤4 detectors fired**
(`LUT_MAX_FIRED`). Why the cutoff: for low-weight syndromes the table is as
accurate as BP+OSD and ~100× faster; for heavier syndromes each pattern is
rare (we measured ~79k distinct patterns per 100k shots at mean weight ~7.4),
so majority votes are noisy and BP+OSD's structural reasoning wins. The
crossover point is empirical — see §7B for how to re-tune it.

Implementation detail: patterns pack into 9 bytes (`bitorder="little"`,
Stim's native layout) and lookup is one vectorised `searchsorted` over the
sorted key array — exact matches in C speed, no Python loops.

## 3. Stage 2a — belief propagation (damped, averaged min-sum)

For the remaining shots, BP passes messages on the factor graph defined by
`H`, initialised from the priors (`llr0 = log((1-p)/p)`). Concretely this is
**flooding min-sum with three stabilisers**, all motivated by the fact that
qLDPC graphs are loopy and plain min-sum oscillates or locks into symmetric
fixed points:

- **Normalisation** (`alpha=0.8`): check-to-variable messages are scaled down;
  min-sum is overconfident, and this corrects the magnitude.
- **Damping** (`damping=0.5`): each message is mixed 50/50 with its previous
  value, killing oscillation cycles.
- **Posterior averaging** (`avg_post`): OSD ordering uses the *time-average*
  of posteriors over iterations, not the last iterate — a much more stable
  likelihood ranking.

Memory trick that makes this fast: the error→detector message is never
stored; it is reconstructed as `post[j] − c2v[e]` (last posterior minus the
edge's own incoming message). Only one float per edge plus posteriors is
kept, so the working set stays in cache. Runs with `max_iter=3` and
`always_osd=True` — i.e. BP is used **only as a likelihood ranker** for OSD,
not as a solver (the hard-decision check is skipped). Shots are parallelised
with numba `prange` over chunks.

## 4. Stage 2b — OSD-0 (turning a ranking into an exact solution)

OSD-0 takes BP's likelihood order and greedily picks the first `m` (=70)
**linearly independent** columns of `H` — the "information set" `S` — then
solves `H·e = D` using only `S`. This *always* returns an `e` explaining `D`
exactly (if BP's ranking is good, a likely one).

The implementation keeps the basis **fully reduced** (each basis vector holds
exactly one pivot bit), so both elimination and solving are XOR-chases over
pivot bits rather than full row operations. Columns live as two `uint64`
masks (70 bits), pivots are found with a De Bruijn `ctz`, and only the top
`~m + lam + 128` columns are sorted (partial sort with full-sort fallback) —
all of it numba-compiled.

## 5. Stage 2c — combination sweep with coset scoring (the accuracy engine)

OSD-0 alone just finds *a* likely error (max-likelihood-flavoured). The sweep
goes one step toward the true marginal: it also tries switching on each of
the `lam=400` most-likely columns **outside** `S` (singles) and every pair
among the top `lam2=100` of them, re-solving `S` each time. Every candidate
explains `D` exactly; each has a cost `Σ cost[j]` (negative log-likelihood).

Scoring is the key idea: instead of taking the single cheapest candidate,
outcomes are scored by **total probability over all candidates that produce
them** (`coset` modes). `coset=2` (their choice) decides **each bit
independently** — flipped iff the candidates flipping it carry most of the
probability mass. This is a truncated deterministic approximation of the marginal
`P(O_j | D)` over the high-probability neighbourhood — the closest thing in
this decoder to the Bayes-optimal decision, and the step that took them from
0.053 → 0.038 → 0.035 across commits.

## 6. Stage 2d — uncertainty-triggered refine (`classp`)

OSD also returns `classp`: the 16-entry distribution over 4-bit outcomes
from the candidate pool. From it, per-bit marginals fall out
(`marg = classp @ bits`). Shots with any bit in the grey zone
(`|marg − 0.5| < 0.5 − refine_thr`, thr 0.3) are **re-decoded with a wider
sweep** (`refine_lam=1000`, `refine_lam2=300`): adaptive compute spent only
where the first pass admits uncertainty. Measured effect: 0.03551 → 0.03522
at +1.7 s runtime. This is the same "uncertainty routing" instinct behind
our MLP-hybrid work, but entirely inside one decoder.

## 7. Verification status (this branch, ml-models)

| Eval set | Claimed (their branch) | Reproduced here |
|---|---|---|
| data/train (200k) | 0.03522 (X .03574 / Z .03471), ~7.6 s | **0.03529** (X .03583 / Z .03475), 7.14 s |
| fresh seed-7 / seed-8 | 0.03473 / 0.03520 | not re-run (their seeds) |
| data/eval_fresh, seed 999 (40k) | — (didn't exist there) | **0.03570** (X .03585 / Z .03555), 1.73 s |

Hygiene verdict: **no test-label reads found in the audited code**.
LUT trains on fresh Stim seeds (no data files);
BP+OSD trains on nothing; evals reported on train + two fresh seeds that
are consistent (not proof of no parameter-selection overfit). No test CSV reads in
decoder code; no banned libraries (imports are only `numpy`, `stim`,
`numba`, `hackathon_2026.problem_info` — all allowed).

## 8. Why it beats our MLP (0.035 vs 0.075)

1. **Exact constraints, every shot.** OSD always returns an `e` with
   `H·e = D`; the MLP has no such guarantee and wastes capacity learning
   near-impossible mappings.
2. **Exact priors.** `p_i` spans 1e-4…2e-2; BP uses them directly, while the
   MLP must rediscover 2470 priors from samples.
3. **The tail.** Rare high-weight syndromes decide the endgame; the MLP sees
   each a handful of times (if at all), while OSD *solves* them from
   structure. Our coverage study showed the same thing from the other side:
   memorisation flattens, structure generalises.
4. **Marginal-aware scoring.** `coset=2` + candidate pooling approximates the
   Bayes marginal; plain BP (or OSD-0 alone) is max-likelihood and
   degeneracy-blind.

## 9. Next steps (ordered by expected gain / effort)

**A. Ensemble BP+OSD × MLP — highest leverage.** The two decoders fail on
different shots (structure-solver vs learned-marginal). Concrete routings:
  1. *Uncertainty routing*: `classp` margin (how sure is OSD?) vs MLP
     confidence (`|P−0.5|`); each shot goes to whoever is surer, or to a
     wider refine sweep when both are unsure. All signals already exist.
  2. *Stacking*: features = OSD `classp` marginals (4) + MLP probs (4) + a few
     syndrome stats → tiny logistic/MLP head trained on a fresh-seed set.
  3. *Distillation*: train the MLP on BP+OSD's soft `classp` outputs as
     targets (unlimited pseudo-labels from Stim) instead of hard flips —
     distils structural knowledge into millisecond inference.
  Expected: 0.5–1pp. Cost: days, all within timing budget.

  **Update after implementation:** this estimated gain was speculative.
  Simple probability blending failed to improve the reference consistently.
  See `EXPERIMENTS.md` and the paired multi-seed results before choosing a router.

**B. Re-tune the operating point.** Grid over `lam/lam2/refine_thr/
LUT_MAX_FIRED/alpha/damping/max_iter`, **per basis**, selected on fresh
seeds (7/8 + our 999 — never test). Two sub-questions with likely payoff:
  1. The LUT↔BP crossover vs fired-count: measure per-weight accuracy of
     table vs BP+OSD; the ≤4 cutoff may be conservative (their table-only
     LER was 0.21, so the headroom is in weights 5–8).
  2. `coset` mode 1 (joint-argmax) vs 2 (per-bit): per-bit LER favours 2 in
     theory, but only with calibrated candidate pools — verify, and try
     temperature scaling (`temp` param exists) + per-bit thresholds on
     `classp` marginals (mirrors our MLP calibration, which helped Z).
  Expected: 0.1–0.4pp. Cost: hours of compute, mostly waiting.

**C. Adaptive compute.** `max_iter=3` + always-OSD for every shot is blunt:
try BP-first with convergence check (`always_osd=False`) for easy shots and
reserve iterations + wide sweeps for non-converged/uncertain ones; add
random-restart ("dithered") OSD for shots where BP ranking looks degenerate
(several near-equal posteriors). Goal: same LER faster, or better LER at the
same ~7 s budget.

**D. Prior calibration.** The DEM's nominal `p` may not be the best effective
prior (merging approximations, correlations). Fit a global scale / per-class
offset on a fresh-seed set by maximising OSD-sweep likelihood; even a single
temperature on `cost` can shift all rankings coherently.

**E. Bigger LUT, smarter fallback.** 10M shots → hit-rate for weights 0–4 is
already high, but measure exactly what the table covers per weight and what
a 20–50M build (or a weight-5/6-restricted table) would add. Alternative:
replace majority vote with the empirical `P(O|D)` and feed it as a BP prior
or an ensemble feature instead of a hard decision.

**F. Higher-order / exact structure for the hardest shots.** Pairs help;
triples among the top ~20 cost O(10³) extra solves per shot — affordable if
restricted to `classp`-uncertain shots. For very-low-weight syndromes,
exact transfer-matrix/trellis marginalisation over the 5-round circuit is
not yet shown tractable; if feasible it would give ground-truth marginals (also a yardstick to calibrate
everything else against).

**G. Speed engineering for the judging rig.** Current 7.6 s is an 8-core
laptop number; judges use 48 cores + B200. Measure thread scaling
(`NUMBA_NUM_THREADS`), chunk sizing, and numba-cache warm-up so the
2.3 s of headroom under `t_cap` is real, not assumed. The refine stage is
the obvious dial: uncertainty threshold ↔ runtime curve.

**H. Validation discipline.** Keep seed-999 `eval_fresh` plus at least one
more fresh seed as the selection sets; `data/test` stays decode-only;
re-run `check_handin.py`-equivalent before any hand-in. Their branch's
habit — reporting train + two fresh seeds on every change — is the template.

## 10. Risks / housekeeping

- `numba==0.68.1` installs cleanly here (verified); keep the pin.
- `lut/` tables (~113 MB) are gitignored by design; the hand-in must bundle
  them **plus** `build_lut.py` (the repo rules require the generating code).
- Bit packing here is `little`-endian; our ML LUT uses big-endian — the two
  table formats are mutually unreadable, keep the folders separate (as done).
- `decode_bposd` builds `problem_info` + graph on **every** call (part of the
  timed path, as in their design); hoisting it to import time is possible
  but changes the timed profile — measure before/after if attempted.

## Implementation follow-up

`EXPERIMENTS.md` supersedes speculative estimates above with measured sequential
ablations and a new independent holdout protocol. `candidate/` is the isolated
entry point for any promoted result; the original reference and ML decoder stay
available for reproducibility.

### What the completed experiments actually support

The earlier independent 300k-shot audit measures reference 0.035200,
uncertainty-routed candidate union 0.034752 and full union 0.034618.
A subsequent sealed 300k-shot audit confirms physical-pair closure:
**reference 0.03506333 -> union 0.03432250 -> closure 0.03414583**.
The final closure gain over union has paired CI [-0.00026383, -0.00008950]
and improves all six basis/seed groups. These are modest, consistent gains,
not a new 0.02 decoder. Simple MLP blending, posterior
averaging across schedules, global order-three enumeration, and more
sum-product iterations did not produce a better practical operating point.

The strongest next research directions are consequently more specific:

1. **Expand distinct higher-weight explanations, not just the list width.**
   The all-pairs meet-in-the-middle index can find every three-mechanism
   explanation, yet that variant did not beat a good OSD union. Physical-pair
   replacement around high-weight candidates did help and is now implemented.
   Its current moves remove two present errors and toggle an equivalent pair:
   they cannot increase total mechanism weight. The next distinct hypothesis
   is short kernel-cycle moves overlapping only ONE present mechanism, which
   can add two errors while preserving the syndrome. Those higher-weight
   contributions may change logical-class entropy. This extension is not yet
   implemented and no gain is claimed. Any enumeration must retain exact
   vector deduplication and original priors.
2. **Sample within the syndrome coset.** Fundamental kernel-vector moves
   preserve H e = D. Metropolis/tempered blocked moves started from several
   good union candidates could explore modes not reached by low-order sweeps.
   This is unimplemented: mixing, burn-in, correlated samples and deterministic
   seeded inference would all need explicit validation. A short chain is not
   automatically an accurate marginal.
3. **Learn ranking or compute allocation, not a weaker replacement decoder.**
   Direct MLP blending failed. A network predicting mechanism reliability or
   whether another search will correct the current decision has a narrower,
   measurable task. Train only on newly generated development data and reserve
   another untouched seed family; the current final audit is now spent.
4. **Profile the remaining hot path on the judge hardware.** Collision-safe
   hashed pooling, a max-heap, shared BP prefixes and fused routing are now
   implemented and were bit-checked against 2.4M archived predictions.
   Candidate-vector reconstruction, repeated GF(2) elimination and closure
   expansion remain worth profiling. Preserve likelihoods, deduplication and
   tie-breaking when claiming a speed-only change.
5. **GPU work only after profiling.** Batched check updates and candidate scoring
   are parallel; per-shot GF(2) elimination is harder to map efficiently.
   B200 performance cannot be inferred from laptop timing. Any CUDA/Triton
   implementation must be ours, not copied from a banned decoder.

Do not retune on either the 720100 OR 830100 audit family. Future research
needs a new sealed family (for example 940100+) in addition to these
now-public regression sets.

The exact mechanism-weight<=2 lower-bound diagnostic gives only ~0.00102,
covering ~40% of probability mass. It is too weak to establish proximity to
the full Bayes limit; convergence of the tried heuristics is not a proof of
optimality.
