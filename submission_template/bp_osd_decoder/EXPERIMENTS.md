# Improving BP+OSD — sequential experiment notebook

Completed: seven sequential sweeps, 83 named settings, 534 basis/seed records,
plus cached ensemble/routing studies. Counts include repeated references and
renamed finalists; they are not 414 independent hypothesis tests.

## Protocol

- The original `bp_osd.py` and `decoder_bposd.py` remain the reference.
  Experimental kernels live in `enhanced.py`; production ML files are untouched.
- Development: three independent seed families, X/Z seeds
  610100/610101, 610102/610103, 610104/610105; 8,000 shots/basis/seed.
  Candidates see detector bits only. Labels are read **only by scoring code**.
- `experiments/search.py` logs full configurations, code/circuit/table hashes,
  per-observable errors, paired per-shot differences and runtime to JSONL.
  Reports include mean X/Z LER, paired confidence intervals, and
  leave-one-seed-family-out parameter selection. A physics decoder has no
  trainable weights here: this is parameter-validation, not fitting a network.
- Final independent holdout uses seed family 720100+, not the development
  seeds, historical `eval_fresh`, or test. Finalists/configuration are frozen
  before the holdout is opened. We also measure `data/train` as a reference.
- No test CSV, test labels, hand labelling, banned decoder packages, or
  leaderboard feedback. Samples and predictions are regenerable, gitignored
  caches; configurations and result logs are retained.
- Search-loop times exclude setup and JIT. Final timing must use five
  `run_local.py` runs at 100k shots/basis. Laptop scores are not GPU-server
  measurements.

## Important qualifications to the earlier notes

1. Seed-999 is no longer an untouched final test: earlier model selection and
   calibration repeatedly used it. New seed families are necessary.
2. The previous claim “all results agree within 1e-4” was wrong: fresh 0.03570
   versus train 0.03529 differ by 0.00041. Different samples need not match
   exactly. Evaluate uncertainty at the **shot** level (the four bits correlate).
3. OSD candidate masses are a **truncated deterministic sum**, not unbiased
   Monte Carlo and not calibrated full posteriors. High candidate confidence
   can still be wrong if the correct class was omitted.
4. The exact mechanism-vector solutions satisfy H e = D; the final independent
   bitwise marginal decisions need not correspond to a single enumerated error.
5. Neither a 0.5–1 percentage-point ensemble gain nor feasible exact trellis
   inference has been demonstrated. Those were hypotheses, not results.
6. Multiplying normalized posteriors by the number of BP iterations does not
   change their order; `avg_post` actually uses their sum, which is sufficient.

## Experiment 1 — parameter ablations and MLP complementarity

**Hypotheses:** more pair candidates could recover missed logical mass;
temperature could compensate truncated-sum overconfidence; a weaker but
complementary MLP might help near ambiguous decisions; more BP iterations
may improve ranking or instead worsen loopy-graph oscillation.

**Design:** one changed knob at a time relative to reference, fixed independent
seed folds. LUT cutoff, temperature, BP iterations/alpha/damping/averaging,
search widths, refinement, and MLP probability mixtures. The MLP files are fixed
`{X,Z}_mlp_1024x512_2600k_s4.npz` and identified by SHA-256.

Raw results: `experiments/results/stage1_development.jsonl`.
Aggregate report: `experiments/results/stage1_development.summary.md`.

**Outcome:** reference 0.035380. Five BP iterations: 0.034917
(paired delta -0.000464, 95% CI [-0.000796, -0.000131]).
Unconditional 1,500 singles/500 pair-width: 0.034833; increasing to 2,300/800
barely changes it (0.034828) while becoming approximately 12x slower than
reference. Neither is a compelling speed/accuracy operating point.

Removing refinement loses 0.000625: uncertainty routing is useful. Doubling
down on the MLP is not: blend weight 0.35 loses 0.000948 (CI excludes zero),
0.05 essentially ties. The learned model's extra errors outweigh its
complementarity under simple probability mixing. This rejects that particular
ensemble, not every learned router. Temperature adjustments and raising the
LUT cutoff also fail to show a consistent benefit. Five iterations are the
most promising small parameter change; test combined effects separately.

## Experiment 2 — third-order OSD candidates

**Hypothesis:** pairs cannot represent three simultaneous nonbasis mechanisms;
enlarging pair width does not fix that structural omission. Enumerating a
restricted set of triples may recover missing probability more efficiently.

If z is the base information-set solution, and a, b, c are nonbasis columns
with information-set expansions k_a, k_b, k_c, a triple candidate toggles
`k_a XOR k_b XOR k_c` and those three outside mechanisms. It satisfies the
same syndrome by construction. Its logical mask is the base logical mask XOR
the three single-candidate logical deltas.

For signed selected-column costs s_r (+w_r if the base bit was zero, -w_r
otherwise), the exact cost is:

    C_abc = C_base + delta_a + delta_b + delta_c
            - 2 sum(s_r over k_a & k_b)
            - 2 sum(s_r over k_a & k_c)
            - 2 sum(s_r over k_b & k_c)
            + 4 sum(s_r over k_a & k_b & k_c)

The pair corrections cancel bits toggled twice; the final term restores bits
toggled three times. Corrections are precomputed where possible. No fitting
or learned likelihood distortion is introduced.

`lam3=0` is required to reproduce the original predictions **and probabilities
bit for bit**. Exhaustive seven-variable toy models test that including all
three nonbasis flips reproduces exact 16-class posterior masses. Both tests,
plus all 64 count-trailing-zero cases, passed before accuracy experiments.

Two triple-subset hypotheses: choose by the BP ranking, or by cheapest
complete single-candidate cost. The latter is not guaranteed better: a triple
can be cheap even when its singles are expensive because pivot costs cancel.

**Outcome:** 40/60 triples (initial/refine subset sizes) plus five BP
iterations scores 0.034745; 60/80 triples at three BP iterations scores
0.034875. The single-cost-ranked subset is worse (0.035182): cheap singles
do not reliably identify cheap triples because pivot costs cancel.
Refine-only triples score 0.035188. Gains are real but small at these widths.
Log-space sum-product BP at ten iterations scores 0.034708, but costs ~17x
the reference search time and does not beat the best union's accuracy.
It reduces the approximation in BP, not the loopy-graph or truncated-search
problem. See `results/stage2_development.summary.md` for all variants/CIs.

## Experiment 3 — combine independently ranked BP searches

**Hypothesis:** different BP schedules may omit different candidates, and
averaging their marginal estimates may be more reliable than either alone.
This does not correctly deduplicate/weight a union of candidate sets: it is
only a simple, cheap-to-test heuristic using cached predictions.

**Outcome:** equal mixtures of iterations 3+5, 3+5+8, and 3+5+alpha0.9 score
0.035130, 0.035146, and 0.035120 respectively, versus 0.035380 reference
but **worse than iterations 5 alone (0.034917)**. Spending 2–4x the compute
does not earn a useful improvement. The constituent searches share much of
their evidence; averaging also dilutes the better schedule. Do not ship.
Raw paired results: `experiments/results/ensembles_development.jsonl`.

## Experiment 4 — union the actual candidates, not their probabilities

**Hypothesis:** schedules discover overlapping but different mechanism vectors.
Averaging their normalized probabilities double-weights overlap and loses
relative partition mass. Instead, pool distinct full error vectors and assign
each its original likelihood exactly once.

Implementation: `union.py`. Each information set still generates zero, one,
or two nonbasis errors. Mechanism vectors pack into 39 uint64 words; hashes
are only a lookup filter and collisions are resolved by comparing the full
vector. A bounded pool keeps the lowest-cost vectors within a log-weight
window (16); this is another explicit truncation, not an exact posterior.
Capacity convergence (512 vs 2048) is tested alongside schedule diversity
and pair width.

Validation before any accuracy claim: exhaustive six-variable models with
two nonbasis variables reproduce exact marginals; repeating an identical
schedule does not multiply its probability; a deliberately constructed
64-bit hash collision retains both different vectors. All tests passed.

**Outcome:** narrow union of iterations 3+5 scores 0.035240, with identical
predictions at capacities 512 and 2048. Wider 3+5+8 union with 600 singles
and pair width 200 scores **0.034417** (paired delta -0.000964,
CI [-0.001302, -0.000625]); consistent gains on all six basis/seed cases.
This outperforms probability averaging, but costs ~5.8x the original search.
The result supports recovering omitted candidates rather than distorting
their prior probabilities. It is not yet an attractive laptop timing point.
Triples at five iterations, 100/140 width, score 0.034510 at ~7.6x cost;
deeper fixed-order enumeration is hitting diminishing returns.

## Experiment 5 — global order-three meet-in-the-middle search

**Hypothesis:** OSD explores “a few nonbasis bits”, not necessarily a few
physical mechanisms. The true low-weight error can still be missed if its
mechanisms rank poorly. Precompute every two-mechanism effect from H and
lookup `D XOR H_j` for every mechanism j to discover ALL weight-three errors.
Add those to the candidate union, retaining original priors and deduplication.

`pair_index.py` builds ~3M pairs/basis with full two-word equality checks and
chains for equal detector syndromes. `i<j<k` counts each triple exactly once.
No training labels are involved. This replaces combinatorial enumeration of
all triples by n indexed queries per shot; the heavy pair build is reusable.
The full posterior is still approximate (weight-four and higher errors can
matter, and the union has a bounded retained pool). Exhaustive three-variable
tests validate all eight errors, including the invisible three-error pattern.

**Outcome:** adding global order-three candidates to a five-iteration,
600-single/200-pair search scores 0.034812. Adding it to the narrower union
scores 0.035161, only slightly better than that same union without it
(0.035240); broad three-schedule union plus exact order-three scores 0.034911.
This does not beat the best ordinary union (0.034417). Correctly enumerating
low-weight events is insufficient: omitted higher-weight events still affect
the logical marginal, and adding mass unevenly can even change decisions the
wrong way. Keep as a tested experiment, not the default.

## Experiment 6 — uncertainty-routed additional search

**Hypothesis:** most of the full union's benefit lies in a small uncertain
subset. Use five-iteration reference probabilities to identify that subset,
then replace its predictions with the wider candidate union.

Development cache replay: routing when any first-pass marginal is between
0.05 and 0.95 scores 0.034573, versus full union 0.034417 and five iterations
alone 0.034917. Routing at 0.2/0.8 scores 0.034719. These are measured
predictions, but the replay runtime is **only an estimate**; actual routed
shots can be harder/slower and must be redecoded and timed.
Logs: `results/routes_development.*`; implementation: `adaptive.py`.

**Actual redecoding:** `routed_replace_005` reproduces the replay's 0.034573
while removing the now-redundant original refinement pass. Five-iteration
small-list BP+OSD is followed by the three-ranking union only when a marginal
is between .05 and .95. Recorded search time ~1.09s/basis/8k, compared with
2.36s for unconditional union; final official-shaped timing remains necessary.
`routed_replace_010` scores 0.034625, `.020` 0.034708, `.030` 0.034797.
No test data or holdout results were used to choose the routing threshold.

## Frozen audit candidates (before opening holdout)

Three predeclared operating points, all with LUT cutoff 4:

| Configuration | Development LER | Role |
|---|---:|---|
| Five iterations, original adaptive pair refinement | 0.034917 | Fast fallback |
| Five-iteration initial pass + uncertainty-routed union (.05/.95) | 0.034573 | Default balanced candidate |
| Full union of iterations 3/5/8, 600 singles, pairs among 200 | 0.034417 | Accuracy-first option |

`candidate/{fast,config,accuracy}.json` are frozen before the audit.
The audit includes the reference and all three predeclared operating points;
it is not another parameter search. Timings can inform which operating point
suits a machine, but no parameters will be retuned to these holdout labels.

## Final independent audit — 300,000 shots, 1.2M logical bits

Seeds 720100–720105; 50,000 shots/basis/seed. No parameter changes after this
audit. These samples were not used for LUT/MLP training or development selection.

| Frozen configuration | Both LER | X | Z | Paired delta vs reference (95% CI) |
|---|---:|---:|---:|---|
| Reference | 0.035200 | 0.035133 | 0.035267 | — |
| Five BP iterations | 0.035081 | 0.035117 | 0.035045 | -0.000119 [-0.000259, +0.000020] |
| Routed union, .05/.95 | **0.034752** | 0.034775 | 0.034728 | -0.000448 [-0.000585, -0.000312] |
| Full three-ranking union | **0.034618** | 0.034628 | 0.034608 | -0.000582 [-0.000719, -0.000444] |

Both union configurations improve on **all six basis/seed groups**. Relative
LER reductions are 1.27% (routed) and 1.65% (full union), not a percentage-point
breakthrough. The fast five-iteration gain shrinks considerably versus
development and its paired interval includes zero: do not call it a proven win.
This is precisely why the independent audit matters after a large sweep.
Full union saves 698 wrong logical bits; routed union saves 538.

The smaller audited improvement versus development is compatible with
selection optimism. There is no evidence here for claiming sub-0.03 decoding
or optimality. Frozen-audit logs: `results/frozen_audit_holdout.*`.

## Speed-only follow-up — reuse BP work

`bp_checkpoints.py` snapshots one BP trajectory at 3/5/8 iterations instead of
running 3+5+8 iterations separately. Different alpha/damping groups still run
separately. Posterior arrays agree bit for bit with independent runs.

`fused.py` goes further for the frozen routed configuration: the first five
iterations already computed for the initial decision are reused on routed
shots, then continued to eight. Easy shots still stop after five. Candidate
enumeration, costs, pooling, thresholds and final decisions are unchanged.

Small-model tests pass; `experiments/verify_speed.py` also compares against
**every archived holdout prediction without reading any ground-truth labels**.
Only after zero differences may this speed specialization replace the generic
implementation. Algorithm/configuration selection remains frozen.

**Equivalence gate passed:** shared-prefix union and fused routed union match
every archived holdout prediction: **600,000 shot predictions / 2.4M bits,
zero changes**. This check reads no truth labels. `candidate/config.json` now
uses the fused implementation; algorithm settings are unchanged.
`results/speed_equivalence.json` retains all twelve per-basis/seed checks.

### Exact candidate-pool indexing

The initial full-size routed timing was 17.17s/200k shots on 16 local threads.
This is **not** a judging-server result: the judges have 48 cores and a B200.
Keep the accuracy-first option rather than discarding it on laptop timing alone.

`pool.py` replaces per-insertion linear duplicate scans and full-pool argmax
scans with collision-checked open addressing and a max-heap. Ties retain the
original first-index argmax rule, so the selected set and its ordering are
unchanged. Tombstones are periodically rebuilt. An avalanche mix is required:
the low bits of word-wise FNV hashes alone would create pathological buckets.
Randomized replacements, equal costs, capacity-one pools, duplicates and forced
hash collisions are checked against the original linear implementation.
No model or routing parameters change; the same full-audit equivalence gate
is required again before recording the optimized end-to-end times.

**Second equivalence gate passed:** `speed_equivalence_hashed.json` again
records zero differences over all 2.4M checked bits, with the source hashes
of the optimized kernels. Full-size routed timing improved from 17.17s to
15.94s median-of-five (100k X + 100k Z, 16 laptop threads).
Matched reference: 7.23s on the same 200k-shot audit subset. On this subset,
LER is 0.03506 reference and 0.03463 routed; the 300k-shot audit above remains
the primary independent accuracy comparison.

The other optional timing run was interrupted; no fast-option timing result
is claimed. Laptop runtimes are diagnostics, not a reason to impose the
judges' 15.4s target on this much smaller machine.

## Accuracy-first follow-up (new development sweep)

After the request to prioritize the judging hardware rather than a laptop
time cutoff, `stage6.json` tests:

- Retention convergence: capacities 512, 2048 and 8192; wider log-weight window.
- More BP checkpoints (3, 5, 8, 12, 20) and pair widths 300 and 500.
- Diversity from distinct alpha/damping schedules.
- Deterministic ranking-prior perturbations: 4 or 12 additional proposals,
  standard deviations 0.5/1.0/2.0, seed 860001. **Original llr0 always scores
  candidates**; the proposal prior cannot change their physical likelihood.

Zero-strength perturbations must produce duplicate candidates without
multiplying probability; that additional unit test passes (15 tests total).
All selected settings retain CPU-shot parallelism for 48 judge threads.
No GPU implementation or B200 runtime is claimed.

This uses the development family 610100+, NOT the opened audit. Any selected
follow-up must be frozen and audited on a NEW family, 830100+; no retuning
against the previous holdout.

**Stage-six outcome:** best development setting is four ranking-prior
perturbations at sigma 1.0, LER **0.034292** versus full-union control
0.034417. Twelve perturbations score 0.034323; pair width 500 with deeper
checkpoints scores 0.034339. More work helps only slightly, with noisy
seedwise differences. Larger capacity/window alone changes one prediction,
not the overall picture (0.034422). Thus retained-pool truncation is not
the major bottleneck at the current search width. No new holdout has yet
been opened for these settings.

### Next isolated hypothesis prepared: physical-pair closure

The global three-error index did not solve higher-weight omissions. A different
use of that index is now correctness-tested in `local_closure.py`: for a good
candidate containing physical errors i,j, replace them with every pair k,l
having the same detector effect. Toggling i,j,k,l preserves H e = D and can
discover other weight-four/five/six explanations directly, regardless of the
OSD information set. Costs and logical masks are recomputed from the original
priors; exact-vector pooling prevents double-counting.

Two seed policies are prepared: globally cheapest candidates, or a few cheap
candidates per logical class to avoid expanding only the currently dominant
class. A toy model recovers its exact posterior; randomized higher-weight
checks verify every new syndrome, cost and logical outcome. The integrated
toy decoder also recovers the exact posterior. All 18 tests passed before
launching `stage7.json`; no closure accuracy gain is claimed yet.

**Stage-seven outcome:** two cheap expansion seeds per represented logical
class scores **0.034224**, versus same-width/window control 0.034422.
Global seed counts 16 and 64 score 0.034250 and 0.034250; repeated/global
expansion does not keep improving, and eight seeds/class over two rounds is
worse (0.034354). Combining jitter and closure also fails to beat class2
alone (0.034333).

The modest improvement is compatible with recovering overlooked degenerate
explanations. Larger expansions can add mass unevenly across logical classes;
there is no monotonic-accuracy guarantee for these truncated sums. The
development differences are too small to trust without the new audit.

**Frozen primary follow-up:** `candidate/accuracy_closure.json`,
two seeds/class, one round, capacity 2048, log-weight gap 20, checkpoints
3/5/8, 600 singles/pairs among 200. Before opening family **830100–830105**,
also freeze the original decoder and previous full union as controls.
The audit has 50k shots/basis/seed (300k total). It is not used
for any further hyperparameter search.

### Second sealed audit: closure confirmed

| Frozen configuration | Both LER | X | Z |
|---|---:|---:|---:|
| Original reference | 0.03506333 | 0.03555833 | 0.03456833 |
| Previous full candidate union | 0.03432250 | 0.03482167 | 0.03382333 |
| **Physical-pair closure, 2 seeds/class** | **0.03414583** | **0.03465000** | **0.03364167** |

Both comparisons improve on all six basis/seed groups:

- Versus reference: delta **-0.00091750**, paired 95% CI
  **[-0.00107382, -0.00076118]**; 1,101 net fewer wrong bits out of 1.2M.
  Relative LER reduction **2.62%**.
- Versus previous full union: delta **-0.00017667**, paired 95% CI
  **[-0.00026383, -0.00008950]**; 212 net fewer wrong bits.

The incremental closure gain is small but independently confirmed. Larger
search width, jitter and multiple closure rounds did not give a better
development operating point. Do not substitute their optimistic development
scores for these audited numbers.

Records: `results/closure_audit_holdout.*` and its two paired-comparison JSONs.
`candidate/config.json` is promoted to the immutable
`accuracy_closure.json` configuration. The earlier routed default survives
as `balanced.json`; the imported reference and primary ML decoder are unchanged.

### Full-size end-to-end checks

- On `data/bposd_audit2` (100k/basis, an explicitly identified subset of the
  second audit), `run_local.py` reports **LER 0.03403**, X 0.03485 / Z 0.03322.
  Five identical prediction runs; median **56.86 s** on 16 laptop threads.
- On `data/train`, LER **0.03443**, X 0.03495 / Z 0.03391; one timing/reference
  run at **58.77 s**. The held-out five-run median is the runtime measurement.
- These are local CPU numbers, **not** a judging-server score or cap failure.
  The kernels support 48 judge threads. No GPU/B200 timing is claimed.
- All **19 correctness tests** pass after promotion. Exact toy posteriors,
  collision-safe deduplication, likelihood costs, syndrome preservation,
  checkpoint equivalence, routing, repeated/empty batches and model aliases
  are covered.

### Standalone deliverable

`build_handin.py` exports `handin/decoder.py`, pinned requirements, known
graphs/LUTs, source generators, circuit inputs and a SHA-256 file manifest.
Only explicitly listed files are copied: **no train/test CSVs, experiment
caches or unrelated MLP weights**. The candidate's package imports are lazy,
so the organizer's graph-construction helper is not a runtime dependency.
A fresh-process smoke test with organizer imports deliberately blocked
successfully decoded both X and Z.

## Lower-bound diagnostic (no labelled data)

`experiments/low_weight_bound.py` enumerates all zero/single/pair mechanism
events and sums per-syndrome, per-bit minima of their **unconditional**
logical masses. Adding omitted probability cannot reduce either mass, so
this is a valid lower bound for the independent-DEM model. It is not a
conditional score renormalized to low-weight events.

- X: lower bound 0.00100247; event probability for mechanism weight <=2
  is 0.400203.
- Z: lower bound 0.00104367; covered event probability is 0.411626.
- Combined bound approximately **0.00102307**.

The derivation is tested against exhaustive exact inference on a small model.
This is a weak bound because ~60% of mass is omitted: it neither establishes
that 0.001 is achievable nor justifies calling 0.034 near-optimal. It is useful
mainly as a guard against unsupported claims of a proven accuracy floor.
Raw record: `results/low_weight_bound.json`.

## Selection rule

Prefer consistent seedwise gains with a favourable paired interval and a
reasonable runtime, not a single noisy lowest number. Report failed approaches
as well as successful ones. A holdout gain is evidence for the frozen change,
not proof that the Bayes limit has been reached.
