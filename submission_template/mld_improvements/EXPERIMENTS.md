# MLD improvement study

## Improvement paths

1. **Repair the comparison protocol.** The existing notebook tests forward-only
   replacement passes; production combines forward/backward absolute outcome
   masses by their maximum. Splitting fixed data into folds does not fit this
   physics decoder and does not protect a 256-setting search from selection
   optimism. Compare against production on independent simulated seed families,
   then freeze finalists before opening another family. Use paired **shot-level**
   intervals because the four logical bits are correlated.
2. **Better pruning look-ahead.** The original score charges each residual
   detector by its most probable remaining mechanism. Instead use its exact
   remaining odd-parity probability, or a hybrid of the two. This approximates
   detector correlations, but leaves every physical mechanism likelihood intact.
3. **Sweep diversity.** Prioritize high-probability mechanisms within the same
   first detector; alternatively only break equal-span ties by probability.
   Forward/backward pruning loses different explanations. Different sweeps'
   outcome-wise maxima remain lower bounds, but are **not** a deduplicated union.
4. **More mass / safer routing.** Broaden the reverse-sweep uncertainty gate,
   tighten pruning, or add a third pass. A confidently wrong approximate
   posterior can otherwise prevent refinement entirely. No monotonic-LER claim
   is justified for truncated posterior sums.
5. **Complementary decoder.** Test the existing physical-pair-closure decoder's
   marginal probabilities against MLD on exactly the same samples, including
   small convex blends. Past MLP blending was unhelpful and the MLP's standalone
   LER is much worse, so do not start with expensive network retraining.
6. **Longer-term structural changes.** Correlated two-detector backward messages,
   boundary-joined bidirectional beams, or logical-class-balanced beams can
   recover missing modes. They require more substantial kernel work and exact
   small-model tests; an enormous parameter grid alone is lower priority.

An additional isolated test, `merge32` / `merge_all`, addresses an early-pruning
bias: a child below the creation threshold can still contribute useful mass to
an **existing** state. These settings admit that contribution without creating
new subthreshold states. Child probabilities are snapshotted before merging,
so an error mechanism is never applied twice in a step.

The existing order has a maximum detector frontier of **26** in both bases:
an unpruned dense table including four logical bits could require **2^30**
states at a step. Exact full-trellis decoding is not assumed feasible.

## Reproducibility and scope

- `search.py`: fresh development seeds **943100–943105**, initially 3,000
  shots/basis/family; fixed configurations recorded alongside source hashes.
- Intended untouched audit family: **954100+**, with finalists frozen first.
- Train CSV is a historical reference, not the selection target.
- No unlabelled test CSVs, test labels, leaderboard queries or banned libraries.
- `test_mld.py` exhaustively compares all new pruning heuristics/orders against
  exact enumeration on a nine-mechanism model padded to 70 detectors.
- Existing production settings remain unchanged during development.
- Generated prediction caches are gitignored; JSON experiment records are kept.

## Development results and frozen audit

Stage 1: 18k shots (3k/basis/family, 943100–943105), 14 individual MLD
configurations, closure and three probability blends, plus 20 cached MLD
combinations. Reference LER **0.03252778**. Tighter search scores 0.03245833,
but its paired interval includes zero. Parity-only look-ahead at exponent 0.5
is worse. Closure is 0.03304167; every tested normalized blend is worse than
reference. There is no demonstrated improvement from this stage alone.

Stage 2: **48k new shots** (8k/basis/family, 943110–943115), nine configurations.
Additional diversity tests reverse/shuffle checks **within each time block**,
or sweep by last detector instead of first. The latter is clearly worse.

| Setting | LER | Delta vs reference, paired shot-level 95% CI | Search seconds |
|---|---:|---|---:|
| Reference | 0.03416146 | — | 7.18 |
| Hybrid .75, tighter two-pass search | **0.03400000** | -0.00016146 [-0.00028352, -0.00003939] | 9.85 |
| Parity .75, tighter search | 0.03400521 | -0.00015625 [-0.00029320, -0.00001930] | 16.91 |
| Original heuristic, tighter search | 0.03405208 | -0.00010938 [-0.00023398, +0.00001523] | 18.66 |
| Hybrid .75, original search sizes | 0.03418229 | +0.00002083 [-0.00013195, +0.00017362] | 4.44 |
| Last-detector order | 0.03458854 | +0.00042708 [+0.00013478, +0.00071938] | 14.52 |

The hybrid combines cheaper, softer look-ahead with a larger retained search.
Its incremental development gain is **31 wrong bits out of 192k**, not a
percentage-point breakthrough. Five of six basis/seed groups improve. Stage-2
max-weight combination of reference and shuffled-check sweeps scores 0.03401563;
normalized averaging generally dilutes gains. Full per-setting groups/CIs,
leave-one-family-out selection, timings and hashes are in `results/`.

**Freeze before opening seeds 954100–954105, 50k shots/group (300k total):**

- Primary accuracy candidate: `hybrid075_tight`, recorded in `candidate.json`.
- Speed diagnostic: `hybrid075`; this is not presumed non-inferior.
- Secondary diversity diagnostic: outcome-wise maximum of `reference` and
  `shuffle_checks`; audit both components and report their predeclared maximum.
- `finalists.json` freezes all individual settings. No parameter changes based
  on audit labels. Production remains the original decoder unless a measured
  accuracy/runtime operating point justifies promotion.
- The primary comparison is hybrid-tight vs reference. Other audit comparisons
  are secondary diagnostics, not a fresh search for whichever mean looks best.

### Absolute closure-mass follow-up (development only)

`closure_mass.py` exposes the existing decoder's retained mechanism vectors on
MLD's **absolute** likelihood scale, rescoring with float64 physical priors
rather than the float32 ranking priors. Exact four-mechanism tests verify the
scale, normalization and deduplication, including duplicated proposal schedules.
This does not modify the released closure decoder.

On stage-1 samples, max-combining closure/MLD outcome masses scores
**0.03255556** vs reference 0.03252778: two additional wrong bits, paired delta
+0.00002778 [-0.00012622, +0.00018177]. No demonstrated gain; do not add its
runtime to the frozen candidate. A maximum is still only a lower-bound
combination, not an exact deduplicated union of MLD and closure paths.

### Compatibility checks

- Eight unit tests cover exhaustive inference, all heuristic/mechanism orders,
  pruning lower bounds, within-round detector permutations, uint64 boundaries,
  impossible/empty/repeated batches, within-call deduplication and closure masses.
  Pooled paired-interval reconstruction is checked against direct calculation.
- `verify_reference.py` loads the original kernel directly from git revision
  `962f067efdfe5ffd7dd8f43f1e4faf99ee42ac1e`, without editing any files.
  Original/default-new **absolute log weights agree bit for bit on 4,000
  sampled shot/sweep comparisons** (1k/basis/direction).
- The only general correctness fix is rejecting final states with residual
  detector disagreement. This matters for impossible inputs with untouched
  detector rows, not the supplied fully supported circuits.

## Independent audit: gains shrink

Frozen settings on **300k new shots**, seeds 954100–954105:

| Setting | Audit LER | Paired delta vs reference (95% CI) |
|---|---:|---|
| Original production MLD | **0.03358667** | — |
| Hybrid .75, tighter search | 0.03356417 | -0.00002250 [-0.00006884, +0.00002384] |
| Hybrid .75, original sizes | 0.03372417 | +0.00013750 [+0.00007414, +0.00020086] |
| Shuffled-check sweep alone | 0.03357500 | -0.00001167 [-0.00006290, +0.00003957] |
| Predeclared max(reference, shuffled) | **0.03354750** | -0.00003917 [-0.00007758, -0.00000076] |

The primary candidate's development advantage mostly disappears. It saves
27 bits but **does not establish an accuracy improvement**. The fast variant
really is less accurate here. The secondary combination saves 47 bits (227
corrections vs 180 new errors) and improves four of six groups, but its interval
only narrowly excludes zero. This is secondary, unadjusted evidence after a
development search, not a robust breakthrough. Relative reduction is **0.117%**,
not 0.117 percentage points. No tuning follows the audit.

`accuracy.json` exports that already-predeclared secondary combination. Both
optional model configurations are unchanged for the additional fresh runtime
check (seeds 965100/965101, 100k/basis). A status-only update to candidate
metadata does not alter its components.

All raw audit measurements remain in `results/`. The generic search wrapper
initially emitted an inappropriate leave-family-out *selection* summary for
audit as well as development; that unused derived file was removed and the
wrapper now produces CV selection summaries for development only. No settings
were chosen from that audit summary.

### Where the remaining errors are

Post-audit diagnostics (`diagnostics.py`) do **not** fit or select a new model.
The baseline has zero completely failed shots, but its mean predicted error
probability is **0.030654** vs observed **0.033587**. Its approximate posterior
is overconfident; this is **not** a certified Bayes-risk floor or proof that the
0.00293 gap can be recovered by changing decisions.

| Fired detectors | Fraction of shots | Observed LER |
|---|---:|---:|
| 0 | 4.57% | 0 |
| 1–4 | 23.35% | 0.00676 |
| 5–9 | 42.73% | 0.02540 |
| 10–14 | 22.97% | 0.05920 |
| 15+ | 6.38% | 0.11852 |

The tail, not the familiar lookup-table head, remains the important target.
Simply rescaling confidence does not change 0.5-threshold decisions.
The two ensemble constituents disagree on only 407 of 1.2M audited bits.
Even an impossible, truth-aware selector between **these two binary outputs**
could save at most their 227 baseline-wrong disagreements, reaching 0.03339750
on this sample. This is a diagnostic for this pair, **not a Bayes lower bound**;
it explains why a large improvement needs a structurally different search/model.

### Best next structural paths, rather than another enormous grid

1. **One detector state, 16 logical weights.** Currently logical classes are
   separate beam entries and weak logical alternatives can be pruned even when
   their detector state survives. Store a length-16 vector per detector state,
   XOR-permute that vector by a mechanism's logical mask, and prune detector
   states by total mass. This directly preserves counterfactual logical mass.
   Compare at matched memory/work, not just matched `max_states`.
2. **Boundary-joined forward/backward beams.** Partition mechanisms into disjoint
   halves, retain prefix/suffix frontier masses, then join compatible boundary
   syndromes with XOR convolution of the logical vectors. Unlike a maximum of
   full-sweep lower bounds, each matched prefix/suffix product represents a
   distinct complete explanation. Validate against exhaustive small models
   before relying on it.
3. **Correlated suffix messages / tail-specific residual learning.** Joint
   two-detector backward messages can model correlations ignored by the cheap
   look-ahead. Alternatively train a small residual/router on physics posterior
   features and high-weight syndromes, not another raw-bit standalone MLP.
   Keep new fitting/development/audit seed families separate.

These larger kernel/model changes are **not implemented or claimed as tested**
by this study. The implemented ablations show why they are higher-priority
next research than repeatedly tightening the existing beam.

## Additional untouched fresh check: ensemble gain replicated

Settings remain frozen. Seeds **965100/965101**, 100k shots/basis, generated only
after the initial audit. Both optional models give identical predictions across
their five `run_local.py` calls.

| Model | New 200k-shot LER | Paired delta vs reference (95% CI) | Train reference LER |
|---|---:|---|---:|
| Production MLD | 0.03360750 | — | 0.03386875 |
| Hybrid/tighter search | 0.03355375 | -0.00005375 [-0.00010870, +0.00000120] | 0.03385875 |
| Reference + shuffled ensemble | **0.03354875** | **-0.00005875 [-0.00010645, -0.00001105]** | 0.03386000 |

Pooled **500k held-out shots / 2M logical bits**:

- Reference **0.03359500**; hybrid **0.03356000**; ensemble **0.03354800**.
- Hybrid: 70 net fewer errors; delta -0.00003500,
  CI [-0.00007044, +0.00000044]. Still inconclusive.
- Ensemble: **94 net fewer errors**; delta **-0.00004700**,
  CI **[-0.00007692, -0.00001708]**. Relative reduction **0.140%**.

This independently reproduces the direction and scale of the secondary gain,
but does **not** establish a major advance or near-optimality. The optional
folder now defaults to that accuracy-first ensemble; its component settings
are exactly those frozen before the first audit. Production is not replaced.
The 0.03372 earlier reported elsewhere is not a matched-sample baseline for
this comparison.

### Speed-only follow-up

Initial sorted deduplication gave fresh median-of-five local times 37.90s for
the hybrid and 60.73s for the ensemble (16 threads, 200k shots). The production
reference's single diagnostic run was 21.53s; do not call that a five-run median.
Sorting syndrome keys also groups easy/hard work into contiguous static Numba
thread ranges. `unique_events` now preserves **first-occurrence input order**
after deduplication to improve load balance. This changes no per-shot search.
`benchmark_order.py` retimes both optional models and checks every output bit
against the original five-run predictions, without reading labels. Judging
B200/48-core runtime is still unknown.

**Final ordered-dedup timing passed:** on the same 100k X + 100k Z fresh shots,
median of five, 16 local threads:

| Optional model | Initially sorted dedup | Preserve-input-order dedup | Prediction changes |
|---|---:|---:|---:|
| Hybrid/tighter | 37.90s | **31.47s** | 0 / 800k bits |
| Reference + shuffled ensemble | 60.73s | **48.98s** | 0 / 800k bits |

Both five-run sequences are deterministic. Order optimization reduces measured
runtime about 17% / 19%, without changing a single prediction. The exported
entry points also reproduce 6,144 cached audit shot predictions exactly,
including the default accuracy preset, empty inputs and repeat calls. All eight
unit tests pass. Final metrics, pooled intervals, parameter and runtime source
hashes are in `results/final_report.json`.

**Recommendation:** keep the original production default until judging-hardware
timing is available. The optional ensemble has the best validated mean LER,
but its 0.14% relative gain buys only about 0.7% runtime beyond `t_cap` if the
reference is under the cap (roughly 15.51s vs 15.4s using the mean LERs).
Laptop runtime is not a judge-machine cutoff; this must be measured rather than
extrapolated. No GPU/B200 timing or sub-0.03 result is claimed.
