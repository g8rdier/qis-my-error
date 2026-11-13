# Decoder scoreboard — LER (per-bit, lower is better) + parameters

Historical ML rows use the seeded validation splits documented per run;
**fresh** = `data/eval_fresh/eval_fresh.csv` (20k shots/basis,
seed 999). The BP+OSD section has a separate independent-audit protocol.
`data/test/test.csv` is NEVER used for training
or model selection — decode-only for the final submission.
Timing: `run_local.py` median-of-5 on this machine (guide only; judges use B200+48 cores).

## MLD improvement study (current production reference)

The current `decoder.py` is the forward/backward pruned MLD, not the historical
MLP described in older rows below. A new study tested 19 distinct MLD settings
plus cached ensembles and closure combinations on 66k development shots,
then froze candidates before a 300k-shot independent audit (954100–954105).

| Frozen model | Matched audit LER | Paired delta vs production (95% CI) |
|---|---:|---|
| Current production MLD | 0.03358667 | — |
| Hybrid look-ahead, tighter search | 0.03356417 | -0.00002250 [-0.00006884, +0.00002384] |
| Max(reference, shuffled checks) | 0.03354750 | -0.00003917 [-0.00007758, -0.00000076] |

The primary gain is inconclusive. The tiny secondary gain then replicated on
another 200k untouched shots (965100/965101): 0.03354875 vs reference 0.03360750.
Across all **500k held-out shots**, the ensemble scores **0.03354800** vs
reference **0.03359500**, 94 fewer wrong bits out of 2M; paired difference
**-0.00004700**, 95% CI **[-0.00007692, -0.00001708]**.
This is a **0.14% relative** reduction with more compute, not a breakthrough.
Optional model runtimes after input-order-preserving deduplication:
**48.98s ensemble / 31.47s hybrid**, median-of-five, 200k fresh shots, 16 local
threads. These are not B200/48-core judge timings. All five calls are identical;
the speed change preserves all 1.6M checked bits, and eight correctness tests pass.
**Production is not replaced.** These matched scores should not be compared
directly with a 0.03372 result on a different sample.
Implemented paths, rejected ideas, exact-inference tests, optional entry points,
runtime checks and raw logs:
[`mld_improvements/EXPERIMENTS.md`](mld_improvements/EXPERIMENTS.md).

## BP+OSD improvement program (separate from the ML decoder)

The historical seed-999 set below was used repeatedly for ML selection; it is
not the final holdout for this experiment. BP+OSD configurations were selected on
three newly sampled seed families, then frozen before independent audits.
No Kaggle test CSV or test labels used.

**Latest sealed audit: 300k shots, seeds 830100–830105 (150k/basis):**

| Frozen configuration | Audit LER | Delta vs reference |
|---|---:|---:|
| Imported reference | 0.03506333 | — |
| Previous full union | 0.03432250 | -0.00074083 |
| **Physical-pair closure, 2 seeds/logical class** | **0.03414583** | **-0.00091750** |

Closure improves all six basis/seed groups and beats the prior full union
by -0.00017667 (paired 95% CI [-0.00026383, -0.00008950]).
Total relative improvement over reference: **2.62%**. This remains above 0.03.
The named model is `bp_osd_decoder/candidate/accuracy_closure.json`,
also selected by that folder's `config.json`.
Official-shaped local check: 56.86s median-of-five on 200k held-out shots,
16 laptop threads, deterministic. Judge timing on B200+48 cores is unmeasured.

**Earlier independent audit: seeds 720100–720105, 300k shots:**

| Frozen BP+OSD option | Independent audit LER | Delta vs reference | Status |
|---|---:|---:|---|
| Imported reference | 0.035200 | — | Preserved unchanged |
| Five iterations only | 0.035081 | -0.000119 | Paired interval includes zero |
| Uncertainty-routed candidate union | **0.034752** | **-0.000448** | Preserved as balanced.json |
| Full candidate union, 3/5/8 rankings | **0.034618** | **-0.000582** | Preserved as accuracy.json |

Within that earlier audit, union variants improve all six basis/seed groups;
the fast parameter-only gain is less convincing. Do not compare the two audit
families as paired samples.
Full per-setting/per-seed results, paired intervals, rejected ideas and reasoning:
[`bp_osd_decoder/EXPERIMENTS.md`](bp_osd_decoder/EXPERIMENTS.md).
Entry point: `bp_osd_decoder/candidate/decoder.py`; configurations and graph
artifacts are kept there, separately from all MLP weights and manifests.
The standalone `bp_osd_decoder/handin/` includes runtime artifacts and source
generators, without labelled CSVs. The primary `submission_template/decoder.py`
remains the existing ML decoder.

## Results (both bases combined; X / Z split in notes)

| Model | Params / config | Val LER | Fresh LER | Decode time* | Status |
|---|---|---|---|---|---|
| all-zeros | — | 0.28815 | — | ~0 s | measured |
| LUT exact + zeros fallback | train 90k/basis | X 0.26150 / Z 0.25610 | X 0.26521 / Z 0.25665 | ~0.1s/200k | measured 11-13 |
| LUT exact + dist1 fallback | train 90k/basis, vectorised searchsorted | X 0.25503 / Z 0.24618 | X 0.25793 / Z 0.24674 | ~3.5s/200k ✓ under cap | measured 11-13 |
| logreg (numpy SGD) | ep30 bs4096 lr0.5 l2 1e-4 | X 0.29120 / Z 0.28340 | X 0.29613 / Z 0.28694 | TBD | measured 11-13 |
| MLP 70-256-128-4, 90k rows | torch Adam 1e-3 bs8192 do0.1; X ep79 seed1 / Z ep40 seed0 | X 0.16700† / Z 0.17380 | X 0.16245 / Z 0.17554 | — | superseded 11-14, weights in *.prev.npz |
| MLP 70-512-256-4, 600k rows | torch Adam 1e-3 bs16384 do0.1; 120ep pat20 seed2 | X 0.09920 / Z 0.09274 | X 0.09690 / Z 0.09477 | — | superseded 11-14 |
| MLP 70-512-256-4, 2.6M rows | torch Adam 1e-3 bs32768 do0.1; 60ep pat10 seed3 | X 0.08473 / Z 0.08048 | X 0.08338 / Z 0.08189 | — | superseded 11-14 |
| MLP 70-1024-512-4, 2.6M rows | torch Adam 1e-3 bs32768 do0.1; 80ep pat12 seed4 | X 0.07790 / Z 0.07177 | X 0.07541 / Z 0.07399 | — | measured 11-14, current |
| hybrid LUT-exact + MLP-miss (=decoder.py) | tables 600k/basis + 1024-MLPs, thr 0.5 | — | X 0.07565 / Z 0.07420 | 0.23s/40k (~1.2s/200k) ✓ | measured 11-14, current hand-in |
| GBM (LightGBM/sklearn) | design only | — | — | — | not implemented |

\*Decode time = `run_local.py` median on 100k shots/basis; target well under t_cap ≈ 15.4 s.

## Per-basis reference (measured 2026-11-13, `data/train/train.csv`)

- all-zeros: X 0.29126, Z 0.28504, both 0.28815.
- Base rates P(flip): X [0.305 0.307 0.242 0.311], Z [0.312 0.337 0.239 0.252];
  P(any flip): X 0.564, Z 0.592.
- Syndrome weight mean: X 7.43, Z 7.31 (NOT sparse 0–3; heavy tail to ~30).
- Distinct D patterns per 100k shots: X 79,304 / Z 78,309. Top pattern is D=0
  (~4.5%); next most common only ~120 occurrences. ~69k patterns cover 90% mass.
  => Pure LUT hit rate on unseen data will be ~20%; fallback decides LER.
- H: 70 x ~2470, mean row weight ~152, col weights 1–7; p in [1.3e-4, 2.3e-2].
- L density: 45% (X) / 62% (Z) of mechanisms touch an observable.

## Model notes (how each works — one line)

- **LUT**: majority-vote O per exact D pattern; near-optimal on hits, misses dominate.
  Measured hit rate only ~24% (val) / ~24% (fresh) — as predicted from 79k
  distinct patterns/100k shots. dist1 fallback gains ~0.7–1.0pp over zeros.
- **logreg**: 4 linear sigmoids on 70 bits; no parity structure; floor + fallback + stack feats.
  Measured: NO better than all-zeros (X val 0.29120 vs 0.28790 all-zero; Z tie).
  Linear-on-raw-bits is a dead end standalone — confirms parity hypothesis. Keep
  only as possible stack feature / emergency fallback.
- **MLP**: 70-256-128-4 BCE, approximates Bayes marginal P(O|D); train torch/GPU,
  infer numpy (torch-free decode). Now the clear priority: only model class that
  can represent parity structure with fast matmul inference.
- **GBM**: trees on bits; parity needs many leaves; inference-speed risk; ensemble member at best.

## Log

- 2026-11-13: scaffolded `models/` (common, lut, logreg, mlp, gbm-doc) + `experiments/`
  + this file. Measured dataset stats + all-zero baseline. Fresh eval set: pending.
- 2026-11-13: generated `data/eval_fresh` (20k/basis, seed 999). LUT measured
  (best both-basis ≈ 0.252 fresh-avg with dist1). Logreg measured: dead end standalone.
- 2026-11-13: torch 2.6+cu124 on RTX 3050 Ti. MLP Z: val 0.1738 in 40ep. MLP X seed-0
  flatlined at all-zero (bad init + patience 6 < 12ep grokking delay seen on Z);
  retry seed-1, 80ep, patience 15 → val 0.16700, still improving at ep 80.
  Lesson: min 80 epochs + patience ≥15 for this data; never trust a flat start.
- 2026-11-13: hybrid (LUT-exact + MLP-miss) beats pure MLP on both bases, both splits;
  wired into decoder.py (LUT tables + numpy MLP forward + basis fingerprint, torch-free).
  † X "val" is the seed-1 split (its own training val); seed-0 val is contaminated
  for the X-MLP (90% overlap with its seed-1 train set) — FRESH is the honest metric.
- Next: run_local.py on eval_fresh (honest LER + timing) and train (reference).
- 2026-11-13: run_local decoder.py → eval_fresh both 0.16734 (X 0.16131 / Z 0.17336),
  0.08s/40k (~0.4s per 200k); train both 0.01582 (in-sample: LUT memorised ~90% of
  train patterns — textbook overfit demo, NEVER select on train LER).
  Headroom vs t_cap is ~100x: budget is free for bigger MLPs / ensembles later.
- 2026-11-14: coverage scaling (experiments/measure_coverage.py, fresh seeds 1000/1001,
  eval on untouched eval_fresh). Table rows → hit rate / hybrid fresh LER:
  X: 10k→16.3%/0.16229, 30k→19.5%/0.16205, 90k→24.1%/0.16146, 200k→27.9%/0.16026,
     500k→32.1%/0.15879. Z: 10k→17.3%/0.17517, 30k→20.6%/0.17471, 90k→24.8%/0.17329,
     200k→28.4%/0.17172, 500k→32.5%/0.16937.
  Verdict: hit rate grows ~logarithmically (50x data → +16pp hits, +0.4–0.6pp LER).
  A huge exact table ALONE never solves it — misses are the informative syndromes
  and sorted order gives no nearest-match shortcut (integer neighbours ≠ Hamming
  neighbours; true dist-1 best-match costs 70 lookups). Table is worth growing
  (cheap, fast, helps the head) but the MLP tail does the real work.
  Shipped: production tables rebuilt from train-full 100k + gen_big 500k = 600k
  rows/basis (experiments/build_production_tables.py) → run_local eval_fresh
  both 0.16354 (X 0.15824 / Z 0.16885), 0.07s/40k. Previous 90k-table score 0.16734.
- 2026-11-14: big MLP (train_mlp.py --use-big --hidden 512 256, 120ep/pat20,
  batch 16384, seed 2, 600k rows/basis): val X 0.09920 / Z 0.09274, fresh
  pure-MLP X 0.09690 / Z 0.09477 vs old 90k-net 0.16245/0.17554 — data scaling
  works. Hybrid with LUT now ties pure MLP (X 0.09695 / Z 0.09477): the stronger
  MLP already beats table majority vote even on exact hits, so the LUT is now
  dead weight (kept: harmless, ~0.1s). run_local eval_fresh both 0.09586,
  0.12s/40k. Old weights kept as *.prev.npz. Next knobs: round-parity features,
  3-seed ensemble, per-observable threshold calibration (all still ms-scale).
- 2026-11-14: weight registry. artifacts/manifest.json maps every *_mlp_*.npz to
  {arch, data, seed, hyperparams, val_ler, fresh_ler, cmd}; artifacts/current.json
  pins per-basis {weights, threshold} consumed by decoder.py (legacy X/Z_mlp.npz
  fallback kept). train_mlp.py auto-tags runs and registers them; pin only after
  fresh validation (--pin flag).
- 2026-11-14: threshold calibration (experiments/calibrate_thresholds.py, tuned on
  seed-2 val, verified on fresh): X tuned thresholds hurt fresh (0.09690→0.09759,
  kept 0.5); Z helped (0.09477→0.09406 val-confirmed, applied). Decoder hybrid
  fresh Z 0.09396, both 0.09546.
- 2026-11-14: sampled data/gen_big2 (2M shots/basis, seeds 2000/2001, 14MB each).
  Pool now 2.6M rows/basis. 70-512-256-4 on it (60ep/pat10, bs32768, seed 3):
  val X 0.08473 / Z 0.08048 (no early stop — still improving at ep 60, mild
  undertraining). Fresh X 0.08338 / Z 0.08189; pinned, manifest stamped.
  Recalibration: no win vs 0.5 either basis (kept). run_local both 0.08276.
  Prev hand-in 0.09546 → now 0.08276.
- 2026-11-14: refactored torch training into models/model_mlp.py (build_torch_model
  + fit_torch + torch_state_to_numpy; lazy torch import, decode env unaffected);
  train_mlp.py is a thin wrapper (+torch.manual_seed for reproducible init).
  Sweep smoke test caught + fixed a real export bug (bare-Sequential keys
  '0.weight' vs assumed 'net.0.weight'); exporter verified both styles,
  torch<->numpy agree 6e-8. New experiments/sweep_mlp.py: 7 configs x 3-fold CV,
  shared rng(11) 300k subset + folds, CSV log + manifest registry. Smoke passed;
  full GPU sweep next.
- 2026-11-14: sweep done, 42 runs on GPU. Mean CV val LER (3-fold, std ≤ 9e-4):
  X: 256x128 .15514 / 512x256 .13974 / 1024x512 .12980 / do.2 .14026 / do.0 .14706
     / 1-layer .17455 / lr3e-4 .17543 (undertrained in 40ep).
  Z: 256x128 .14430 / 512x256 .12842 / 1024x512 .12409 / do.2 .13066 / do.0 .13193
     / 1-layer .15834 / lr3e-4 .15962.
  Lessons: width wins monotonically; depth 2>>1; dropout 0.1 ~= optimal
  (0.0 clearly worse, 0.2 slightly worse); lr 1e-3 beats 3e-4 at equal budget.
  All runs still improving at ep 40 — mild undertraining, ranking consistent.
  Full log: experiments/sweep_results.csv + manifest (sweep tags).
  Winner 1024x512 training on full 2.6M pool (80ep/pat12, bs32768, seed 4).
- 2026-11-14: 1024x512/2.6M finished in ~9.5 min total (282s+278s): val X 0.07790 /
  Z 0.07177, still improving at ep 80 (best at end — 80ep mildly undertrains).
  Fresh pure-MLP X 0.07541 / Z 0.07399; pinned, manifest stamped.
  run_local hybrid both 0.07493, 0.23s/40k (~1.2s/200k, 13x under cap).
  Prev hand-in 0.08276 → now 0.07493. Pure MLP still ties-or-beats hybrid
  (X 0.07541 vs 0.07565); LUT kept as harmless fallback.
