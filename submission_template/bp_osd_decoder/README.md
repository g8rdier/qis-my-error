# BP+OSD decoder (ported from branch `lut-bp-osd-decoder`)

Two-stage decoder that holds this repo's best honest LER (~0.035 vs the ML
decoder's 0.075): an exact-match lookup table for easy shots (≤4 detectors
fired, built from 10M fresh Stim shots per basis) + a from-scratch
belief-propagation + ordered-statistics decoder for everything else.

**Latest improvement:** a new sealed 300k-shot audit measures reference
**0.03506333**, previous full union **0.03432250**, and physical-pair closure
**0.03414583** (2.62% relative improvement over reference). All six seed/basis
groups improve. See `EXPERIMENTS.md` for the sequential
ablation record, paired intervals, failures, and full-size timings.
The original reference below is preserved; the improved entry is in `candidate/`.

## Provenance

- `bp_osd.py` and `build_lut.py` are **line-equivalent source copies** of
  `submission_template/{bp_osd,build_lut}.py` on branch `lut-bp-osd-decoder`
  (commits `1f4f334`…`94f0fc3`). PowerShell may change BOM/line endings;
  the algorithm was not edited. Experimental extensions are separate.
- `decoder_bposd.py` is that branch's `decoder.py` decode logic, adapted only
  for this folder (`LUT_DIR` points here, imports are package-relative).
  Same `(circuit, events) -> (n,4)` contract.
- `run_bposd.py` is a `run_local.py`-style harness scoped to this decoder.
  It **refuses `data/test`**: evaluation needs labels, test stays decode-only.
- The ML decoder in `submission_template/decoder.py` + `models/` is untouched.

## Their reported numbers (from that branch's commit messages)

| Eval set | LER (both) | X | Z | Runtime |
|---|---|---|---|---|
| data/train (200k) | 0.03522 | 0.03574 | 0.03471 | median 7.6 s on 8-core laptop |
| fresh seed-7 | 0.03473 | — | — | — |
| fresh seed-8 | 0.03520 | — | — | — |

Train/fresh are consistent; this is evidence against severe leakage, not
proof of absence of tuning overfit. Their LUT trains on fresh Stim seeds
(20261113/20261114), BP+OSD trains on nothing (built from H/L/p at decode
time). No test CSV/label reads in the audited decoder/build code; no banned libraries
(`pymatching`, `ldpc`, … — only `numpy` + `stim` + `numba`, all allowed).

## Build the tables (one time, ~113 MB, gitignored)

```powershell
.venv/Scripts/python.exe submission_template/bp_osd_decoder/build_lut.py --shots 10000000 --chunk 2000000
```

Defaults: circuits from `data/train`, output to `bp_osd_decoder/lut/`.
Smaller `--shots` works for a quick pipeline check but lowers table coverage.

## Evaluate (labelled sets only)

```powershell
.venv/Scripts/python.exe submission_template/bp_osd_decoder/run_bposd.py --problem-set data/eval_fresh --runs 5
.venv/Scripts/python.exe submission_template/bp_osd_decoder/run_bposd.py --problem-set data/train --runs 3
```

## Verified numbers on this branch (ml-models)

| Eval set | LER (both) | X | Z | Runtime | Date |
|---|---|---|---|---|---|
| data/eval_fresh (40k) | 0.03570 | 0.03585 | 0.03555 | median 1.73 s (run 1: 23 s incl. numba JIT; cached after) | 11-14 |
| data/train (200k) | 0.03529 | 0.03583 | 0.03475 | median 7.14 s | 11-14 |

(Their claims: train 0.03522, fresh seeds 7/8: 0.03473/0.03520.
Different samples need not agree to 1e-4; train was reproduced within
0.00007. The reference's import warning comes from intentional uint64
wraparound in its De Bruijn table. All 64 bit positions have now been
unit-tested. LUT files and Numba's compiled-kernel cache are distinct.)

## How it works (short version)

See `next-steps.md` for the full deep dive. In one paragraph: the table
answers syndromes it has memorised (majority vote ≈ Bayes-optimal per
pattern) for the easiest shots; for the rest, damped normalised-min-sum BP
estimates per-mechanism posteriors, OSD-0 takes the most-likely independent
column set of H and solves `H·e = D` exactly, a singles+pairs combination
sweep re-scores candidates by total coset probability, the per-bit
most-probable outcome is predicted, and still-uncertain shots (from OSD's own
`classp` marginals) get a second, wider sweep.

## Improvement experiments

- `EXPERIMENTS.md`: hypotheses, sequential outcomes (including failures),
  validation discipline, and explanations.
- `experiments/results/`: reproducible per-seed JSONL and summary tables.
- `enhanced.py`: third-order candidate search and sum-product BP ablations.
- `union.py`: deduplicated full-error-vector candidate union.
- `adaptive.py`, `fused.py`, `bp_checkpoints.py`: uncertainty routing and
  prediction-preserving shared BP computation.
- `pool.py`: exact-collision-safe candidate indexing and bounded-pool heap.
- `local_closure.py`: physical-pair replacements preserving the syndrome,
  scored with original priors.
- `candidate/`: separate import-precomputed hand-in entry point. Its JSON
  configuration is promoted only after independent holdout validation.
- `tests/`: exhaustive toy posterior checks, collision handling,
  baseline equivalence, and API/determinism tests.

Run the improved default without modifying the ML decoder:

```powershell
.venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/bp_osd_decoder/candidate --problem-set data/train --runs 5 --threads 16 --out submission_template/bp_osd_decoder/experiments/cache/train_predictions.csv
```

`candidate/config.json` selects the audited accuracy-first
`accuracy_closure.json` model. `balanced.json`, `fast.json` and `accuracy.json`
preserve earlier operating points. `candidate/MODELS.md` maps every file to its
model, parameters and audit. `experiments/benchmark.py --config <filename>`
selects a configuration and retains logs/source hashes.
On the judges' 48-core machine use `--threads 48`; all heavy kernels parallelize
over shots. These implementations use CPU threads, not the GPU. Laptop times
cannot establish the score penalty or thread scaling on the B200 judging rig.

## Standalone hand-in

`handin/` is built by `build_handin.py` (gitignored generated output).
It contains a root `decoder.py`, exact dependency pins, graph/LUT artifacts,
their generators, circuit inputs and a SHA-256 `bundle_manifest.json`.
It includes no sample CSVs or unrelated ML weights. Rebuild to a new/empty
output directory; the builder refuses overwrites.

The latest model's local five-run median is 56.86s for 200k shots, with
identical predictions. This is a laptop measurement, not judge timing.
Nineteen tests pass; standalone import/decode was verified with the
organizer's build-only package deliberately blocked.
