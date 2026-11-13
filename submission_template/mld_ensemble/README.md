# Ready-to-use MLD ensemble

Everything for this decoder is in this folder. **No training is needed before
use.** Runtime dependencies are only NumPy, Stim and Numba; it does not import
the parent decoders, experimental folders, or `hackathon_2026`.

## Data and accuracy

This is an **analytic physics decoder**, not a fitted neural network:
**0 fitted training shots and no neural weights**. The frozen `assets.npz`
(22 KB) contains both circuits, their H/L matrices, physical error probabilities
and detector permutations. Generating extra shots does not fit new parameters.

- Settings selected using 66,000 development shots.
- Earlier independent evaluation: **500,000 shots, LER 0.033548**.
- New release evaluation: **2,000,000 newly generated shots** (1M/basis,
  seeds 1086100/1086101): **LER 0.03364375**, shot-level 95% interval
  **0.03345966–0.03382784**. Detailed results: `validation.json`.
- Supplied labelled `data/train` reference: **LER 0.033860** on 200,000 shots.
- No test labels or leaderboard feedback are used.

This folder is about **4.1 MB**, mostly the requested 4 MB prediction CSV.
All eight tests pass; exported predictions match the research model exactly.

## How it works

It sums likely physical-error explanations with a pruned dynamic programme.
One component orders checks chronologically; the other uses a fixed
within-round shuffle. Each runs a forward sweep and refines uncertain shots
backwards. Their absolute logical-outcome weights are combined by a maximum,
then each output bit is selected by its marginal probability. This is an
approximation, not a mathematically exact posterior or deduplicated union.

The MLD tiers, heuristic exponent, and compaction interval are configured
independently for the X and Z bases. The full-data optimization notebook
(`submission_template/mld_full_data_optimization.ipynb`) searches those
parameters separately and reports per-basis LERs.

The model initializes at import. Calls are deterministic and CPU-parallel;
48 judge threads are supported. Prior local timing was 48.98s / 200k shots on
16 laptop threads, **not** a B200/48-core timing.

## Use / swap models

```powershell
.venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/mld_ensemble --problem-set data/train --runs 5 --threads 16 --out check.csv
```

To select it as the main decoder, replace `submission_template/decoder.py`
with the included `entrypoint.py`, or simply use this module-level import:

```python
from mld_ensemble.decoder import decode
```

Equivalently: `Copy-Item submission_template/mld_ensemble/entrypoint.py submission_template/decoder.py`.

The main entry point is deliberately **not changed** by this release, so
colleagues can replace this one folder or choose another model independently.

`results.csv` is the prediction for `data/test/test.csv`, with header
`id,L0,L1,L2,L3`, original input order and binary values. **Inference never reads
this CSV**, validation reports or saved predictions.
`model.json` records the input/output hashes; `manifest.json` fingerprints the
release files. Input CSV hashes tolerate LF/CRLF checkout differences.

## Reproduce

Use a **new** data output directory; `generate_data.py` wipes existing files.

```powershell
.venv/Scripts/python.exe data/generate_data.py data/mld_generated --shots 1000000 --seed 1086100
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py evaluate --data data/mld_generated/mld_generated.csv --out NEW_VALIDATION.json
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py predict --data data/test/test.csv --out NEW_RESULTS.csv
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py verify --data data/test/test.csv --out submission_template/mld_ensemble/results.csv
.venv/Scripts/python.exe submission_template/mld_ensemble/prepare.py build --out NEW_ASSETS.npz
.venv/Scripts/python.exe -m unittest discover -s submission_template/mld_ensemble -v
```

`build` can also take `--circuits data/train`. It rebuilds the small physics
asset, not a fitted model. Tools refuse to overwrite existing outputs. Pin the
versions in `requirements.txt` when reproducing samples and assets.
For a new release, rerun validation and prediction, update `model.json`, then
run `prepare.py manifest` in a fresh release folder. Do not mix versioned assets.
