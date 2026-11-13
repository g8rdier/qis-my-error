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

The model initializes at import. Calls are deterministic and CPU-parallel;
48 judge threads are supported. Prior local timing was 48.98s / 200k shots on
16 laptop threads, **not** a B200/48-core timing.

## Runtime improvement ideas (not implemented)

The measured local baseline is **48.98s per 200k shots** (median of five,
16 laptop threads). At perfect 3x thread scaling that would be about 16.3s on
48 threads, close to the 15.4s reference cap; real B200/48-core timing is
unknown. Since the ensemble gains only **0.14% relative LER**, reducing runtime
has unusually high leverage on the final score. The ideas below are sketches,
not measured wins. Each needs paired accuracy and five-run timing on fresh data.

### 1. Do not run both components on every shot

This is the first thing to try. Run the chronological component for all shots,
then run the shuffled component only where the first result is uncertain.
Thresholds can be calibrated by replaying the 500k development/audit predictions
(the cache lives in the research branch) or by regenerating equivalent samples;
no new labels are needed. The second component is roughly half the work,
so routing at a 10–20% uncertainty rate could remove 35–45% of runtime. The
cost is part of the ensemble's small accuracy gain, which must be measured,
not assumed.

Also test a **single-component mode** using the already-developed
hybrid/tighter search: it measured **31.47s** and LER **0.033560** on the 500k
comparison, versus 48.98s and **0.033548** for the ensemble. That is a 36%
runtime cut for 24 extra wrong bits out of 2M. Once runtime is scored, this may
be the better operating point. A `combination: "single"` switch in `model.json`
would make it a small, reversible change.

### 2. Add a safe head fast path

About **4.6%** of shots have no fired detectors and **28%** have at most four
(4.6% zero plus 23% one-to-four). Precompute 16 logical-outcome weights offline
with the full kernel for the covered syndromes that fit a chosen table budget,
store them in `assets.npz`, and use them only when the margin is safely outside
the table's truncation error. Fall back to the full dynamic programme
otherwise. Precompute probabilities, **not majority-vote labels**, so the
per-bit marginal decision is preserved. Build the table from physical faults or
development seeds only—never from test labels—and validate on untouched seeds.

### 3. Stop a shot as soon as its answer is fixed

Add a suffix logical-influence mask: for each sweep step, know which logical
bits any remaining mechanism can still flip. If all surviving states agree on a
bit and no remaining column can flip it, that bit is final. When all four bits
are final, emit the result and skip the rest of the shot. A weaker variant is
to stop when all surviving states share one full 4-bit class and the remaining
suffix cannot introduce another class. Both are exact for the pruned beam if
the mask is conservative. Require bit-identical predictions against the current
kernel on the full 500k audit set before accepting them.

### 4. Reduce kernel and cache overhead

Candidate, accuracy-preserving micro-optimizations:

- Store hash-table indices as `uint16`/`uint32` instead of `int64`. At
  `max_states <= 4096` the table fits comfortably and the smaller footprint
  should reduce cache misses.
- Rebuild the hash table with generation stamps instead of clearing and
  reinserting every state at each compaction.
- Replace the full top-k `argsort` with `partition` when the state cap is hit.
- Specialize the compiled kernel for the fixed `70` detectors and `4`
  observables instead of passing them as runtime arguments.
- Pack `D0/D1` vectorized in NumPy once per batch instead of looping over 70
  detectors inside every shot.
- Test `float32` state weights/scores. This is the riskiest item: it can change
  rare threshold decisions, so it requires exact-prediction and full-audit
  checks, not just a lower timing number.

Validate each change independently. A 5–15% kernel improvement is useful, but
not at the cost of even a few extra wrong logical bits.

### 5. Improve 48-thread load balance

The kernel uses static contiguous `prange` ranges. If hard shots cluster, a few
threads finish last while the rest idle. Flatten `(component, shot)` into one
parallel batch, especially when routing leaves few shots in the second stage,
and assign work in a cost-aware interleaved order (for example, round-robin by
fired-detector count). Do **not** simply sort all easy and hard shots together:
the earlier sorted-deduplication experiment was about 19% slower. Preserve
random input order when in doubt. Re-time on 200k shots, and ideally on a
48-thread machine.

### 6. Revisit the ensemble before optimizing it

Compare LER-versus-runtime points on the same held-out set:

- original reference MLD: 21.53s (one diagnostic run), LER 0.033595;
- hybrid/tighter single component: 31.47s, LER 0.033560;
- two-component ensemble: 48.98s, LER 0.033548.

The ensemble buys **0.000012 LER** over the hybrid—24 fewer wrong bits out of
2M—while costing 56% more. A slightly wider **single** beam may dominate both.
Optimize the scored objective `LER * max(1, t / t_cap)^alpha`, not LER alone.

### 7. GPU is a later, larger project

The dynamic programme is branchy, hash-table heavy and serial within a shot, so
it is not a drop-in GPU port. More plausible uses are a GPU-scored head table,
or a batched dense beam used only for hard shots. The B200 has enormous compute,
but a poor GPU mapping can easily lose to 48 CPU threads. Treat this as a
research track after the CPU routing and kernel work, not as an expected quick
win.

### Measurement protocol

Use new development and untouched audit seed families, paired shot-level
confidence intervals, and median-of-five `run_local.py` timings. Keep
predictions deterministic and bit-identical across runs. Import-time
precomputation is free in the competition, but it must not read the test labels,
`results.csv`, or memoize test predictions. Report the accuracy/runtime Pareto
frontier rather than a single cherry-picked setting.

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
