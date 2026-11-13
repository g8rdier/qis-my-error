# AGENTS.md — Project Guide for AI Agents

## What this project is

A decoder for the `[[18,4,4]]` qLDPC code (Wang et al. 2026) under SI1000 circuit-level noise
(p = 0.002, 5 rounds of error correction), for the QEC decoder hackathon (Nov 2026, Kaggle).

**The practical framing:** this is mostly a standard ML problem. We have labelled data
(`data/train/train.csv`, 200k shots) and unlabelled data (`data/test/test.csv`, 200k shots),
70 binary features per row (detector events D0–D69) and 4 binary labels per row
(logical observable flips L0–L3). Predict the labels for `test.csv`.

**But it is also a well-defined linear-algebra problem over GF(2)** (see "The decoding
problem" below), which means we have exact structure (matrices H, L and error probabilities p)
that plain ML doesn't usually get. The best solutions will likely exploit both views.

**We are scored on accuracy AND speed.** The leaderboard metric is LER (lower is better,
see below). The final ranking additionally penalises runtime: the top 10 decoders are timed on
a B200 GPU + 48 CPU cores machine, and the final score is `LER * max(1, t / t_cap)^alpha`. Max score is at 15.4 seconds or faster. But run on the judges' computer, which is much faster than mine. If we can, we should utilize their GPU and threads.
There is a hard limit of 5 minutes. Import-time work (loading weights, building tables) is
**not** timed — only the `decode()` calls are. So: front-load everything into import time.

Hand-ins can include pretrained weights/models, as long as we trained them and provide the code that generated the weights. 

## The decoding problem (the physics/linear-algebra view)

- Hidden binary error vector `e` (~2470 error mechanisms, each 1 with its own probability p_i).
- We observe detector events `D` (70 bits): `D = H·e mod 2`, where H is the parity check matrix (70 × ~2470).
- We must predict observable flips `O` (4 bits): `O = L·e mod 2`, where L is the observable matrix (4 × ~2470).
- Get H, L, p per basis with `problem_info(circuit)` from the `hackathon_2026` package.
  - X circuit: H is (70, 2470), L is (4, 2470), p ranges ~1.3e-4 to ~2.3e-2.
  - Z circuit: H is (70, 2465). (Same shapes otherwise.)
- Key subtlety: the map e → D is many-to-one (degenerate code). The Bayes-optimal prediction
  for a given D is the marginal `P(O_j = 1 | D)` per observable j — i.e. average over all e
  consistent with D, weighted by their probabilities. A decoder that finds the single most
  likely e (max-likelihood, as plain BP does) is not the same thing and is slightly suboptimal
  for degenerate codes. A good ML decoder approximates the marginal directly.
- Baseline: predicting all zeros gives LER ≈ 0.29.

## Data

| Path | What |
| --- | --- |
| `data/train/train.csv` | 200k **labelled** shots (100k X then 100k Z), columns `id, basis, D0..D69, L0..L3` |
| `data/test/test.csv` | 200k **unlabelled** shots, columns `id, basis, D0..D69` — these are the ones to predict |
| `data/train/{X,Z}_circuit.stim` | The noisy circuits (identical in train and test folders) |
| `data/{train,test}/args.json` | Generation settings (both 100k shots/basis, 5 rounds, p=0.002) |

- `id` is `gen_<basis>_<index>` (e.g. `gen_X_00042`). Train and test reuse the same ids — keep
  predictions aligned per row, match on `id` + `basis` column position, not name.
- X rows must be decoded with the X circuit, Z rows with the Z circuit (two separate decoders
  or a basis-conditioned one).
- Detector columns are ordered in time: D0–D6 round 0, D7–D62 rounds 1–4, D63–D69 final
  measurement. `circuit.get_detector_coordinates()` gives (x, y, t) per detector.
- **We can generate unlimited extra training data.** Sampling is cheap (Stim is fast):

  ```bash
  # CLI, from repo root (writes labelled CSV + circuits + args.json; wipes the folder first):
  .venv/Scripts/python.exe data/generate_data.py data/generated --shots 1000000 --seed 7
  ```

  ```python
  # Or in Python:
  from hackathon_2026 import apply_circuit_noise, create_circuit, test_dataset
  circuit = apply_circuit_noise(create_circuit("X", 5), p=0.002)
  detector_events, observable_flips = test_dataset(circuit, shots=1_000_000, seed=7)
  ```

  `--rounds` / `--noise-probability` let you generate at other noise strengths (e.g. train on
  noisier data for more logical flips per shot, or curriculum-train). `--exclude-observable-flips`
  mimics the Kaggle format and writes a separate `solutions.csv`.
- `data/train` was sampled with a fixed seed. If we tune a decoder heavily against it, also
  check the result on a **freshly generated set** (different seed) to detect overfitting to
  that one fixed sample.

## Scoring

- **LER** = fraction of the 4 predicted bits that are wrong, averaged over all shots
  (`logical_error_rate(pred, true)`). Equals Kaggle's MAE for 0/1 predictions. Lower is better.
- X and Z shots count equally.
- **Final score** = `LER * max(1, t/t_cap)^alpha` (t = wall-clock of the two decode calls,
  median of 5 runs; t_cap ≈ 15.4 s for a BP+OSD reference on 32 threads, alpha = 0.2 per the
  current README — check the Kaggle page for updates, the README has two conflicting versions).
  Roughly: each doubling of runtime beyond t_cap costs ~15%. A 5% better LER "buys" ~1.3× t_cap,
  10% buys ~1.7×, 20% buys ~3.1×. So a materially more accurate decoder can afford to be
  somewhat slower, but not much — speed matters.
- Hard limit 5 min. Ties broken by shorter runtime.

## Tools available (the `hackathon_2026` package, `src/hackathon_2026/`)

Import from repo root (`from hackathon_2026 import ...`), installed in `.venv`
(use `.venv/Scripts/python.exe` on this Windows machine; the system Python does NOT have stim).

| Function | Use |
| --- | --- |
| `create_circuit(basis, rounds)` | Build the noiseless circuit for basis "X"/"Z" |
| `apply_circuit_noise(circuit, p)` | Apply SI1000 noise at strength p (competition: 0.002) |
| `problem_info(circuit)` | `{"H": (70, ~2470) uint8, "L": (4, ~2470) uint8, "p": (~2470,) float}` — merged, unique-column DEM |
| `test_dataset(circuit, shots, seed=None)` | Sample `(detector_events, observable_flips)` boolean arrays |
| `load_kaggle_csv(path)` | Load any of the CSVs → dict with `id`, `basis`, `detector_events` (bool (n,70)), `observable_flips` (bool (n,4) if present) |
| `save_submission(path, ids, preds)` | Write `id,L0,L1,L2,L3` in the exact column order Kaggle expects |
| `logical_error_rate(pred, true)` | The leaderboard metric |
| `final_score(ler, runtime, t_cap, alpha)` | The final ranking formula |

Other tooling:

- `submission_template/run_local.py` — runs `submission_template/decoder.py`'s `decode()` on a
  problem set exactly like the judges do (one call per basis, median of 5 runs, reproducibility
  check across runs), reports LER when labels exist, and writes a submission file:

  ```bash
  .venv/Scripts/python.exe submission_template/run_local.py --problem-set data/train --runs 5
  ```

- `submission_template/decoder.py` — the hand-in entry point. Signature:

  ```python
  def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
      # (shots, 70) bool -> (shots, 4) uint8 (0/1)
  ```

  Everything inside `decode()` counts towards runtime; anything done at import/module level
  (loading a trained model or LUT from disk) is free. If there's randomness, seed it — the
  predictions must be bit-identical across runs.
- `notebooks/getting_started.ipynb` — end-to-end walkthrough.
- `notebooks/tutorials/Project.ipynb` — BP tutorial (uses PyMatching as fallback, which is
  **banned** in our decoder; the BP core logic itself is fine to reimplement).
- `data/generate_data.py` — CLI data generator (see above).

## Rules that constrain implementation

- **Write the decoder ourselves.** Banned: `pymatching`, `ldpc`, `stimbposd`, `beliefmatching`,
  `tesseract-decoder`, `fusion-blossom`, and any code copied from them or pretrained models.
  Allowed: `numpy`, `scipy`, `numba`, `pytorch`, `jax`, `scikit-learn`, graph libs, optimisation
  solvers. (Tutorial notebooks use banned libs — don't carry that code over.)
- Unlimited self-generated training data is explicitly allowed.
- Predictions must come from the decoder; no hand-labelling or leaderboard probing.
- Up to 10 submissions/day, 2 final selections.

## Decoder ideas

### 1. Lookup-table (LUT) decoder

Train: sample many labelled shots, store `D → O` (e.g. majority-vote O per observed D pattern,
which approximates the Bayes marginal per pattern). Decode: exact match on the detector pattern;
fallback for unseen D (predict all zeros, or nearest-seen pattern, or fall back to BP/ML).

- Detector events are sparse at p=0.002 (most shots fire 0–3 detectors), so the number of
  *distinct* D patterns with non-negligible probability may be small (order 10³–10⁴). Measure
  the pattern-frequency distribution first — if e.g. 50k sampled patterns cover 99.9% of the
  probability mass, a LUT is nearly optimal.
- Storage idea: pack the 70 bits into one 128-bit int (or two uint64s) as a key; use a dict or
  a sorted numpy array. Import-time build is free; lookup is O(1) hash or O(log n) binary
  search — both trivially under t_cap.
- The fallback for unseen patterns is where the accuracy is won or lost; consider
  "most likely O among patterns within Hamming distance 1–2" as a cheap soft fallback.

### 2. Belief propagation (BP) decoder

No training data needed. Runs message passing on the factor graph defined by H and the priors p,
to estimate `P(e_i = 1 | D)` per mechanism. If the argmax ê satisfies `H·ê = D`, predict
`Ô = L·ê mod 2`.

- BP on qLDPC codes often doesn't converge / gets stuck in symmetric fixed points. Standard
  fix in the literature is BP+OSD (ordered statistics decoding) as post-processing: use BP's
  soft probabilities to reorder columns of H, then solve `H·ê = D` greedily preferring the most
  likely errors, e.g. by row-reduction over GF(2) + a few sweep steps. We'd implement OSD
  ourselves (pure numpy GF(2) linear algebra — allowed, it's not a QEC-decoder library).
- Alternatives on failure: perturbation/damping heuristics, random schedules, or fall back to
  LUT/ML. A hybrid (BP for common patterns, something else for the tail) is plausible.
- BP+OSD is what the organisers' reference decoder is (t_cap = its runtime on 32 threads),
  so matching its speed with 48 threads should be feasible. It's also the accuracy bar to beat.

### 3. Classic ML decoder

Treat it as multilabel binary classification: 70 binary features → 4 binary outputs,
metric = per-bit error rate. Unlimited training data (generate millions of shots), so the
model class matters less than features + capacity + inference speed.

Candidate architectures, roughly in increasing power/cost:

- **Linear / logistic regression over GF(2)-ish features** — extremely fast (one matmul),
  probably too weak alone, but a great baseline and ensemble member.
- **Gradient boosting (XGBoost/LightGBM)** on the 70 bits — handles sparse binary features
  well; per-shot inference can be slow unless batched carefully.
- **Small MLP** (numpy or torch): 70 → hidden → 4, trained on the per-bit BCE loss with the
  label being the *true* O. With enough data this directly approximates `P(O_j | D)`, i.e. the
  Bayes-marginal decoder — which is the theoretical optimum. Batched inference on 100k×70 is
  milliseconds on CPU, comfortably under t_cap. Train per basis (two models) or concatenate
  with a basis feature.
- **Input features to consider:** raw 70 bits; basis one-hot; per-round parity features
  (XORs of detector groups from H's structure); syndrome "weight" counts; optionally a BP
  posterior as an extra feature (hybrid decoder).
- **Loss:** predict P(O_j=1|D) with BCE, threshold at 0.5 at inference (optimal for per-bit
  error). Consider weighted training on rare nonzero-O shots since all-zero dominates.

This computer has an NVIDIA RTX 3050 Ti available to train.

### 4. PCA / dimensionality-reduction analysis (idea to explore)

- Run PCA (or just count) on the 70 detector bits across many sampled shots: the covariance
  spectrum tells us the effective dimensionality of the syndrome distribution and how many
  distinct patterns carry real probability mass. This directly informs LUT feasibility
  (see 1) and the size of any table-based approach.
- Note: data is binary and sparse — plain PCA is linear; also consider correspondence
  analysis / categorical embeddings, or just empirical pattern-frequency statistics, which for
  binary sparse data are often more informative than PCA.
- If a low-dimensional structure exists (say ~20 components capture the mass), a tiny model on
  the projected features could be both fast and accurate. Also useful for visualising clusters
  of error mechanisms and sanity-checking decoder confusion.
- Notably: the column space of H (over GF(2), not R) is what actually constrains D. PCA over
  the reals is a different thing — treat it as an exploratory/feature tool, not as capturing
  the code structure.

### 5. Sorted-table + binary-search decoder (the "generate, sort, lookup" idea)

The user's concrete proposal: generate a huge labelled dataset, sort the rows by their detector
pattern, and at decode time binary-search for the exact pattern.

- Pack each 70-bit D into a 128-bit key (numpy: view as 2×uint64 or use `np.uint128` via
  object/str fallback; or store as a 9-byte big-endian integer). Sort keys with `np.argsort`
  / `np.sort` once at import; store keys + label bytes in flat arrays.
- Decode: `np.searchsorted` on the sorted keys → O(log n) per shot, vectorised over shots;
  effectively instant for 100k shots. Total decode time dominated by the bit-packing, which
  is a bitcast, so this is well under t_cap.
- The real questions are (a) coverage: what fraction of *test* patterns appear in the table
  as n grows (measure the hit-rate curve vs. table size — heavy tail of rare patterns);
  (b) the fallback for misses. Misses are exactly the high-weight, informative syndromes,
  so the fallback quality determines the endgame LER.
- Natural extension: **k-nearest-neighbour by Hamming distance** on the sorted table
  (search a window around the binary-search insertion point, since sorted-by-integer-order
  roughly clusters similar patterns only partially — Hamming neighbors are NOT adjacent in
  integer order, so instead consider: for each of ≤70 single-bit flips, search for the flipped
  key; that's 70 binary searches per miss, still fast). Or build multiple sorted tables, one
  per "one detector masked out", to get distance-1 neighbours cheaply.
- Another extension: store not just the majority label but the empirical P(O|D) per pattern,
  and let a small model handle the tail. The sorted table then becomes a probability cache
  for the head of the distribution plus ML for the tail.

### 6. Other ideas worth keeping on the list

- **Exact enumeration / syndrome marginalisation:** the number of mechanisms (~2470) is too
  large to enumerate all e, but per-syndrome exact marginals via the transfer-matrix /
  trellis structure of the 5-round circuit might be tractable for common low-weight syndromes.
- **Correlating X and Z decoders:** the two circuits are separate problems; two independent
  decoders (or one shared model with basis input) is fine.
- **Ensembling:** LUT (head patterns) + ML (everything) + BP (tail) combined by predicted
  probability; keep inference under t_cap by routing.
- **Noise-strength curriculum:** train on slightly-higher-p data (more logical flips, better
  class balance), test at p=0.002; also data augmentation = XOR-injection of extra detector
  flips consistent with H.
- **Hard-coded algebra:** since H is only 70×~2470 and L is 4×~2470, a precomputed
  pseudo-inverse / row-reduced basis of the kernel of H (GF(2)) gives all coset leaders
  structures; small enough to compute offline and bake into the decoder.

## Practical notes / conventions

- Python env: `.venv` (uv-managed, `uv sync`); on Windows run everything with
  `.venv/Scripts/python.exe`. Package `hackathon_2026` installed editable from `src/`.
- Always evaluate on `data/train` (labels available) AND a freshly generated set with a new
  seed to detect overfitting to one fixed sample.
- Time every decoder with `submission_template/run_local.py` before trusting its speed;
  target well under t_cap (15.4 s), ideally < 1 s for table/matmul decoders.
- Any randomness in `decode()` must be seeded; predictions must be identical across the 5 runs.
- Kaggle pages override this repo's README where they differ.
- Deliverable per run: `submission_template/decoder.py` + `requirements.txt` + any data files
  it loads + the code that generated them.
