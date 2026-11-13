# Plan: release only the best pretrained decoder to main

Status: planning only. No main-branch merge, test predictions, new model training,
or Kaggle upload is performed by this plan.

## 1. What is already trained and tested?

Model: `physical_pair_closure_class2`, frozen configuration
`candidate/accuracy_closure.json`.

- The learned lookup tables were built from **10M simulated shots per basis**
  (20M total), seeds 20261113/20261114.
- BP/OSD, candidate union and physical-pair closure are **not neural networks**.
  They use the supplied circuit's H/L/p and do not improve simply by sampling
  more training rows.
- Seven development sweeps: 83 named settings, 534 basis/seed measurements
  including repeated references/aliases, mostly reusing the same 48k development
  shots. Those are NOT 534 independent test sets.
- The final closure model was frozen before its **300k-shot independent audit**,
  seeds 830100–830105. LER 0.03414583 versus 0.03506333 reference.
- Closure improves all six basis/seed groups. Paired delta versus reference:
  -0.00091750, 95% CI [-0.00107382, -0.00076118].
- Its five full-size timing runs used a **200k subset of that same audit**,
  yielding LER 0.03403, identical predictions, 56.86s local median.
  Repeated timing runs are NOT additional independent accuracy samples.
- Train-set reference: 200k shots, LER 0.03443.
- Nineteen correctness tests and a standalone runtime-dependency smoke test pass.
- The earlier 720100 audit tested predecessor configurations, not this final
  closure model. Do not add it to the final model's independent test count.
- No Kaggle test CSV was used for fitting or model selection. Judge hardware
  timing and optimality remain unverified.

Conclusion: a well-tested release candidate, not a proven optimal decoder.

## 2. Generate more data where it is useful

1. Freeze the current configuration/assets as the reference before any change.
2. Generate a new development set, separate from all existing audit families.
   Measure error contributions from lookup-served versus physics-served shots.
3. Build a challenger lookup from **50M fresh shots per basis** (100M total),
   streaming packed chunks and retaining only detector weight <=4. Store counts
   and observable counts during construction; do not save giant training CSVs.
   Keep the physics search configuration fixed for this comparison.
4. Select between the original 10M and challenger 50M tables on development
   data only. More samples can improve the head table, but can also introduce
   noisy rare-pattern matches; a larger table is not automatically better.
5. Freeze the chosen release and evaluate on **1M fresh shots per basis**
   (2M total), across multiple independently seeded streams. These final
   validation seeds are reserved before development and never used for tuning.
6. Report paired shot-level intervals, basis/bit errors, and lookup/search
   contributions. Keep the original assets if the challenger does not earn a
   convincing improvement. If the final audit fails, do not retune on it.

Generating data does not itself train the physics solver. Any further search
algorithm changes need their own development experiment and sealed audit.
No claim that the final model is "as good as possible."

## 3. Minimal, replaceable folder on main

Proposed path: `decoders/best/`. Stable public interface:

    decode(circuit, detector_events) -> uint8 array, shape (shots, 4)

Proposed contents (11 files):

    decoder.py          # public decode API; asset loading, routing, warmup
    bp_osd.py           # BP checkpoints, GF(2) basis and candidate search
    closure.py          # physical pair index and equivalent-pair expansion
    candidates.py       # exact deduplication, bounded pool, weighted voting
    assets.npz          # BOTH bases: saved physics graphs + trained head LUTs
    model.json          # single model ID/config, seeds, versions, hashes, metrics
    prepare.py          # optional build/verify/predict release utility
    test_decoder.py     # compact invariant/API/regression tests
    requirements.txt    # numpy/stim/numba pins; no PyTorch required
    README.md           # usage, validation summary, reproduction instructions
    results.csv         # frozen release predictions on data/test/test.csv

Only the winning inference path is retained. No old MLPs, other decoder presets,
experiment logs, datasets, caches, or copied research notebooks.

The existing 36-file hand-in is a working export, not the minimal main package.
Refactoring into the above modules must preserve the frozen decoder's outputs.
Use within-folder imports compatible with both normal repository imports and
the judge's `import decoder` entry point.

### Assets can be small

Measured from the current 10M tables:

| Basis | All table entries | Entries actually trusted (<=4 fired) |
|---|---:|---:|
| X | 5,976,175 | 102,447 |
| Z | 5,858,853 | 101,087 |

The decoder never trusts heavier-pattern entries. Discarding those entries
cannot change its decisions. Their outputs are always replaced by physics
decoding anyway.

- Original LUT files: 118,351,296 bytes total.
- Retained lookup payload: 2,035,340 bytes before compression.
- In-memory compression measurement including both existing graphs:
  **623,188 bytes** for one combined NPZ.

That is a size estimate for the CURRENT trained model, not a file already
prepared for release. A larger challenger table may have a different size.
Use normal Git for a similarly small asset; no need for Git LFS unless the
actual chosen asset size warrants it.

All prediction-affecting assets ship with the model. No one must train or
generate data to use it. Graph/pair-index initialization and Numba compilation
on first import are setup, not training; keep them outside timed decode calls.
Optional rebuilding code remains included as the rules require.

## 4. Refactor and performance gates

- Compare the lean/pruned package against the frozen selected model on all
  archived final-audit detector inputs, without reading truth labels for this
  equivalence check. Require zero prediction differences for packaging-only work.
- Run the unit/invariant tests and fresh-environment smoke test.
- Ensure deployment needs only declared packages and packaged assets; no
  organizer helper, repository paths, training CSVs or experiment cache at runtime.
- Verify deterministic predictions, both bases, row permutations, empty inputs,
  and different batch sizes.
- Benchmark cold import separately from five warmed full-size decode runs.
- Honor configured CPU thread counts, including 48 on the judging rig. Laptop
  15.4s is NOT an acceptance cutoff. GPU/B200 performance is not yet established.

## 5. Generate and verify results.csv

Only after the model/assets/code are frozen:

1. Read the unlabelled `data/test/test.csv` and the corresponding X/Z circuits.
2. Decode each basis with its matching physics graph.
3. Scatter predictions back into the original CSV row order; do not join to
   train data or assume reused train/test IDs identify the same shots.
4. Write exactly:

       id,L0,L1,L2,L3

5. Require 200k prediction rows, binary values only, no missing/duplicate test
   IDs, correct output-column order, and exact alignment with the sample
   submission/test IDs. No index/basis/detector columns in the output.
6. Regenerate once and require identical bytes or parsed predictions.
7. Record the input CSV, model assets, configuration, source and results hashes
   together in model.json. Any prediction-affecting change invalidates results.

Test labels are unavailable, so this produces a submission, NOT a measured
test LER. Do not tune on test rows or leaderboard feedback.

## 6. Bring only this folder to main

1. Keep all experiments on `ml-models`; do NOT merge that entire branch.
2. After approval, create a clean release branch/worktree from up-to-date main.
3. Copy only `decoders/best/`, including its ready-made assets and results.csv.
4. Install from the folder's requirements and rerun the fresh-clone smoke tests
   and submission-format checks from that worktree.
5. Stage only the explicit model folder; inspect the diff and file sizes.
6. Commit that isolated addition and review/merge via a PR to main.
7. Kaggle upload is a separate explicit action. Keep the code hand-in package
   ready with the same model/assets/results identity.

The main branch's existing runner already accepts `--decoder <folder>`, so the
folder can be selected without migrating the ML research code. A future winner
can replace this folder while preserving the same decode API and CSV schema.

## Go/no-go checklist

- [ ] Original model frozen and checksummed.
- [ ] Additional simulated development/training/final-audit data kept separate.
- [ ] Chosen head table justified; best known physics settings retained.
- [ ] Lean package matches frozen model for packaging-only changes.
- [ ] New final audit completed and reported honestly.
- [ ] Runtime and clean-environment checks pass.
- [ ] Prebuilt assets included; regeneration optional.
- [ ] results.csv generated from the exact frozen release and checked.
- [ ] Only the requested model folder appears in the main-bound diff.
- [ ] User approves commit/merge and any Kaggle upload.
