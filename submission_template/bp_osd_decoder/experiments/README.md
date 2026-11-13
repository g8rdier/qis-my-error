# Reproduce the improvement sequence

All commands run from the repository root with `.venv/Scripts/python.exe`.
Original LUTs must first be built with `../build_lut.py` (10M shots/basis).
No script accepts a test CSV for fitting or parameter selection.

1. **Correctness tests** (small models enumerate all hidden errors):

   ```powershell
   .venv/Scripts/python.exe -m unittest discover -s submission_template/bp_osd_decoder/tests -v
   ```

2. **Reference/parameter/MLP ablations**:

   ```powershell
   .venv/Scripts/python.exe submission_template/bp_osd_decoder/experiments/search.py --name stage1
   ```

3. **Triples and sum-product BP**: pass `--config-file .../stage2.json`
   and `--ids baseline triple_bp20 triple_bp40 triple_bp60 triple_single40
   triple_refine60 triple_iter5 iter5_pairs300 iter5_refine_wider spa3 spa5 spa10`.

4. **Candidate union / combined settings**: pass `--config-file .../stage3.json`
   and the desired configuration names from that file, plus `baseline`.

   Subsequent sequential studies use `stage4.json` (global order-three
   enumeration), `stage5.json` (routed compute), `stage6.json` (accuracy-first
   larger searches and ranking perturbations), and `stage7.json` (physical
   pair closure). Each complete run has a separate result filename and journal
   explanation. Do not discard a candidate just for exceeding 15.4s locally.

5. **Freeze first, then audit**: after selecting on development results only,
   evaluate the frozen configuration with `--split holdout --shots 50000
   --folds 3`. Do not search on those results. `make_audit_set.py` then exports
   exactly 100k/basis from the same holdout samples to `data/bposd_audit`.

6. **Official-shaped end-to-end timing**, once candidate/config.json is frozen:

   ```powershell
   .venv/Scripts/python.exe submission_template/bp_osd_decoder/candidate/build_graphs.py
   .venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/bp_osd_decoder/candidate --problem-set data/bposd_audit --threads 8 --runs 5 --out submission_template/bp_osd_decoder/experiments/cache/audit_predictions.csv
   ```

   Repeat on `data/train` as an additional reference. The primary ML
   `submission_template/decoder.py` is never overwritten.

The later closure audit is frozen in `finalists_stage7.json` and uses
`--split holdout --seed-base 830100 --shots 50000 --folds 3`.
Export it with `make_audit_set.py --seed-base 830100 --out data/bposd_audit2`.
That set is now a regression/audit set, not new tuning data.
`paired_audit.py` compares frozen options using the archived predictions.

## Data and artifacts

- `cache/`: self-generated samples and per-configuration predicted marginals;
  gitignored, reproducible from seeds, circuit files, pinned dependencies and
  source hashes. Source/circuit/parameter changes produce different cache keys.
- `results/*.jsonl`: per-seed settings, errors, runtimes and paired differences.
- `results/*.summary.{json,md}`: averages, shot-level paired intervals and
  leave-one-seed-family-out selection checks.
- `../candidate/config.json`: promoted configuration, separate from reference
  defaults. No candidate can silently overwrite an MLP weight artifact.
- `../candidate/MODELS.md`: exact configuration/model/audit identities.
- `../candidate/artifacts/`: graph arrays, circuit fingerprints and generating
  code in `build_graphs.py`. BP+OSD has no trained neural-network weights.
- `../handin/`: generated self-contained hand-in, built by `../build_handin.py`;
  no sample CSVs/caches included. Refuses to overwrite an existing output.

Default development seeds are a different family from all LUT/MLP sample seeds.
Runtime numbers in the search table are diagnostics, not judge-shaped medians:
they exclude graph/table preparation and may overlap CPU work on tests.
