# Runtime optimization with frozen predictions

The selected submission is `submission_tweak_tieband01.csv`, SHA-256
`878631d6b0a31f7c2ec06d55b58fb82b7f6b8aa26f27e7321f5ebb63b08a17a1`.
Every retained change must reproduce every shot and all four logical bits.
Matching aggregate LER is insufficient.

The optimization in `mld.py` extends an existing condition from the small-state
loop to the vectorized loop. When `hasv0[k]` is false, every `cv0` coefficient
for that mechanism is exactly zero. The vectorized loop leaves the initialized
zero `ce` values alone instead of repeatedly multiplying and adding zeros.
Updates to `cs` retain their original operations and detector order. When a
nonzero `cv0` exists, the original arithmetic path runs unchanged.

`decoder.py`, physical probabilities, all three stages, pruning thresholds,
state caps, compaction frequency, refinement decisions, precision, and the
`TIE_BAND=0.01` output rule are unchanged. The scheduling, smaller-integer,
and native-bit-scan trial was discarded because its measured speed difference
was within observed variation. It is not part of this change.

## Reproduce the selected-submission check

Install the **hand-in** requirements from `submission_template/requirements.txt`.
The repository-root requirements are for development and include an editable
SSH checkout; they are not needed by the standalone decoder.

```bash
python3.13 -m venv /tmp/qec-verify
/tmp/qec-verify/bin/python -m pip install -r submission_template/requirements.txt
/tmp/qec-verify/bin/python -I submission_template/verify_runtime.py \
  --data data/test/test.csv \
  --submission submission_tweak_tieband01.csv \
  --threads 48 --runs 5 --report /tmp/runtime-check.json
```

The verifier imports before timing, calls `decode` once per basis per run,
checks every output bit on every run, and additionally requires a byte-identical
CSV in the original input order. It reports the total X+Z runtime and source
hashes. It uses only the hand-in dependencies and does not import the research
project package or use labels. Local timings are not official-machine timings.

For intermediate numerical comparisons, run `verify_runtime_parity.py` once
per decoder revision in separate processes, with the same `--threads` setting:

```bash
/tmp/qec-verify/bin/python -I submission_template/verify_runtime_parity.py \
  --decoder /path/to/reference-folder --threads 16 --out /tmp/reference.npz
/tmp/qec-verify/bin/python -I submission_template/verify_runtime_parity.py \
  --decoder submission_template --threads 16 --out /tmp/candidate.npz
```

Both folders need `decoder.py`, `mld.py`, and `circuits/{X,Z}_circuit.stim`.
The script checks 2,048 fresh shots per basis, reversed and partitioned batches,
duplicates, empty input, exact bit-index operations, and the zero-coefficient
condition. It saves all intermediate stage log weights, final marginals, and
predictions. Compare the NPZ key sets and every corresponding array using
`numpy.testing.assert_array_equal`; equality tolerances must not be used.

The judge's minimal decoder folder still needs only `decoder.py`, `mld.py`,
`requirements.txt`, and `circuits/{X,Z}_circuit.stim`. The verification tools
and reports are for reproducibility and are never loaded by `decode`.

## Local validation results

On an AMD Ryzen 7 250 (8 cores, 16 hardware threads), both bases and all
200,000 test shots, with import/JIT excluded:

| Implementation | Threads | Runs | Median X+Z runtime | Changed prediction bits |
| --- | ---: | ---: | ---: | ---: |
| Original | 16 | 3 | 88.94 s | 0 |
| Skip zero updates | 16 | 3 | 86.10 s | 0 |
| Skip zero updates | 48 | 1 | 86.13 s | 0 |

The 16-thread median improved by 3.2%. Rechecking the original afterward
took 89.63 s. Every full run reproduced all 800,000 prediction bits and the
entire selected CSV byte for byte. All 21 generated-data arrays, including
every stage's log weights and final marginals, matched the original exactly
with 1 and 16 candidate threads.

These are local measurements, not the organisers' median of five on their
48-vCPU machine. Full timings, package versions, source hashes, and static
checks are recorded in `runtime_validation.json`.
