# MLD improvement study

**The production `submission_template/decoder.py` is unchanged.** This folder
contains optional candidates and a reproducible comparison against that MLD,
not against the historical forward-only notebook implementation.

Implemented/tested: parity and hybrid suffix look-ahead, alternative mechanism
and within-round detector orders, broader/tighter/refined searches, merging
small contributions into existing states, closure probability/absolute-mass
combinations, and maximum-weight versus probability-average MLD ensembles.
No pretrained models from others or banned decoder libraries are used.

## Outcome

On 300,000 independent audit shots:

| Model | LER | Conclusion |
|---|---:|---|
| Existing production MLD | 0.03358667 | Keep as default |
| Hybrid look-ahead + tighter search | 0.03356417 | Difference not statistically clear |
| Max of reference + shuffled-check sweeps | 0.03354750 | Tiny, borderline secondary gain; more compute |

These are **paired results on the same new shots**, not a direct comparison
with a 0.03372 score from another dataset. The checked-in notebook's reported
best is 0.035075 on its own 10k-shot forward-only subset.

Detailed paths, failures, confidence intervals, audit diagnostics and the next
structural research priorities: [EXPERIMENTS.md](EXPERIMENTS.md).
Raw configurations, measurements and source hashes: `results/`.

The small ensemble gain replicated on **another 200,000 fresh shots**. Pooled
over **500,000 independent held-out shots**:

| Model | LER | Net fewer wrong bits vs reference |
|---|---:|---:|
| Existing production MLD | 0.033595 | — |
| Hybrid/tighter search | 0.033560 | 70 (inconclusive) |
| Reference + shuffled-check ensemble | **0.033548** | **94** |

The ensemble's paired 95% interval for the LER difference is
**[-0.00007692, -0.00001708]**: a replicated but very small **0.14% relative**
gain. It is the optional folder's default. The production entry point remains
unchanged because the extra compute may not be worth it after runtime penalties.

Median-of-five on this **16-thread laptop**, 200k shots: ensemble **48.98s**,
hybrid/tighter **31.47s**. Judge-machine runtime is unmeasured. An input-order
optimization preserved all **1.6M checked output bits**; all eight tests pass.
Final aggregate: `results/final_report.json`.

## Try the optional candidate

From the repository root, using the existing environment/requirements:

```powershell
.venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/mld_improvements --problem-set data/train --threads 16 --runs 5 --out mld_ensemble.csv
```

This defaults to the two-sweep-order ensemble in `accuracy.json`. To try the
hybrid/tighter comparison in `candidate.json` instead:

```powershell
$env:MLD_IMPROVEMENTS_CONFIG = (Resolve-Path submission_template/mld_improvements/candidate.json).Path
.venv/Scripts/python.exe submission_template/run_local.py --decoder submission_template/mld_improvements --problem-set data/train --threads 16 --runs 5 --out mld_hybrid.csv
Remove-Item Env:MLD_IMPROVEMENTS_CONFIG
```

Model tables and Numba compile at import; decoding honors the caller's thread
count (including 48 judge threads). There is no GPU implementation or claimed
judging-machine timing. The runtime never reads CSVs or experiment caches;
deduplication is only within each call. This is **repo-integrated**, not a
standalone hand-in: it uses the parent `mld.py`, this folder's `runtime.py`,
the selected JSON, and the existing `hackathon_2026` package.

## Reproduce / extend

```powershell
.venv/Scripts/python.exe -m unittest discover -s submission_template/mld_improvements -v
.venv/Scripts/python.exe submission_template/mld_cross_validation.py --stage development --seed 943100 --shots 3000 --closure --names reference broader_route tight third_pass parity05 parity075 parity10 hybrid05 hybrid075 probability span_probability max065 merge32 merge_all
.venv/Scripts/python.exe submission_template/mld_cross_validation.py --stage development --seed 943110 --shots 8000 --names reference tight hybrid075 merge32 last_detector reverse_checks shuffle_checks parity075_tight hybrid075_tight
.venv/Scripts/python.exe submission_template/mld_cross_validation.py --stage audit --seed 954100 --shots 50000 --config submission_template/mld_improvements/finalists.json
.venv/Scripts/python.exe submission_template/mld_improvements/audit_report.py
.venv/Scripts/python.exe submission_template/mld_improvements/verify_reference.py
.venv/Scripts/python.exe submission_template/mld_improvements/verify_candidate.py
```

Search/report scripts refuse to overwrite named result logs. Existing cached
predictions are gitignored; regenerating a run requires a new result prefix
(new seed or shot count), or archiving **only that run's own** generated results.
For genuinely new research, use new development and untouched audit seeds;
the documented seed families are now opened. The test CSV is never used for
fitting, scoring, selection or these experiments.
