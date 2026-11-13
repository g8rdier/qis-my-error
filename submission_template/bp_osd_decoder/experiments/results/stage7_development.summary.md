# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| closure_class2 | 0.034224 | 0.034740 | 0.033708 | -0.001156 | [-0.001530, -0.000782] |
| closure_class4 | 0.034240 | 0.034813 | 0.033667 | -0.001141 | [-0.001519, -0.000762] |
| closure_global16 | 0.034250 | 0.034781 | 0.033719 | -0.001130 | [-0.001497, -0.000764] |
| closure_global64 | 0.034250 | 0.034781 | 0.033719 | -0.001130 | [-0.001510, -0.000750] |
| closure_global32_two | 0.034292 | 0.034854 | 0.033729 | -0.001089 | [-0.001474, -0.000704] |
| jitter_closure_class4 | 0.034333 | 0.034792 | 0.033875 | -0.001047 | [-0.001440, -0.000654] |
| closure_class8_two | 0.034354 | 0.034885 | 0.033823 | -0.001026 | [-0.001425, -0.000627] |
| closure_control | 0.034422 | 0.034823 | 0.034021 | -0.000958 | [-0.001297, -0.000620] |
| baseline | 0.035380 | 0.035854 | 0.034906 | +0.000000 | [+0.000000, +0.000000] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `closure_class2` without this seed; left-out LER 0.034172.
- Fold 1: choose `closure_global16` without this seed; left-out LER 0.034859.
- Fold 2: choose `closure_class4` without this seed; left-out LER 0.033859.
