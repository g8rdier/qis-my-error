# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| iter3_5_alpha_equal | 0.035120 | 0.035740 | 0.034500 | -0.000260 | [-0.000471, -0.000050] |
| iter3_5_equal | 0.035130 | 0.035625 | 0.034635 | -0.000250 | [-0.000453, -0.000047] |
| iter3_5_8_equal | 0.035146 | 0.035719 | 0.034573 | -0.000234 | [-0.000503, +0.000034] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `iter3_5_equal` without this seed; left-out LER 0.035109.
- Fold 1: choose `iter3_5_alpha_equal` without this seed; left-out LER 0.036016.
- Fold 2: choose `iter3_5_alpha_equal` without this seed; left-out LER 0.034359.
