# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| exact3_iter5_pairs200 | 0.034812 | 0.035167 | 0.034458 | -0.000568 | [-0.000909, -0.000227] |
| exact3_union358 | 0.034911 | 0.035219 | 0.034604 | -0.000469 | [-0.000808, -0.000129] |
| exact3_union35 | 0.035161 | 0.035729 | 0.034594 | -0.000219 | [-0.000527, +0.000089] |
| exact3_iter5 | 0.035224 | 0.035708 | 0.034740 | -0.000156 | [-0.000515, +0.000202] |
| baseline | 0.035380 | 0.035854 | 0.034906 | +0.000000 | [+0.000000, +0.000000] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `exact3_iter5_pairs200` without this seed; left-out LER 0.034672.
- Fold 1: choose `exact3_iter5_pairs200` without this seed; left-out LER 0.035984.
- Fold 2: choose `exact3_union358` without this seed; left-out LER 0.034187.
