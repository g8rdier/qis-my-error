# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| spa10 | 0.034708 | 0.035219 | 0.034198 | -0.000672 | [-0.001060, -0.000284] |
| iter5_pairs300 | 0.034729 | 0.035073 | 0.034385 | -0.000651 | [-0.001003, -0.000299] |
| triple_iter5 | 0.034745 | 0.034948 | 0.034542 | -0.000635 | [-0.000994, -0.000277] |
| triple_bp60 | 0.034875 | 0.035271 | 0.034479 | -0.000505 | [-0.000784, -0.000226] |
| spa5 | 0.034948 | 0.035490 | 0.034406 | -0.000432 | [-0.000786, -0.000079] |
| triple_bp40 | 0.035031 | 0.035490 | 0.034573 | -0.000349 | [-0.000605, -0.000093] |
| iter5_refine_wider | 0.035036 | 0.035323 | 0.034750 | -0.000344 | [-0.000681, -0.000007] |
| spa3 | 0.035083 | 0.035885 | 0.034281 | -0.000297 | [-0.000648, +0.000054] |
| triple_bp20 | 0.035172 | 0.035615 | 0.034729 | -0.000208 | [-0.000423, +0.000006] |
| triple_single40 | 0.035182 | 0.035635 | 0.034729 | -0.000198 | [-0.000357, -0.000039] |
| triple_refine60 | 0.035187 | 0.035573 | 0.034802 | -0.000193 | [-0.000410, +0.000025] |
| baseline | 0.035380 | 0.035854 | 0.034906 | +0.000000 | [+0.000000, +0.000000] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `iter5_pairs300` without this seed; left-out LER 0.034750.
- Fold 1: choose `iter5_pairs300` without this seed; left-out LER 0.035719.
- Fold 2: choose `spa10` without this seed; left-out LER 0.034281.
