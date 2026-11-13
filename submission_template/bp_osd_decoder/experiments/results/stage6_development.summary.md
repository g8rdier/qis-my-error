# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| jitter4_sigma10 | 0.034292 | 0.034667 | 0.033917 | -0.001089 | [-0.001472, -0.000705] |
| jitter12_sigma10 | 0.034323 | 0.034635 | 0.034010 | -0.001057 | [-0.001451, -0.000663] |
| wide_union500 | 0.034339 | 0.034792 | 0.033885 | -0.001042 | [-0.001416, -0.000667] |
| wide_union300 | 0.034365 | 0.034792 | 0.033938 | -0.001016 | [-0.001387, -0.000645] |
| diverse_union300 | 0.034375 | 0.034740 | 0.034010 | -0.001005 | [-0.001385, -0.000625] |
| accuracy_union_control | 0.034417 | 0.034823 | 0.034010 | -0.000964 | [-0.001302, -0.000625] |
| union_capacity2048 | 0.034422 | 0.034823 | 0.034021 | -0.000958 | [-0.001297, -0.000620] |
| union_capacity8192_gap24 | 0.034422 | 0.034823 | 0.034021 | -0.000958 | [-0.001297, -0.000620] |
| jitter4_sigma20 | 0.034458 | 0.034865 | 0.034052 | -0.000922 | [-0.001281, -0.000563] |
| jitter4_sigma05 | 0.034484 | 0.034823 | 0.034146 | -0.000896 | [-0.001282, -0.000510] |
| baseline | 0.035380 | 0.035854 | 0.034906 | +0.000000 | [+0.000000, +0.000000] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `jitter12_sigma10` without this seed; left-out LER 0.034156.
- Fold 1: choose `wide_union300` without this seed; left-out LER 0.035312.
- Fold 2: choose `jitter4_sigma10` without this seed; left-out LER 0.033891.
