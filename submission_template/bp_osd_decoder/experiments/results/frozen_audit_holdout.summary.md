# BP+OSD frozen independent holdout

Independent seed folds; paired CIs use per-shot bit-error differences.
Configurations were frozen before opening these seeds; no parameter selection here.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| accuracy_union | 0.034618 | 0.034628 | 0.034608 | -0.000582 | [-0.000719, -0.000444] |
| routed_replace_005 | 0.034752 | 0.034775 | 0.034728 | -0.000448 | [-0.000585, -0.000312] |
| fast_iter5 | 0.035081 | 0.035117 | 0.035045 | -0.000119 | [-0.000259, +0.000020] |
| baseline | 0.035200 | 0.035133 | 0.035267 | +0.000000 | [+0.000000, +0.000000] |
