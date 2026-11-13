# BP+OSD frozen independent holdout

Independent seed folds; paired CIs use per-shot bit-error differences.
Configurations were frozen before opening these seeds; no parameter selection here.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| frozen_pair_closure | 0.034146 | 0.034650 | 0.033642 | -0.000918 | [-0.001074, -0.000761] |
| previous_accuracy_union | 0.034323 | 0.034822 | 0.033823 | -0.000741 | [-0.000884, -0.000598] |
| baseline | 0.035063 | 0.035558 | 0.034568 | +0.000000 | [+0.000000, +0.000000] |
