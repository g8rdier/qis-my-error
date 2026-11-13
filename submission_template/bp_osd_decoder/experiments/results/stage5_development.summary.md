# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| accuracy_union | 0.034417 | 0.034823 | 0.034010 | -0.000964 | [-0.001302, -0.000625] |
| routed_005 | 0.034573 | 0.035042 | 0.034104 | -0.000807 | [-0.001134, -0.000481] |
| routed_replace_005 | 0.034573 | 0.035042 | 0.034104 | -0.000807 | [-0.001134, -0.000481] |
| routed_replace_010 | 0.034625 | 0.035156 | 0.034094 | -0.000755 | [-0.001079, -0.000432] |
| routed_replace_020 | 0.034708 | 0.035198 | 0.034219 | -0.000672 | [-0.000997, -0.000347] |
| routed_020 | 0.034719 | 0.035198 | 0.034240 | -0.000661 | [-0.000987, -0.000336] |
| routed_replace_030 | 0.034797 | 0.035187 | 0.034406 | -0.000583 | [-0.000903, -0.000264] |
| fast_iter5 | 0.034917 | 0.035323 | 0.034510 | -0.000464 | [-0.000796, -0.000131] |
| baseline | 0.035380 | 0.035854 | 0.034906 | +0.000000 | [+0.000000, +0.000000] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `accuracy_union` without this seed; left-out LER 0.034266.
- Fold 1: choose `accuracy_union` without this seed; left-out LER 0.035359.
- Fold 2: choose `accuracy_union` without this seed; left-out LER 0.033625.
