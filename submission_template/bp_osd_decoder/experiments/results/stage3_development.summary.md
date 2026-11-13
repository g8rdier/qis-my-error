# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| union358_pairs200 | 0.034417 | 0.034823 | 0.034010 | -0.000964 | [-0.001302, -0.000625] |
| triple5_bp100 | 0.034510 | 0.034760 | 0.034260 | -0.000870 | [-0.001247, -0.000493] |
| union35_pairs300 | 0.034568 | 0.035073 | 0.034063 | -0.000812 | [-0.001121, -0.000504] |
| triple5_pairs200 | 0.034646 | 0.034844 | 0.034448 | -0.000734 | [-0.001096, -0.000373] |
| union35_pairs200 | 0.034693 | 0.035271 | 0.034115 | -0.000688 | [-0.000986, -0.000389] |
| triple5_bp60 | 0.034708 | 0.034865 | 0.034552 | -0.000672 | [-0.001034, -0.000310] |
| union358 | 0.034969 | 0.035302 | 0.034635 | -0.000411 | [-0.000746, -0.000076] |
| union_diverse | 0.035099 | 0.035667 | 0.034531 | -0.000281 | [-0.000601, +0.000039] |
| union35 | 0.035240 | 0.035781 | 0.034698 | -0.000141 | [-0.000444, +0.000163] |
| union35_cap2048 | 0.035240 | 0.035781 | 0.034698 | -0.000141 | [-0.000444, +0.000163] |
| union_single5 | 0.035328 | 0.035844 | 0.034812 | -0.000052 | [-0.000410, +0.000305] |
| baseline | 0.035380 | 0.035854 | 0.034906 | +0.000000 | [+0.000000, +0.000000] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `union358_pairs200` without this seed; left-out LER 0.034266.
- Fold 1: choose `union358_pairs200` without this seed; left-out LER 0.035359.
- Fold 2: choose `triple5_bp100` without this seed; left-out LER 0.034031.
