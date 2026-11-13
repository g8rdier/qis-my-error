# BP+OSD development ablations

Independent seed folds; paired CIs use per-shot bit-error differences.
CIs do not correct for model selection. These are NOT final holdout scores.
Setup/JIT excluded; use run_local.py for official-shaped timing.

| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| route_union358_pairs200_0.05 | 0.034573 | 0.035042 | 0.034104 | -0.000807 | [-0.001134, -0.000481] |
| route_union358_pairs200_0.1 | 0.034641 | 0.035177 | 0.034104 | -0.000740 | [-0.001062, -0.000417] |
| route_triple5_bp100_0.05 | 0.034641 | 0.034875 | 0.034406 | -0.000740 | [-0.001108, -0.000371] |
| route_triple5_bp100_0.1 | 0.034688 | 0.034958 | 0.034417 | -0.000693 | [-0.001057, -0.000328] |
| route_union358_pairs200_0.2 | 0.034719 | 0.035198 | 0.034240 | -0.000661 | [-0.000987, -0.000336] |
| route_triple5_bp60_0.05 | 0.034771 | 0.034906 | 0.034635 | -0.000609 | [-0.000967, -0.000252] |
| route_union358_pairs200_0.3 | 0.034802 | 0.035208 | 0.034396 | -0.000578 | [-0.000898, -0.000258] |
| route_triple5_bp60_0.1 | 0.034807 | 0.034969 | 0.034646 | -0.000573 | [-0.000927, -0.000219] |
| route_triple5_bp100_0.2 | 0.034807 | 0.035021 | 0.034594 | -0.000573 | [-0.000939, -0.000207] |
| route_union358_pairs200_0.4 | 0.034833 | 0.035240 | 0.034427 | -0.000547 | [-0.000868, -0.000226] |
| route_triple5_bp100_0.4 | 0.034891 | 0.035125 | 0.034656 | -0.000490 | [-0.000841, -0.000138] |
| route_triple5_bp60_0.2 | 0.034896 | 0.035052 | 0.034740 | -0.000484 | [-0.000837, -0.000132] |
| route_triple5_bp60_0.4 | 0.034927 | 0.035125 | 0.034729 | -0.000453 | [-0.000798, -0.000108] |
| route_triple5_bp100_0.3 | 0.034927 | 0.035156 | 0.034698 | -0.000453 | [-0.000816, -0.000090] |
| route_triple5_bp60_0.3 | 0.034974 | 0.035177 | 0.034771 | -0.000406 | [-0.000762, -0.000051] |

## Leave-one-seed-family-out selection check

- Fold 0: choose `route_triple5_bp100_0.05` without this seed; left-out LER 0.034469.
- Fold 1: choose `route_union358_pairs200_0.05` without this seed; left-out LER 0.035609.
- Fold 2: choose `route_triple5_bp100_0.05` without this seed; left-out LER 0.034141.
