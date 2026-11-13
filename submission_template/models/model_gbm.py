"""Model 4 — Gradient boosting on the 70 detector bits (baseline / comparison).

HOW IT WOULD WORK:
  Four binary classifiers (one per observable) or one multilabel GBM over the
  raw 70 bits. Trees split on individual detector bits ("is D_i on?"), so the
  model builds up conjunctions of firing detectors leaf by leaf. Handles
  sparse binary features naturally; no scaling needed. Train one booster set
  per basis.

WHY IT MIGHT WORK HERE — AND ITS LIMITS:
  + Strong out-of-the-box on tabular binary data; robust to imbalance.
  + Captures low-order interactions (pairs/triples of co-firing detectors).
  - Parity/XOR structure needs exponentially many leaves in the worst case;
    an MLP represents the same function far more compactly. Expect GBM to sit
    between logreg and MLP on LER.
  - INFERENCE SPEED RISK: per-shot tree traversal for 100k shots x 4 outputs
    x 2 bases is single-threaded-unfriendly; must batch carefully and time
    with run_local.py. If it approaches t_cap, it loses to the MLP even at
    equal LER. Candidate backends: LightGBM (fast, histogram-based) or
    sklearn HistGradientBoosting (no new heavy dep beyond sklearn).

HYPERPARAMETERS (starting point when implemented):
  LightGBM: num_leaves=63, min_data_in_leaf=200, lr=0.05, rounds~500 with
  early stopping on val LER, feature_fraction=0.8, bagging_fraction=0.8.
  sklearn HGB: max_leaf_nodes=63, min_samples_leaf=200, lr=0.05, max_iter=500.

STATUS: NOT IMPLEMENTED — design doc only. Requires `lightgbm` or
  `scikit-learn` (neither is installed in .venv yet; install with uv).
  Implement in experiments/train_gbm.py when prioritised. The MLP (model_mlp)
  is ranked ahead of this model because parity structure + inference speed
  both favour it; build GBM mainly as an ensemble-diversity member.

EXPECTED ARTIFACTS: artifacts/<BASIS>_gbm.txt (LightGBM) or .pkl (sklearn).
"""
