"""Shared helpers for all decoder models.

DATA HYGIENE (read this before training anything):
  - TRAINING data (labels MAY be used):
      * data/train/train.csv  — 200k labelled shots (100k X + 100k Z), seed 42.
      * data/eval_fresh/...   — self-generated labelled set, different seed (overfit check).
      * data/generated/...    — any other self-generated labelled sets (Stim).
  - TEST data (labels MUST NOT be used, NEVER train on it):
      * data/test/test.csv    — 200k unlabelled shots. Decoded only for submission.
  - OVERFIT CHECK: always report scores on (a) a held-out split of train AND
    (b) a freshly generated set with a different seed. Tuning only against
    data/train/train.csv overfits to that one fixed sample (see AGENTS.md).

Conventions:
  - One model per basis ("X" / "Z"); never mix bases in a single fit.
  - Metric is always per-bit LER = mean(pred != true) over all 4 observables.
  - decode() must be deterministic (seed everything) and fast; heavy work
    (training, table building) happens at import/artifact-load time.
"""
import os
import time

import numpy as np

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.dirname(MODELS_DIR)
REPO_ROOT = os.path.dirname(SUBMISSION_DIR)

TRAIN_CSV = os.path.join(REPO_ROOT, "data", "train", "train.csv")
TEST_CSV = os.path.join(REPO_ROOT, "data", "test", "test.csv")
FRESH_EVAL_DIR = os.path.join(REPO_ROOT, "data", "eval_fresh")

N_DET = 70
N_OBS = 4
BASES = ("X", "Z")

ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "manifest.json")
CURRENT_PATH = os.path.join(ARTIFACTS_DIR, "current.json")


def registry_load():
    """manifest.json: {artifact_filename: {basis, arch, data, seed, ..., val_ler, fresh_ler}}."""
    import json

    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def registry_add(filename, entry):
    """Add one weight artifact + its training config/metrics to the manifest."""
    import json

    ensure_artifacts_dir()
    man = registry_load()
    man[filename] = entry
    with open(MANIFEST_PATH, "w") as f:
        json.dump(man, f, indent=2)
    return man


def current_load():
    """current.json: {basis: {weights: filename, threshold: [t0..t3]}} ({} if missing)."""
    import json

    if not os.path.exists(CURRENT_PATH):
        return {}
    with open(CURRENT_PATH) as f:
        return json.load(f)


def current_save(pointer):
    """Persist the decoder pointer (which weights + thresholds per basis)."""
    import json

    ensure_artifacts_dir()
    with open(CURRENT_PATH, "w") as f:
        json.dump(pointer, f, indent=2)
    return pointer


def load_train(basis=None):
    """Load labelled TRAINING data only. Never point this at data/test."""
    from hackathon_2026 import load_kaggle_csv

    assert TRAIN_CSV.endswith("train.csv"), "refusing to load non-train CSV as training data"
    data = load_kaggle_csv(TRAIN_CSV)
    if basis is None:
        return data
    m = data["basis"] == basis
    return {
        "id": data["id"][m],
        "detector_events": data["detector_events"][m],
        "observable_flips": data["observable_flips"][m],
    }


def train_val_split(D, O, val_fraction=0.1, seed=0):
    """Seeded shuffle split. Returns (D_tr, O_tr, D_val, O_val)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(D))
    n_val = int(round(len(D) * val_fraction))
    va, tr = idx[:n_val], idx[n_val:]
    return D[tr], O[tr], D[va], O[va]


def ler(pred, true):
    """Per-bit logical error rate (== Kaggle MAE on 0/1 predictions)."""
    return float(np.mean(np.asarray(pred).astype(bool) != np.asarray(true).astype(bool)))


def all_zero_ler(O):
    return ler(np.zeros_like(O), O)


def pack_keys(D):
    """Pack (n, 70) bool detector rows into hashable keys (bytes per row)."""
    return np.packbits(np.asarray(D, dtype=np.uint8), axis=1).tobytes()


def unpack_keys(blob, n):
    """Inverse of pack_keys: blob -> (n, 70) bool array."""
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n, -1)
    bits = np.unpackbits(arr, axis=1)[:, :N_DET]
    return bits.astype(bool)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def ensure_artifacts_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    return ARTIFACTS_DIR
