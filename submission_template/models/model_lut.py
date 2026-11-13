"""Model 1 — Lookup-table (LUT) decoder with fallbacks.

HOW IT WORKS:
  Training: group labelled shots by exact detector pattern D, store the
  majority-vote observable vector O per pattern (the empirical Bayes decision
  for that pattern). Optionally also store P(O_j=1 | D) per pattern.
  Decoding: pack each 70-bit syndrome into 9 bytes, binary-search a sorted
  key array (np.searchsorted on void dtype — C speed, no Python loop).
  Hits predict the stored majority vote; misses use a fallback.

WHY IT FITS / DOESN'T FIT THIS DATASET:
  Measured on data/train (100k shots/basis): ~79k DISTINCT patterns per basis,
  mean syndrome weight ~7.4, top non-zero pattern occurs only ~127 times.
  Patterns covering 90% of mass: ~69k. So a LUT memorises the head (esp. D=0,
  ~4.5% of shots) but ~75% of test shots MISS the table — the fallback
  decides the endgame LER, not the table. Verdict: useful as a fast-path /
  ensemble member, not as a standalone decoder.

FALLBACKS (in order of sophistication):
  0. all-zeros (baseline; LER ~0.29).
  1. distance-1 vote: all 70 single-bit flips of D, looked up with 70
     vectorised searchsorted calls; majority vote over hits weighted by
     training counts. ~0.5-1.5 s per 100k shots.
  2. delegate misses to model_logreg / model_mlp (hybrid LUT+ML).

INFERENCE COST: exact path = 1 searchsorted over ~70k keys per 100k shots
  (~10 ms). dist1 adds 70 searchsorted calls on the miss subset.
  Deterministic. Import-time: load table from .npz (free under timing rules).

STATUS: implemented numpy-only, runnable today. See experiments/train_lut.py.
"""
import os

import numpy as np

from .common import ARTIFACTS_DIR, N_OBS, ensure_artifacts_dir, ler

TABLE_SUFFIX = "_lut.npz"
N_PACKED = 9  # ceil(70/8) bytes per syndrome


def pack_matrix(D):
    """(n,70) bool -> (n,9) uint8 packed rows (np.packbits, big-endian)."""
    return np.packbits(np.asarray(D, dtype=np.uint8), axis=1)


def _flip_col(packed, j):
    """Copy of packed rows with detector bit j toggled. Vectorised."""
    out = packed.copy()
    out[:, j // 8] ^= np.uint8(1 << (7 - (j % 8)))
    return out


def fit_table(D, O):
    """Majority-vote O per distinct D pattern. Returns dict key -> (prob4, count)."""
    keys = [r.tobytes() for r in pack_matrix(D)]
    table = {}
    for k, o in zip(keys, np.asarray(O, dtype=np.uint8)):
        if k in table:
            table[k][0] += o
            table[k][1] += 1
        else:
            table[k] = [o.astype(np.int64).copy(), 1]
    return {k: ((s / c).astype(np.float32), c) for k, (s, c) in table.items()}


def save_table(table, basis, out_dir=None):
    """Store SORTED void-key arrays for vectorised searchsorted lookup."""
    d = out_dir or ARTIFACTS_DIR
    os.makedirs(d, exist_ok=True)
    raw = np.array([np.frombuffer(k, dtype=np.uint8) for k in table], dtype=np.uint8)
    keys = raw.reshape(len(raw), N_PACKED).view(f"V{N_PACKED}").reshape(-1)
    order = np.argsort(keys)
    path = os.path.join(d, basis + TABLE_SUFFIX)
    np.savez_compressed(
        path,
        keys=keys[order],
        probs=np.array([v[0] for v in table.values()], dtype=np.float32)[order],
        counts=np.array([v[1] for v in table.values()], dtype=np.int64)[order],
    )
    return path


def load_table(basis, in_dir=None):
    z = np.load(os.path.join(in_dir or ARTIFACTS_DIR, basis + TABLE_SUFFIX))
    return {"keys": z["keys"], "probs": z["probs"], "counts": z["counts"]}


def _exact_lookup(table, packed):
    """Vectorised exact lookup. Returns (probs (n,4) NaN on miss, hitmask, idx)."""
    q = packed.reshape(len(packed), N_PACKED).view(f"V{N_PACKED}").reshape(-1)
    idx = np.searchsorted(table["keys"], q)
    n_keys = len(table["keys"])
    valid = idx < n_keys
    hit = np.zeros(len(q), dtype=bool)
    hit[valid] = table["keys"][idx[valid]] == q[valid]
    safe = np.where(hit, idx, 0)
    probs = np.full((len(q), N_OBS), np.nan, dtype=np.float32)
    probs[hit] = table["probs"][safe[hit]]
    return probs, hit, safe


def predict(table, D, fallback="zeros", fallback_model=None):
    """Predict (n,4) uint8. fallback: 'zeros' | 'dist1' | 'model'."""
    packed = pack_matrix(np.asarray(D, dtype=bool))
    probs, hit, _ = _exact_lookup(table, packed)
    if fallback == "dist1":
        miss = np.flatnonzero(~hit)
        if len(miss):
            sub = packed[miss]
            votes = np.zeros((len(miss), N_OBS), np.float64)
            tot = np.zeros(len(miss), np.float64)
            for j in range(70):
                fp, fh, fidx = _exact_lookup(table, _flip_col(sub, j))
                if fh.any():
                    w = table["counts"][fidx[fh]].astype(np.float64)
                    votes[fh] += fp[fh] * w[:, None]
                    tot[fh] += w
            has = tot > 0
            probs[miss[has]] = (votes[has] / tot[has, None]).astype(np.float32)
    elif fallback == "model" and fallback_model is not None:
        miss = np.flatnonzero(~hit)
        if len(miss):
            out = np.zeros((len(packed), N_OBS), dtype=np.uint8)
            known = ~np.isnan(probs[:, 0])
            out[known] = (probs[known] >= 0.5).astype(np.uint8)
            out[miss] = fallback_model(np.asarray(D, dtype=bool)[miss])
            return out
    out = np.zeros((len(packed), N_OBS), dtype=np.uint8)
    known = ~np.isnan(probs[:, 0])
    out[known] = (probs[known] >= 0.5).astype(np.uint8)
    return out


def hit_rate(table, D):
    _, hit, _ = _exact_lookup(table, pack_matrix(np.asarray(D, dtype=bool)))
    return float(hit.mean()) if len(hit) else 0.0


def evaluate(table, D, O, fallback="zeros", fallback_model=None):
    p = predict(table, D, fallback=fallback, fallback_model=fallback_model)
    return {"ler": ler(p, O), "hit_rate": hit_rate(table, D), "n": len(D)}
