"""Model 2 — Logistic regression baseline (linear, per-observable).

HOW IT WORKS:
  Four independent binary logistic regressions on the raw 70 detector bits:
    P(O_j = 1 | D) = sigmoid(w_j . D + b_j),  j = 0..3.
  Trained with minibatch SGD on binary cross-entropy + L2 (numpy-only, so it
  runs in this env with zero new dependencies). Threshold at 0.5 at inference
  (optimal for per-bit error). One weight matrix (70x4) + bias per basis.

WHY THIS MODEL:
  This is the Bayes-marginal decoder restricted to a LINEAR function of D.
  It captures first-order correlations (which detectors predict which
  observable) but cannot express parity/XOR structure: a logical flip often
  depends on the PARITY of a detector subset, which is invisible to a linear
  model. So expect: much faster than everything, better than all-zeros, but
  clearly below an MLP. Its value is (a) a calibrated floor for "what linear
  buys", (b) a 280-parameter fallback that always runs, (c) features for
  stacking (its 4 output probabilities can feed the MLP/ensemble).

TRAINING: full-batch or minibatch SGD, BCE + L2. Deterministic given seed.
INFERENCE COST: one (n,70)x(70,4) matmul + sigmoid — microseconds per 100k
  shots on CPU. Artifact: tiny .npz (W, b). No torch/sklearn needed.

STATUS: implemented numpy-only, runnable today. See experiments/train_logreg.py.
"""
import os

import numpy as np

from .common import ARTIFACTS_DIR, N_OBS, ensure_artifacts_dir

SUFFIX = "_logreg.npz"


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def train(D, O, seed=0, epochs=30, batch=4096, lr=0.5, l2=1e-4, verbose=True):
    """Minibatch SGD on BCE+L2. D,O: bool arrays. Returns (W (70,4), b (4,))."""
    rng = np.random.default_rng(seed)
    X = np.asarray(D, dtype=np.float64)
    Y = np.asarray(O, dtype=np.float64)
    n, d = X.shape
    W = np.zeros((d, N_OBS))
    b = np.zeros(N_OBS)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb, yb = X[idx], Y[idx]
            p = _sigmoid(xb @ W + b)
            g = (p - yb) / len(idx)
            W -= lr * (xb.T @ g + l2 * W)
            b -= lr * g.sum(axis=0)
        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1):
            p = _sigmoid(X @ W + b)
            eps = 1e-12
            bce = -(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)).mean()
            print(f"  epoch {ep + 1}/{epochs} train BCE {bce:.4f}", flush=True)
    return W, b


def predict_proba(D, W, b):
    return _sigmoid(np.asarray(D, dtype=np.float64) @ W + b).astype(np.float32)


def predict(D, W, b):
    return (predict_proba(D, W, b) >= 0.5).astype(np.uint8)


def save(W, b, basis):
    ensure_artifacts_dir()
    path = os.path.join(ARTIFACTS_DIR, basis + SUFFIX)
    np.savez_compressed(path, W=W, b=b)
    return path


def load(basis):
    z = np.load(os.path.join(ARTIFACTS_DIR, basis + SUFFIX))
    return z["W"], z["b"]
