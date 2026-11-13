"""Model 3 — Small MLP: 70 -> hidden -> 4 (the Bayes-marginal approximator).

HOW IT WORKS:
  Multilabel binary classifier trained on per-bit BCE against the TRUE flips:
    input:  70 raw detector bits (float).
    hidden: 1-2 fully connected layers, ReLU, dropout, e.g. 70->256->128->4.
    output: 4 logits, one per observable; P(O_j=1|D) = sigmoid(logit_j).
  With enough data this DIRECTLY approximates the Bayes-optimal marginal
  P(O_j | D) — the theoretical optimum for per-bit LER under degeneracy
  (averaging over all error mechanisms consistent with D, not just the single
  most likely one). Threshold at 0.5. One network per basis.

WHY THIS MODEL:
  The map D -> O is dominated by parity/XOR structure (a logical flip fires
  when an ODD number of certain mechanisms fire). Linear models can't see
  parity; an MLP with one hidden layer can. GPU training box available
  (RTX 3050 Ti), so train in PyTorch with CUDA, then EXPORT weights to .npz
  and run inference in pure numpy (2 matmuls + ReLU + sigmoid) — no torch
  needed inside decode(), keeping the timed path to milliseconds on CPU.
  Batched inference on 100k x 70 is a ~100k x 256 matmul: trivially < t_cap.

ARCHITECTURE / HYPERPARAMETERS (starting point, tune and log in RESULTS.md):
  70 -> 256 (ReLU, dropout 0.1) -> 128 (ReLU, dropout 0.1) -> 4 logits.
  Adam(lr=1e-3), batch 4096-16384, BCEWithLogitsLoss, early-stop on val LER.
  Class imbalance: per-observable P(flip) ~0.24-0.34 — mild; plain BCE is
  fine, optionally pos_weight. Train on 1M+ generated shots when possible
  (Stim sampling is cheap; GPU trains in minutes).

FEATURE OPTIONS (ablate one at a time):
  v0: raw 70 bits only (start here).
  v1: + syndrome weight (1 feat: sum(D)/70) — helps calibrate.
  v2: + per-round parities (5 feats: XOR of D within each round block).
  v3: + logreg probabilities (4 feats from model_logreg — stacking).

STATUS: inference path (numpy_forward) implemented and runnable today.
  Training needs torch: `uv pip install torch --index-url ...cu124` (GPU box)
  or CPU torch elsewhere. See experiments/train_mlp.py.

DEPLOYMENT: artifacts/<BASIS>_mlp.npz holds Ws, bs, architecture tag.
  decode() path uses only numpy_forward — deterministic, no randomness.
"""
import os

import numpy as np

from .common import ARTIFACTS_DIR, ensure_artifacts_dir, ler

SUFFIX = "_mlp.npz"


def numpy_forward(D, params):
    """Pure-numpy inference. params: dict with W1,b1,W2,b2(,W3,b3)."""
    X = np.asarray(D, dtype=np.float32)
    h = X @ params["W1"] + params["b1"]
    np.maximum(h, 0, out=h)  # ReLU
    if "W3" in params:  # 2-hidden-layer net
        h = h @ params["W2"] + params["b2"]
        np.maximum(h, 0, out=h)
        logits = h @ params["W3"] + params["b3"]
    else:  # 1-hidden-layer net
        logits = h @ params["W2"] + params["b2"]
    return 1.0 / (1.0 + np.exp(-logits))


def predict(D, params, threshold=0.5):
    return (numpy_forward(D, params) >= threshold).astype(np.uint8)


def save(params, basis, arch_tag="mlp256x128", filename=None):
    ensure_artifacts_dir()
    name = filename or (basis + SUFFIX)
    path = os.path.join(ARTIFACTS_DIR, name)
    np.savez_compressed(path, arch_tag=np.array(arch_tag), **params)
    return path, name


def load(basis):
    z = np.load(os.path.join(ARTIFACTS_DIR, basis + SUFFIX), allow_pickle=True)
    params = {k: z[k] for k in z.files if k != "arch_tag"}
    return params, str(z["arch_tag"])


# ---- torch training core (shared by experiments/train_mlp.py and sweep_mlp.py).
# torch is imported lazily so this module stays importable (and decode-safe)
# on machines without torch.

def build_torch_model(hidden=(256, 128), dropout=0.1):
    import torch.nn as nn

    layers, d = [], 70
    for h in hidden:
        layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
        d = int(h)
    layers += [nn.Linear(d, 4)]
    return nn.Sequential(*layers)


def torch_state_to_numpy(state_dict):
    """Linear-layer state dict -> {W1,b1,W2,b2,...} row-major for numpy_forward.
    Robust to key nesting ('0.weight' from a bare Sequential or
    'net.0.weight' from a wrapped module)."""
    sd = {k: v.numpy() for k, v in state_dict.items()}
    items = []
    for k, v in sd.items():
        parts = k.split(".")
        idx = next(int(p) for p in parts if p.isdigit())
        items.append((idx, parts[-1], v))
    items.sort(key=lambda t: (t[0], t[1] != "weight"))  # weight before bias
    params, i = {}, 0
    for j in range(0, len(items), 2):
        assert items[j][1] == "weight" and items[j + 1][1] == "bias", items[j][:2]
        i += 1
        params[f"W{i}"] = items[j][2].T.astype(np.float32)
        params[f"b{i}"] = items[j + 1][2].astype(np.float32)
    return params


def fit_torch(model, Dtr, Otr, Dva, Ova, device, lr=1e-3, batch=8192, epochs=40,
              patience=8, seed=0, tag="", log_every=10):
    """Minibatch BCE training with early stopping on val LER. Returns
    (best_state_cpu_dict, best_val_ler, epochs_run, seconds). Deterministic
    given seed (model init must be seeded by caller)."""
    import time

    import torch
    import torch.nn as nn

    t_start = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(np.asarray(Dtr, dtype=np.float32))  # CPU; batched up
    Ytr = torch.from_numpy(np.asarray(Otr, dtype=np.float32))
    Xva = torch.from_numpy(np.asarray(Dva, dtype=np.float32)).to(device)
    Yva = torch.from_numpy(np.asarray(Ova, dtype=np.float32)).to(device)
    n = len(Xtr)

    def val_ler():
        model.eval()
        with torch.no_grad():
            pv = (torch.sigmoid(model(Xva)) >= 0.5).cpu().numpy()
        return ler(pv, Ova)

    best, best_state, bad = 1.0, None, 0
    ep_run = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = Ytr[idx].to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            nb += len(idx)
        ep_run = ep + 1
        vl = val_ler()
        if ep == 0 or ep_run % log_every == 0 or ep_run == epochs:
            print(f"{tag} ep {ep_run}/{epochs} train BCE {tot / nb:.4f} val LER {vl:.5f}", flush=True)
        if vl < best - 1e-5:
            best = vl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"{tag}: early stop at ep {ep_run}, best val {best:.5f}", flush=True)
                break
    return best_state, best, ep_run, time.perf_counter() - t_start


def torch_spec():
    """Canonical training spec for experiments/train_mlp.py (requires torch)."""
    return {
        "arch": [70, 1024, 512, 4],
        "dropout": 0.1,
        "lr": 1e-3,
        "batch": 32768,
        "epochs": 80,
        "patience": 12,
        "optimizer": "adam",
        "loss": "bce_with_logits",
        "early_stop_metric": "val_ler",
        "threshold": 0.5,
        "data": "train100k+gen_big500k+gen_big2_2000k per basis (2.6M), val = seed-4 10% split",
    }
