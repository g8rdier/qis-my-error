"""Competition hand-in decoder: LUT head + MLP tail (per basis, torch-free).

Strategy (see RESULTS.md for measurements):
  - Exact detector-pattern hits -> stored LUT majority vote (near-optimal).
  - Misses -> MLP P(O|D) marginal approximator (numpy forward, milliseconds).
  - If MLP weights are absent for a basis, misses fall back to dist-1 LUT vote.
Everything loads at import (free under timing rules); decode() is pure numpy,
deterministic, and has no randomness.
"""
import hashlib
import os

import numpy as np
import stim

from models.common import current_load

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "models", "artifacts")


def _load_npz(name):
    try:
        return np.load(os.path.join(ART, name), allow_pickle=True)
    except FileNotFoundError:
        return None


_BASIS_MAP = _load_npz("basis_map.npz")

_TABLES, _MLPS, _THR = {}, {}, {}
try:
    _CUR = current_load()
except Exception:
    _CUR = {}
for _b in "XZ":
    _t = _load_npz(_b + "_lut.npz")
    if _t is not None:
        _TABLES[_b] = {"keys": _t["keys"], "probs": _t["probs"], "counts": _t["counts"]}
    _wname = _CUR.get(_b, {}).get("weights", _b + "_mlp.npz")  # legacy fallback
    _m = _load_npz(_wname)
    if _m is not None:
        _MLPS[_b] = ({k: _m[k] for k in _m.files if k != "arch_tag"}, str(_m["arch_tag"]))
    _THR[_b] = np.array(_CUR.get(_b, {}).get("threshold", [0.5] * 4), dtype=np.float32)

_N_PACKED = 9


def _detect_basis(circuit):
    if _BASIS_MAP is not None:
        sha = hashlib.sha256(str(circuit).encode()).hexdigest()
        for b in "XZ":
            if _BASIS_MAP[f"sha_{b}"] == sha:
                return b
    # Fallback: X has 2470 mechanisms, Z 2465 (from stored map).
    ncol = int(_BASIS_MAP[f"ncol_X"]) if _BASIS_MAP is not None else 2470
    dem = circuit.detector_error_model()
    n = sum(1 for ins in dem.flatten() if ins.type == "error")
    # NOTE: raw DEM count overcounts merged columns; use unique-column count via
    # cheap pass only when fingerprint missed (unknown circuit variant).
    return "X" if n >= 2470 else "Z"


def _pack(D):
    return np.packbits(np.asarray(D, dtype=np.uint8), axis=1)


def _exact(table, packed):
    q = packed.reshape(len(packed), _N_PACKED).view(f"V{_N_PACKED}").reshape(-1)
    idx = np.searchsorted(table["keys"], q)
    valid = idx < len(table["keys"])
    hit = np.zeros(len(q), dtype=bool)
    hit[valid] = table["keys"][idx[valid]] == q[valid]
    probs = np.full((len(q), 4), np.nan, dtype=np.float32)
    probs[hit] = table["probs"][np.where(hit, idx, 0)[hit]]
    return probs, hit


def _mlp_forward(D, params):
    X = np.asarray(D, dtype=np.float32)
    h = X @ params["W1"] + params["b1"]
    np.maximum(h, 0, out=h)
    if "W3" in params:
        h = h @ params["W2"] + params["b2"]
        np.maximum(h, 0, out=h)
        logits = h @ params["W3"] + params["b3"]
    else:
        logits = h @ params["W2"] + params["b2"]
    return 1.0 / (1.0 + np.exp(-logits))


def _dist1(table, packed, miss):
    sub = packed[miss]
    votes = np.zeros((len(miss), 4), np.float64)
    tot = np.zeros(len(miss), np.float64)
    for j in range(70):
        f = sub.copy()
        f[:, j // 8] ^= np.uint8(1 << (7 - (j % 8)))
        fp, fh = _exact(f, f)[:2]
        if fh.any():
            w = table["counts"][np.searchsorted(table["keys"], f.reshape(len(f), _N_PACKED).view(f"V{_N_PACKED}").reshape(-1))[fh]]
            votes[fh] += fp[fh] * w[:, None].astype(np.float64)
            tot[fh] += w
    return votes, tot


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    D = np.asarray(detector_events, dtype=bool)
    n = len(D)
    out = np.zeros((n, 4), dtype=np.uint8)
    if n == 0:
        return out
    basis = _detect_basis(circuit)
    table = _TABLES.get(basis)
    if table is None:  # no artifacts at all: all-zeros
        return out
    packed = _pack(D)
    probs, hit = _exact(table, packed)
    miss = np.flatnonzero(~hit)
    if len(miss) and basis in _MLPS:
        params, _ = _MLPS[basis]
        probs[miss] = _mlp_forward(D[miss], params).astype(np.float32)
    elif len(miss):
        votes, tot = _dist1(table, packed, miss)
        has = tot > 0
        probs[miss[has]] = (votes[has] / tot[has, None]).astype(np.float32)
    known = ~np.isnan(probs[:, 0])
    out[known] = (probs[known] >= _THR[basis]).astype(np.uint8)
    return out
