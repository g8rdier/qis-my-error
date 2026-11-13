"""Decoder: lookup table (LUT) for frequent syndromes, BP + OSD (combination sweep, weighted vote) for the rest.

- lut_X.npz / lut_Z.npz are built by build_lut.py (run it from the repository root). For each syndrome seen during
  sampling they hold the per-bit majority of the observable flips and how often the syndrome was seen. Only syndromes
  seen at least LUT_MIN_COUNT times are answered from the table, because rarer entries are noisy.
- Everything else goes to bposd.py: a few iterations of min-sum BP, then OSD ordered by the BP posteriors (averaged
  over iterations), with candidate solutions weighted by their prior probability to vote on each observable bit.
"""
import hashlib
import os
import sys

import numba
import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bposd import Graph, decode_bposd  # noqa: E402
from hackathon_2026 import problem_info  # noqa: E402

USE_LUT = True
LUT_MIN_COUNT = 5
BPOSD = dict(max_iter=3, scaling=0.6, lam1=1000, lam2=150, osd_always=True, avg_post=True)

_TABLES = {}
_GRAPHS = {}


def _sha(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _table(circuit: stim.Circuit):
    """The LUT built from this exact circuit (matched by a hash of its text)."""
    if not _TABLES:
        for b in "XZ":
            with np.load(os.path.join(HERE, f"lut_{b}.npz")) as f:
                _TABLES[str(f["circuit_sha"])] = (f["keys_lo"], f["offsets"], f["pred"], f["count"])
    sha = _sha(circuit)
    if sha not in _TABLES:
        raise ValueError("no lookup table was built for this circuit")
    return _TABLES[sha]


def _graph(circuit: stim.Circuit) -> Graph:
    sha = _sha(circuit)
    if sha not in _GRAPHS:
        _GRAPHS[sha] = Graph(**problem_info(circuit))
    return _GRAPHS[sha]


@numba.njit(cache=True, parallel=True)
def _lookup(lo, hi, keys_lo, offsets, pred, count, out, cnt):
    for i in numba.prange(len(lo)):
        a, b = offsets[hi[i]], offsets[hi[i] + 1]
        key = lo[i]
        while a < b:  # binary search in this hi-bucket
            m = (a + b) >> 1
            if keys_lo[m] < key:
                a = m + 1
            else:
                b = m
        if a < offsets[hi[i] + 1] and keys_lo[a] == key:
            cnt[i] = count[a]
            p = pred[a]
            for j in range(4):
                out[i, j] = (p >> j) & 1


def lookup(circuit, detector_events):
    """Returns (predictions (N, 4) uint8, count (N,) uint32: how often the syndrome was seen, 0 if not in the table)."""
    packed = np.packbits(np.asarray(detector_events, dtype=bool), axis=1, bitorder="little")  # (N, 9)
    lo = np.ascontiguousarray(packed[:, :8]).view(np.uint64).ravel()
    hi = packed[:, 8].astype(np.int64)
    out = np.zeros((len(lo), 4), np.uint8)
    cnt = np.zeros(len(lo), np.uint32)
    _lookup(lo, hi, *_table(circuit), out, cnt)
    return out, cnt


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    D = np.asarray(detector_events, dtype=bool)
    if USE_LUT:
        out, cnt = lookup(circuit, D)
        rest = np.flatnonzero(cnt < LUT_MIN_COUNT)
    else:
        out = np.zeros((len(D), 4), np.uint8)
        rest = np.arange(len(D))
    if len(rest):
        out[rest], _ = decode_bposd(_graph(circuit), D[rest], **BPOSD)
    return out
