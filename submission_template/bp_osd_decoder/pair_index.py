"""Precomputed meet-in-the-middle index of ALL two-mechanism effects.

~3M pairs per circuit. An open-address table indexes detector syndromes;
equal-syndrome pairs are chained. Keys are verified with both full words,
so a hash collision is never treated as a detector match. No sample labels.
"""
from functools import lru_cache
import hashlib

from numba import njit
import numpy as np


@njit(cache=True)
def _slot(lo, hi, mask):
    x = lo ^ (hi * np.uint64(0x9E3779B97F4A7C15))
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return np.int64(x & np.uint64(mask))


@njit(cache=True)
def find(lo, hi, cols, heads, ii, jj):
    mask = len(heads) - 1
    slot = _slot(lo, hi, mask)
    while heads[slot] >= 0:
        k = heads[slot]
        a, b = ii[k], jj[k]
        if (cols[a, 0] ^ cols[b, 0]) == lo and (cols[a, 1] ^ cols[b, 1]) == hi:
            return k
        slot = (slot + 1) & mask
    return -1


@njit(cache=True)
def build(cols):
    n = len(cols)
    count = n * (n - 1) // 2
    size = 1
    while size < 2 * count:
        size *= 2
    heads = np.full(size, -1, np.int32)
    nxt = np.full(count, -1, np.int32)
    ii = np.empty(count, np.int32)
    jj = np.empty(count, np.int32)
    mask, k = size - 1, 0
    for a in range(n):
        for b in range(a + 1, n):
            lo, hi = cols[a, 0] ^ cols[b, 0], cols[a, 1] ^ cols[b, 1]
            slot = _slot(lo, hi, mask)
            while heads[slot] >= 0:
                prev = heads[slot]
                i, j = ii[prev], jj[prev]
                if ((cols[i, 0] ^ cols[j, 0]) == lo
                        and (cols[i, 1] ^ cols[j, 1]) == hi):
                    break
                slot = (slot + 1) & mask
            ii[k], jj[k] = a, b
            nxt[k] = heads[slot]
            heads[slot] = k
            k += 1
    return heads, nxt, ii, jj


_INDICES = {}


def get_index(cols):
    sha = hashlib.sha256(cols.tobytes()).hexdigest()
    if sha not in _INDICES:
        _INDICES[sha] = build(cols)
    return _INDICES[sha]
