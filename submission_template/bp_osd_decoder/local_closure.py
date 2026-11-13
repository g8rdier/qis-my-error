"""Syndrome-preserving physical-pair replacement around good candidates.

If H_i XOR H_j == H_k XOR H_l, toggling those four mechanisms preserves D.
Unlike order-three search around an information set, this can expand a
physical weight-four/five/six candidate directly into other such candidates.
Every new vector is scored with the original priors and deduplicated.
"""
from numba import njit
import numpy as np
from bp_osd_decoder.pair_index import find
from bp_osd_decoder.pool import insert
from bp_osd_decoder.enhanced import _ctz


@njit(cache=True)
def vector_hash(vector):
    h = np.uint64(0xCBF29CE484222325)
    for x in vector:
        h = (h ^ x) * np.uint64(0x100000001B3)
    return h


@njit(cache=True)
def expand(cols, logical, prior, heads, nxt, ii, jj,
           vectors, costs, logs, hashes, used, gap, index,
           seed_count=32, rounds=1, per_class=False):
    drops = 0
    for _ in range(rounds):
        order = np.argsort(costs[:used])
        count = 0
        maximum = min(used, seed_count * (16 if per_class else 1))
        seeds = np.empty(maximum, np.int64)
        counts = np.zeros(16, np.int64)
        for i in order:
            if count == maximum:
                break
            if not per_class or counts[logs[i]] < seed_count:
                seeds[count] = i
                counts[logs[i]] += 1
                count += 1
        # Copy seeds: subsequent pool replacements must not change the sources.
        vv, cc, ll = vectors[seeds[:count]].copy(), costs[seeds[:count]].copy(), logs[seeds[:count]].copy()
        best = costs[:used].min()
        for s in range(count):
            present = np.empty(len(prior), np.int64)
            n = 0
            for word in range(vv.shape[1]):
                x = vv[s, word]
                while x:
                    r = _ctz(x) + 64 * word
                    x &= x - np.uint64(1)
                    present[n] = r
                    n += 1
            for a in range(n):
                i = present[a]
                for b in range(a + 1, n):
                    j = present[b]
                    lo, hi = cols[i, 0] ^ cols[j, 0], cols[i, 1] ^ cols[j, 1]
                    k = find(lo, hi, cols, heads, ii, jj)
                    while k >= 0:
                        u, v = np.int64(ii[k]), np.int64(jj[k])
                        if i == u and j == v:
                            k = nxt[k]
                            continue
                        cost = cc[s] - prior[i] - prior[j]
                        on_u = ((vv[s, u // 64] >> np.uint64(u % 64)) & np.uint64(1)) != 0
                        on_v = ((vv[s, v // 64] >> np.uint64(v % 64)) & np.uint64(1)) != 0
                        if u == i or u == j:
                            on_u = not on_u
                        if v == i or v == j:
                            on_v = not on_v
                        cost += -prior[u] if on_u else prior[u]
                        cost += -prior[v] if on_v else prior[v]
                        best = min(best, cost)
                        if cost <= best + gap:
                            vec = vv[s].copy()
                            lg = ll[s]
                            for r in (i, j, u, v):
                                vec[r // 64] ^= np.uint64(1) << np.uint64(r % 64)
                                lg ^= logical[r]
                            used, drop = insert(
                                vec, cost, lg, vectors, costs, logs, hashes,
                                used, vector_hash(vec), index)
                            drops += drop
                        k = nxt[k]
    return used, drops
