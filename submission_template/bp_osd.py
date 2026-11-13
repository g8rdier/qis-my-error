"""Belief propagation (normalised min-sum) with an ordered-statistics (OSD-0) fallback, written from scratch.

Decoding problem: the error model gives H (detectors x error mechanisms), L (observables x mechanisms) and the
prior probability p of each mechanism. Given detector events D, find a likely error e with H e = D (mod 2) and
predict the logical flips L e (mod 2).

1. BP: messages are passed between error nodes and detector nodes of the graph defined by H, starting from the
   priors. Detector-to-error messages are damped (mixed with the previous iteration) and the posteriors are
   averaged over the iterations, which stops min-sum oscillating on this loopy graph. By default OSD always runs
   after a fixed number of iterations; with always_osd=False the hard decision is checked against D instead.
2. OSD-0, if BP has not matched D after max_iter iterations: sort the mechanisms from most to least likely by their
   BP posterior, pick the first linearly independent set of columns of H (Gaussian elimination over GF(2)), and
   solve H e = D using only those columns. This always gives an e that explains D.

Columns of H have 70 bits, stored as two uint64 words. Shots are decoded in parallel with numba.
"""
import numpy as np
from numba import njit, prange


def build_graph(H: np.ndarray, L: np.ndarray, p: np.ndarray):
    """Precompute everything the kernel needs from the error model."""
    H = np.asarray(H, dtype=np.uint8)
    m, n = H.shape
    assert m <= 128
    ec, ev = np.nonzero(H)  # edges sorted by check (detector), then by variable (mechanism)
    check_ptr = np.zeros(m + 1, np.int64)
    np.cumsum(np.bincount(ec, minlength=m), out=check_ptr[1:])
    order = np.argsort(ev, kind="stable")  # edge ids grouped by variable
    var_ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(ev, minlength=n), out=var_ptr[1:])
    cols = np.zeros((n, 2), np.uint64)  # column j of H as a 128-bit mask
    for i, j in zip(ec, ev):
        cols[j, i // 64] |= np.uint64(1) << np.uint64(i % 64)
    logical = (np.asarray(L, np.uint8).T * (1 << np.arange(L.shape[0]))).sum(1).astype(np.uint8)  # 4-bit mask per column
    llr0 = np.log((1 - p) / p).astype(np.float64)
    return dict(ec=ec.astype(np.int64), ev=ev.astype(np.int64), check_ptr=check_ptr, var_edges=order.astype(np.int64),
                var_ptr=var_ptr, cols=cols, logical=logical, llr0=llr0, m=m, n=n)


@njit(cache=True)
def _osd0(post, cols, logical, D0, D1, m):
    """Solve H e = D using the most likely independent columns (ordered by posterior LLR, smallest first)."""
    order = np.argsort(post)
    piv = np.empty(m, np.int64)
    b0 = np.empty(m, np.uint64)
    b1 = np.empty(m, np.uint64)
    c0 = np.empty(m, np.uint64)  # which selected columns make up each basis vector (bitmask over <= 128 slots)
    c1 = np.empty(m, np.uint64)
    sel = np.empty(m, np.int64)
    rank = 0
    one = np.uint64(1)
    for t in range(order.shape[0]):
        j = order[t]
        v0 = cols[j, 0]
        v1 = cols[j, 1]
        k0 = np.uint64(0)
        k1 = np.uint64(0)
        for k in range(rank):
            pk = piv[k]
            hit = ((v0 >> np.uint64(pk)) & one) if pk < 64 else ((v1 >> np.uint64(pk - 64)) & one)
            if hit:
                v0 ^= b0[k]
                v1 ^= b1[k]
                k0 ^= c0[k]
                k1 ^= c1[k]
        if v0 == 0 and v1 == 0:
            continue
        # new pivot: lowest set bit of the reduced column
        if v0 != 0:
            pk = 0
            while not ((v0 >> np.uint64(pk)) & one):
                pk += 1
        else:
            pk = 64
            while not ((v1 >> np.uint64(pk - 64)) & one):
                pk += 1
        if rank < 64:
            k0 ^= one << np.uint64(rank)
        else:
            k1 ^= one << np.uint64(rank - 64)
        piv[rank] = pk
        b0[rank] = v0
        b1[rank] = v1
        c0[rank] = k0
        c1[rank] = k1
        sel[rank] = j
        rank += 1
        if rank == m:
            break
    # reduce the syndrome in insertion order; later basis vectors never contain earlier pivots
    s0 = D0
    s1 = D1
    k0 = np.uint64(0)
    k1 = np.uint64(0)
    for k in range(rank):
        pk = piv[k]
        hit = ((s0 >> np.uint64(pk)) & one) if pk < 64 else ((s1 >> np.uint64(pk - 64)) & one)
        if hit:
            s0 ^= b0[k]
            s1 ^= b1[k]
            k0 ^= c0[k]
            k1 ^= c1[k]
    out = np.uint8(0)
    for r in range(rank):
        bit = ((k0 >> np.uint64(r)) & one) if r < 64 else ((k1 >> np.uint64(r - 64)) & one)
        if bit:
            out ^= logical[sel[r]]
    return out


@njit(parallel=True, cache=True)
def decode_batch(D, ec, ev, check_ptr, var_edges, var_ptr, cols, logical, llr0, max_iter, alpha, always_osd, damping,
                 avg_post, stats):
    """D: (shots, m) uint8. Returns (shots,) uint8 logical masks. stats[s] = BP iterations used, or -1 if OSD ran."""
    shots, m = D.shape
    n = llr0.shape[0]
    E = ec.shape[0]
    out = np.zeros(shots, np.uint8)
    for s in prange(shots):
        d = D[s]
        v2c = np.empty(E, np.float64)
        c2v = np.empty(E, np.float64)
        post = llr0.copy()
        hard = np.zeros(n, np.uint8)
        acc = np.zeros(n, np.float64)  # sum of posteriors over iterations (used when avg_post)
        for e in range(E):
            v2c[e] = llr0[ev[e]]
            c2v[e] = 0.0
        converged = False
        it = 0
        while it < max_iter:
            it += 1
            # detector -> error: normalised min-sum
            for i in range(m):
                sgn = 1.0 if d[i] == 0 else -1.0
                min1 = 1e300
                min2 = 1e300
                arg = -1
                for e in range(check_ptr[i], check_ptr[i + 1]):
                    x = v2c[e]
                    if x < 0:
                        sgn = -sgn
                        x = -x
                    if x < min1:
                        min2 = min1
                        min1 = x
                        arg = e
                    elif x < min2:
                        min2 = x
                for e in range(check_ptr[i], check_ptr[i + 1]):
                    mag = min2 if e == arg else min1
                    sg = sgn if v2c[e] >= 0 else -sgn
                    c2v[e] = damping * c2v[e] + (1.0 - damping) * alpha * sg * mag
            # error -> detector, posteriors and hard decision
            for j in range(n):
                tot = llr0[j]
                for q in range(var_ptr[j], var_ptr[j + 1]):
                    tot += c2v[var_edges[q]]
                post[j] = tot
                acc[j] += tot
                hard[j] = 1 if tot < 0 else 0
                for q in range(var_ptr[j], var_ptr[j + 1]):
                    e = var_edges[q]
                    v2c[e] = tot - c2v[e]
            if always_osd:
                continue  # OSD runs regardless, so skip the convergence check
            # does the hard decision explain the detectors?
            ok = True
            for i in range(m):
                par = 0
                for e in range(check_ptr[i], check_ptr[i + 1]):
                    par ^= hard[ev[e]]
                if par != d[i]:
                    ok = False
                    break
            if ok:
                converged = True
                break
        if converged:
            o = np.uint8(0)
            for j in range(n):
                if hard[j]:
                    o ^= logical[j]
            out[s] = o
            stats[s] = it
        else:
            D0 = np.uint64(0)
            D1 = np.uint64(0)
            for i in range(m):
                if d[i]:
                    if i < 64:
                        D0 |= np.uint64(1) << np.uint64(i)
                    else:
                        D1 |= np.uint64(1) << np.uint64(i - 64)
            out[s] = _osd0(acc if avg_post else post, cols, logical, D0, D1, m)
            stats[s] = -1
    return out


def decode_bp_osd(graph, detector_events, max_iter=5, alpha=0.9, always_osd=True, damping=0.5, avg_post=True,
                  return_stats=False):
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    stats = np.zeros(len(D), np.int64)
    masks = decode_batch(D, graph["ec"], graph["ev"], graph["check_ptr"], graph["var_edges"], graph["var_ptr"],
                         graph["cols"], graph["logical"], graph["llr0"], max_iter, alpha, always_osd, damping, avg_post, stats)
    out = np.unpackbits(masks[:, None], axis=1, count=4, bitorder="little")
    return (out, stats) if return_stats else out
