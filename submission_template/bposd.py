"""Belief propagation (normalised min-sum) with an OSD fallback (combination sweep + weighted vote), written from scratch with numba.

Decoding problem: D = H e mod 2, O = L e mod 2, e_i ~ Bernoulli(p_i). H has 70 rows, so every column fits in two
uint64 words (bits 0-63 and 64-69), which makes the GF(2) elimination in OSD very cheap.
"""
import numba
import numpy as np


class Graph:
    """Sparse Tanner graph of H plus the bit-packed columns used by OSD."""

    def __init__(self, H: np.ndarray, L: np.ndarray, p: np.ndarray):
        H = np.asarray(H, dtype=np.uint8)
        m, n = H.shape
        assert m <= 128
        checks, vars_ = np.nonzero(H)  # row-major: edges grouped by check
        self.n_checks, self.n_vars = m, n
        self.chk_ptr = np.searchsorted(checks, np.arange(m + 1)).astype(np.int64)
        self.edge_var = vars_.astype(np.int64)
        order = np.argsort(vars_, kind="stable")  # edges grouped by variable
        self.var_ptr = np.searchsorted(vars_[order], np.arange(n + 1)).astype(np.int64)
        self.var_edge = order.astype(np.int64)
        self.prior_llr = np.log((1 - p) / p).astype(np.float64)
        rows = np.arange(m, dtype=np.uint64)
        lo_rows = rows < 64
        self.col_lo = (H[lo_rows].astype(np.uint64) << rows[lo_rows, None]).sum(0).astype(np.uint64)
        self.col_hi = (H[~lo_rows].astype(np.uint64) << (rows[~lo_rows, None] - np.uint64(64))).sum(0).astype(np.uint64)
        self.col_obs = (np.asarray(L, dtype=np.uint8).astype(np.uint64) << np.arange(L.shape[0], dtype=np.uint64)[:, None]).sum(0).astype(np.uint8)

    def arrays(self):
        return (self.chk_ptr, self.edge_var, self.var_ptr, self.var_edge, self.prior_llr,
                self.col_lo, self.col_hi, self.col_obs)


@numba.njit(cache=True, inline="always")
def _bit(lo, hi, t):
    if t < 64:
        return (lo >> np.uint64(t)) & np.uint64(1)
    return (hi >> np.uint64(t - 64)) & np.uint64(1)


@numba.njit(cache=True, inline="always")
def _mask_cost(k_lo, k_hi, rank, slot_w, slot_obs):
    w = 0.0
    obs = np.uint8(0)
    for r in range(rank):
        if _bit(k_lo, k_hi, r):
            w += slot_w[r]
            obs ^= slot_obs[r]
    return w, obs


@numba.njit(cache=True)
def _osd(order, n_checks, col_lo, col_hi, col_obs, prior_llr, syn_lo, syn_hi, lam1, lam2):
    """OSD with combination sweep and a probability-weighted vote per observable bit.

    Columns are taken in `order` (most likely flipped first) to build a GF(2) basis (OSD-0). Each candidate solution
    of H e = D is the OSD-0 solution plus a set of non-basis columns (all singles among the first lam1 of them, all
    pairs among the first lam2), with the basis part adjusted to keep the syndrome. Every candidate is weighted by its
    prior probability exp(-sum of llr); the returned mask has bit k set if the weight with observable k flipped is larger.
    """
    b_lo = np.zeros(n_checks, np.uint64)
    b_hi = np.zeros(n_checks, np.uint64)
    c_lo = np.zeros(n_checks, np.uint64)  # which basis slots (selected columns) make up this basis vector
    c_hi = np.zeros(n_checks, np.uint64)
    has = np.zeros(n_checks, np.bool_)
    slot_obs = np.zeros(128, np.uint8)
    slot_w = np.zeros(128)
    lam = max(lam1, lam2)
    np_lo = np.zeros(lam, np.uint64)  # non-basis columns expressed in basis slots
    np_hi = np.zeros(lam, np.uint64)
    np_obs = np.zeros(lam, np.uint8)
    np_w = np.zeros(lam)
    n_np = 0
    rank = 0
    for j in order:
        if rank == n_checks and n_np == lam:
            break
        v_lo, v_hi = col_lo[j], col_hi[j]
        k_lo = np.uint64(0)
        k_hi = np.uint64(0)
        if rank < 64:
            k_lo = np.uint64(1) << np.uint64(rank)
        else:
            k_hi = np.uint64(1) << np.uint64(rank - 64)
        inserted = False
        for t in range(n_checks - 1, -1, -1):
            if _bit(v_lo, v_hi, t):
                if has[t]:
                    v_lo ^= b_lo[t]
                    v_hi ^= b_hi[t]
                    k_lo ^= c_lo[t]
                    k_hi ^= c_hi[t]
                else:
                    has[t] = True
                    b_lo[t], b_hi[t], c_lo[t], c_hi[t] = v_lo, v_hi, k_lo, k_hi
                    slot_obs[rank] = col_obs[j]
                    slot_w[rank] = prior_llr[j]
                    rank += 1
                    inserted = True
                    break
        if not inserted and n_np < lam:
            # dependent column: k now holds its own slot bit (for the rank it would have had) plus the slots
            # it reduces to; drop the own bit to get its expression in existing basis slots
            if rank < 64:
                k_lo ^= np.uint64(1) << np.uint64(rank)
            else:
                k_hi ^= np.uint64(1) << np.uint64(rank - 64)
            np_lo[n_np], np_hi[n_np], np_obs[n_np], np_w[n_np] = k_lo, k_hi, col_obs[j], prior_llr[j]
            n_np += 1
    s_lo, s_hi = syn_lo, syn_hi
    k0_lo = np.uint64(0)
    k0_hi = np.uint64(0)
    for t in range(n_checks - 1, -1, -1):
        if _bit(s_lo, s_hi, t) and has[t]:
            s_lo ^= b_lo[t]
            s_hi ^= b_hi[t]
            k0_lo ^= c_lo[t]
            k0_hi ^= c_hi[t]
    # weighted vote; weights exp(-(w - w_ref)) with w_ref the OSD-0 cost (rescaled when a cheaper one appears)
    w0, o0 = _mask_cost(k0_lo, k0_hi, rank, slot_w, slot_obs)
    w_ref = w0
    tot = 1.0
    ones = np.zeros(4)
    for b in range(4):
        if (o0 >> b) & 1:
            ones[b] += 1.0
    n1 = min(lam1, n_np)
    n2 = min(lam2, n_np)
    n_cand = n1 + n2 * (n2 - 1) // 2
    for q in range(n_cand):
        if q < n1:  # single flip of non-basis column q
            m_lo, m_hi = k0_lo ^ np_lo[q], k0_hi ^ np_hi[q]
            extra_w, extra_o = np_w[q], np_obs[q]
        else:  # pair (i, j), i < j < n2, enumerated row by row
            r = q - n1
            i = 0
            while r >= n2 - 1 - i:
                r -= n2 - 1 - i
                i += 1
            j = i + 1 + r
            m_lo, m_hi = k0_lo ^ np_lo[i] ^ np_lo[j], k0_hi ^ np_hi[i] ^ np_hi[j]
            extra_w, extra_o = np_w[i] + np_w[j], np_obs[i] ^ np_obs[j]
        w, o = _mask_cost(m_lo, m_hi, rank, slot_w, slot_obs)
        w += extra_w
        o ^= extra_o
        if w < w_ref:
            f = np.exp(w - w_ref)
            tot *= f
            for bb in range(4):
                ones[bb] *= f
            w_ref = w
        x = np.exp(w_ref - w)
        tot += x
        for bb in range(4):
            if (o >> bb) & 1:
                ones[bb] += x
    obs = np.uint8(0)
    for b in range(4):
        if 2.0 * ones[b] > tot:
            obs |= np.uint8(1 << b)
    return obs


@numba.njit(cache=True, parallel=True)
def bp_osd(syndromes, chk_ptr, edge_var, var_ptr, var_edge, prior_llr, col_lo, col_hi, col_obs,
           max_iter, scaling, lam1, lam2, osd_always, avg_post, out_obs, converged):
    """syndromes: (N, n_checks) uint8. Writes out_obs (N,) observable bitmask and converged (N,) bool."""
    n_checks = len(chk_ptr) - 1
    n_vars = len(var_ptr) - 1
    n_edges = len(edge_var)
    for s in numba.prange(syndromes.shape[0]):
        D = syndromes[s]
        syn_lo = np.uint64(0)
        syn_hi = np.uint64(0)
        any_fired = False
        for t in range(n_checks):
            if D[t]:
                any_fired = True
                if t < 64:
                    syn_lo |= np.uint64(1) << np.uint64(t)
                else:
                    syn_hi |= np.uint64(1) << np.uint64(t - 64)
        if not any_fired:
            out_obs[s] = 0
            converged[s] = True
            continue
        v2c = np.empty(n_edges)
        c2v = np.zeros(n_edges)
        post = prior_llr.copy()
        post_sum = np.zeros(n_vars)
        for e in range(n_edges):
            v2c[e] = prior_llr[edge_var[e]]
        done = False
        for it in range(max_iter):
            # check -> variable (min-sum)
            for c in range(n_checks):
                sgn = -1.0 if D[c] else 1.0
                m1 = np.inf
                m2 = np.inf
                arg = -1
                for e in range(chk_ptr[c], chk_ptr[c + 1]):
                    x = v2c[e]
                    if x < 0:
                        sgn = -sgn
                        x = -x
                    if x < m1:
                        m2 = m1
                        m1 = x
                        arg = e
                    elif x < m2:
                        m2 = x
                for e in range(chk_ptr[c], chk_ptr[c + 1]):
                    mag = m2 if e == arg else m1
                    sg = -sgn if v2c[e] < 0 else sgn
                    c2v[e] = scaling * sg * mag
            # variable -> check, posterior
            for v in range(n_vars):
                tot = prior_llr[v]
                for k in range(var_ptr[v], var_ptr[v + 1]):
                    tot += c2v[var_edge[k]]
                post[v] = tot
                post_sum[v] += tot
                for k in range(var_ptr[v], var_ptr[v + 1]):
                    e = var_edge[k]
                    v2c[e] = tot - c2v[e]
            # hard decision and syndrome check
            ok = True
            for c in range(n_checks):
                par = 0
                for e in range(chk_ptr[c], chk_ptr[c + 1]):
                    if post[edge_var[e]] < 0:
                        par ^= 1
                if par != D[c]:
                    ok = False
                    break
            if ok:
                done = True
                break
        if done and not osd_always:
            obs = np.uint8(0)
            for v in range(n_vars):
                if post[v] < 0:
                    obs ^= col_obs[v]
            out_obs[s] = obs
            converged[s] = True
        else:
            order = np.argsort(post_sum if avg_post else post)  # most likely to be flipped first
            out_obs[s] = _osd(order, n_checks, col_lo, col_hi, col_obs, prior_llr, syn_lo, syn_hi, lam1, lam2)
            converged[s] = done


def decode_bposd(graph: Graph, detector_events: np.ndarray, max_iter: int = 30, scaling: float = 0.6,
                 lam1: int = 0, lam2: int = 0, osd_always: bool = False, avg_post: bool = False):
    """Returns (predictions (N, 4) uint8, converged (N,) bool)."""
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    obs = np.zeros(len(D), np.uint8)
    conv = np.zeros(len(D), np.bool_)
    bp_osd(D, *graph.arrays(), max_iter, scaling, lam1, lam2, osd_always, avg_post, obs, conv)
    pred = ((obs[:, None] >> np.arange(4, dtype=np.uint8)) & 1).astype(np.uint8)
    return pred, conv
