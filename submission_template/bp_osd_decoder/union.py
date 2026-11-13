"""Multi-ranking OSD candidate union with exact error-vector deduplication.

Unlike an average of normalized class probabilities, every distinct mechanism
vector is counted ONCE with its original independent-mechanism likelihood.
A bounded pool retains the cheapest distinct candidates. Full packed vectors
are compared, so 64-bit hash collisions cannot silently merge errors.
Reference bp_osd.py is unchanged.
"""
import numba
from numba import njit, prange
import numpy as np

from bp_osd_decoder.enhanced import _bit, _ctz
from bp_osd_decoder.pair_index import find, get_index
from bp_osd_decoder.bp_checkpoints import posterior_checkpoints
from bp_osd_decoder.pool import create as create_pool_index, insert as insert_hashed
from bp_osd_decoder.local_closure import expand as expand_pair_closure


@njit(cache=True)
def _rank_post(D, graph_ec, ev, check_ptr, prior, iterations, alpha, damping):
    """Same damped, averaged min-sum ranking as the reference."""
    E, n = len(ev), len(prior)
    c2v = np.zeros(E, np.float32)
    post = prior.copy()
    acc = np.zeros(n, np.float32)
    xs = np.empty(E, np.float32)
    a, damp = np.float32(alpha), np.float32(damping)
    big = np.float32(1e30)
    for _ in range(iterations):
        for i in range(len(D)):
            lo, hi = check_ptr[i], check_ptr[i + 1]
            neg = D[i]
            min1, min2, arg = big, big, -1
            for e in range(lo, hi):
                x = post[ev[e]] - c2v[e]
                xs[e] = x
                neg ^= x < 0
                ax = abs(x)
                min2 = min(min2, max(min1, ax))
                if ax < min1:
                    arg = e
                min1 = min(min1, ax)
            scale = (1 - damp) * a * (np.float32(-1) if neg else np.float32(1))
            for e in range(lo, hi):
                mag = min2 if e == arg else min1
                sg = -scale if xs[e] < 0 else scale
                c2v[e] = damp * c2v[e] + sg * mag
        post[:] = prior
        for e in range(E):
            post[ev[e]] += c2v[e]
        acc += post
    return acc


@njit(cache=True)
def _basis(post, cols, D0, D1, m, lam):
    """Fully reduced information set and expansions of nonbasis columns."""
    n = len(post)
    b0, b1 = np.empty(m, np.uint64), np.empty(m, np.uint64)
    c0, c1 = np.empty(m, np.uint64), np.empty(m, np.uint64)
    sel = np.empty(m, np.int64)
    kidx = np.empty(128, np.int64)
    outside = np.empty(lam, np.int64)
    nk0, nk1 = np.empty(lam, np.uint64), np.empty(lam, np.uint64)
    K = min(n, m + lam + 128)
    for attempt in range(2):
        if K < n and attempt == 0:
            threshold = np.partition(post.copy(), K - 1)[K - 1]
            cand = np.nonzero(post <= threshold)[0]
            order = cand[np.argsort(post[cand])]
        else:
            order = np.argsort(post)
        rank, nn = 0, 0
        pm0, pm1 = np.uint64(0), np.uint64(0)
        for t in range(len(order)):
            if rank == m and nn >= lam:
                break
            j = order[t]
            v0, v1 = cols[j, 0], cols[j, 1]
            k0, k1 = np.uint64(0), np.uint64(0)
            for w in range(2):
                x = (v0 & pm0) if w == 0 else (v1 & pm1)
                while x:
                    k = kidx[_ctz(x) + 64 * w]
                    x &= x - np.uint64(1)
                    v0 ^= b0[k]
                    v1 ^= b1[k]
                    k0 ^= c0[k]
                    k1 ^= c1[k]
            if v0 == 0 and v1 == 0:
                if nn < lam:
                    outside[nn], nk0[nn], nk1[nn] = j, k0, k1
                    nn += 1
                continue
            if v0:
                pk = _ctz(v0)
                pb0, pb1 = np.uint64(1) << np.uint64(pk), np.uint64(0)
            else:
                pk = 64 + _ctz(v1)
                pb0, pb1 = np.uint64(0), np.uint64(1) << np.uint64(pk - 64)
            if rank < 64:
                k0 ^= np.uint64(1) << np.uint64(rank)
            else:
                k1 ^= np.uint64(1) << np.uint64(rank - 64)
            for k in range(rank):
                if (b0[k] & pb0) | (b1[k] & pb1):
                    b0[k] ^= v0
                    b1[k] ^= v1
                    c0[k] ^= k0
                    c1[k] ^= k1
            b0[rank], b1[rank] = v0, v1
            c0[rank], c1[rank] = k0, k1
            sel[rank], kidx[pk] = j, rank
            pm0 |= pb0
            pm1 |= pb1
            rank += 1
        if rank == m and nn >= lam:
            break
    kd0, kd1 = np.uint64(0), np.uint64(0)
    for w in range(2):
        x = (D0 & pm0) if w == 0 else (D1 & pm1)
        while x:
            k = kidx[_ctz(x) + 64 * w]
            x &= x - np.uint64(1)
            kd0 ^= c0[k]
            kd1 ^= c1[k]
    return sel[:rank], outside[:nn], nk0[:nn], nk1[:nn], kd0, kd1


@njit(cache=True)
def _make_vector(sel, z0, z1, j, k, words):
    vector = np.zeros(words, np.uint64)
    for w in range(2):
        x = z0 if w == 0 else z1
        while x:
            r = _ctz(x) + 64 * w
            x &= x - np.uint64(1)
            col = sel[r]
            vector[col // 64] ^= np.uint64(1) << np.uint64(col % 64)
    if j >= 0:
        vector[j // 64] ^= np.uint64(1) << np.uint64(j % 64)
    if k >= 0:
        vector[k // 64] ^= np.uint64(1) << np.uint64(k % 64)
    return vector


@njit(cache=True)
def _hash(vector):
    # Hash is an index/filter ONLY; packed vector equality resolves collisions.
    h = np.uint64(0xCBF29CE484222325)
    for x in vector:
        h = (h ^ x) * np.uint64(0x100000001B3)
    return h


@njit(cache=True)
def _insert(vector, cost, logical, vectors, costs, logs, hashes, used):
    h = _hash(vector)
    # A small retained set: vector comparison only when hashes coincide.
    for i in range(used):
        if hashes[i] == h and np.all(vectors[i] == vector):
            return used, 0
    if used < len(costs):
        slot = used
        used += 1
        dropped = 0
    else:
        slot = np.argmax(costs)
        if cost >= costs[slot]:
            return used, 1
        dropped = 1
    vectors[slot] = vector
    costs[slot], logs[slot], hashes[slot] = cost, logical, h
    return used, dropped


@njit(cache=True)
def _put(vector, cost, logical, vectors, costs, logs, hashes, used, index):
    if len(index[0]):
        return insert_hashed(vector, cost, logical, vectors, costs, logs, hashes,
                             used, _hash(vector), index)
    return _insert(vector, cost, logical, vectors, costs, logs, hashes, used)


@njit(cache=True)
def _collect(post, prior, cols, logical, D0, D1, m, lam, lam2,
             vectors, costs, logs, hashes, used, gap, index):
    sel, outside, k0, k1, z0, z1 = _basis(post, cols, D0, D1, m, lam)
    rank, nn = len(sel), len(outside)
    sgn = np.empty(rank, np.float64)
    basecost, baselog = 0.0, np.uint8(0)
    for r in range(rank):
        c = prior[sel[r]]
        if _bit(z0, z1, r):
            basecost += c
            baselog ^= logical[sel[r]]
            sgn[r] = -c
        else:
            sgn[r] = c
    best = min(basecost, costs[:used].min()) if used else basecost
    words = vectors.shape[1]
    drops = 0
    if basecost <= best + gap:
        vec = _make_vector(sel, z0, z1, -1, -1, words)
        used, d = _put(vec, basecost, baselog, vectors, costs, logs, hashes, used, index)
        drops += d
    dc, dl = np.empty(nn, np.float64), np.empty(nn, np.uint8)
    for a in range(nn):
        ca, la = float(prior[outside[a]]), logical[outside[a]]
        for w in range(2):
            x = k0[a] if w == 0 else k1[a]
            while x:
                r = _ctz(x) + 64 * w
                x &= x - np.uint64(1)
                ca += sgn[r]
                la ^= logical[sel[r]]
        dc[a], dl[a] = ca, la
        cost = basecost + ca
        best = min(best, cost)
        if cost <= best + gap:
            vec = _make_vector(sel, z0 ^ k0[a], z1 ^ k1[a], outside[a], -1, words)
            used, d = _put(vec, cost, baselog ^ la, vectors, costs, logs, hashes, used, index)
            drops += d
    n2 = min(lam2, nn)
    for a in range(n2):
        for b in range(a + 1, n2):
            cost = basecost + dc[a] + dc[b]
            for w in range(2):
                x = (k0[a] & k0[b]) if w == 0 else (k1[a] & k1[b])
                while x:
                    r = _ctz(x) + 64 * w
                    x &= x - np.uint64(1)
                    cost -= 2.0 * sgn[r]
            best = min(best, cost)
            if cost <= best + gap:
                vec = _make_vector(sel, z0 ^ k0[a] ^ k0[b], z1 ^ k1[a] ^ k1[b],
                                   outside[a], outside[b], words)
                used, d = _put(vec, cost, baselog ^ dl[a] ^ dl[b],
                               vectors, costs, logs, hashes, used, index)
                drops += d
    return used, drops


@njit(cache=True)
def _collect_exact3(D0, D1, cols, logical, prior, heads, nxt, ii, jj,
                    vectors, costs, logs, hashes, used, gap, index):
    """Every weight<=3 vector; i<j<r counts each triple exactly once."""
    best = costs[:used].min()
    words, dropped = vectors.shape[1], 0
    if D0 == 0 and D1 == 0:
        used, drop = _put(np.zeros(words, np.uint64), 0., np.uint8(0),
                          vectors, costs, logs, hashes, used, index)
        dropped += drop
        best = 0.
    # r=-1 searches exact pairs. All other r search pairs XOR column r.
    for r in range(-1, len(prior)):
        lo, hi = D0, D1
        if r >= 0:
            lo ^= cols[r, 0]
            hi ^= cols[r, 1]
            if lo == 0 and hi == 0:
                cost = float(prior[r])
                best = min(best, cost)
                if cost <= best + gap:
                    v = np.zeros(words, np.uint64)
                    v[r // 64] |= np.uint64(1) << np.uint64(r % 64)
                    used, drop = _put(v, cost, logical[r], vectors, costs, logs, hashes, used, index)
                    dropped += drop
        k = find(lo, hi, cols, heads, ii, jj)
        while k >= 0:
            a, b = ii[k], jj[k]
            if r < 0 or b < r:
                cost = float(prior[a]) + float(prior[b])
                log = logical[a] ^ logical[b]
                if r >= 0:
                    cost += prior[r]
                    log ^= logical[r]
                best = min(best, cost)
                if cost <= best + gap:
                    v = np.zeros(words, np.uint64)
                    v[a // 64] |= np.uint64(1) << np.uint64(a % 64)
                    v[b // 64] |= np.uint64(1) << np.uint64(b % 64)
                    if r >= 0:
                        v[r // 64] |= np.uint64(1) << np.uint64(r % 64)
                    used, drop = _put(v, cost, log, vectors, costs, logs, hashes, used, index)
                    dropped += drop
            k = nxt[k]
    return used, dropped


@njit(parallel=True, cache=True)
def _batch(D, ec, ev, check_ptr, cols, logical, prior, schedules, lam, lam2, capacity, gap,
           heads, nxt, ii, jj, share_prefix, hashed_pool, jitter_priors, jitter_schedule,
           exact_order3, closure_seeds, closure_rounds, closure_per_class):
    cp = np.zeros((len(D), 16), np.float32)
    retained = np.zeros(len(D), np.int64)
    overflow = np.zeros(len(D), np.int64)
    for s in prange(len(D)):
        D0, D1 = np.uint64(0), np.uint64(0)
        for r in range(D.shape[1]):
            if D[s, r]:
                if r < 64:
                    D0 |= np.uint64(1) << np.uint64(r)
                else:
                    D1 |= np.uint64(1) << np.uint64(r - 64)
        vecs = np.empty((capacity, (len(prior) + 63) // 64), np.uint64)
        costs = np.empty(capacity, np.float64)
        logs = np.empty(capacity, np.uint8)
        hashes = np.empty(capacity, np.uint64)
        index = create_pool_index(capacity, hashed_pool)
        used, dropped = 0, 0
        snapshots = (posterior_checkpoints(D[s], ev, check_ptr, prior, schedules)
                     if share_prefix else np.empty((0, 0), np.float32))
        for schedule_index in range(len(schedules)):
            schedule = schedules[schedule_index]
            post = (snapshots[schedule_index] if share_prefix else
                    _rank_post(D[s], ec, ev, check_ptr, prior,
                               int(schedule[0]), schedule[1], schedule[2]))
            used, drop = _collect(post, prior, cols, logical, D0, D1, D.shape[1], lam, lam2,
                                  vecs, costs, logs, hashes, used, gap, index)
            dropped += drop
        # Perturb the ranking prior ONLY. Candidate costs always use original
        # llr0, so a changed proposal distribution cannot distort the weights.
        for proposal_prior in jitter_priors:
            post = _rank_post(D[s], ec, ev, check_ptr, proposal_prior,
                              int(jitter_schedule[0]), jitter_schedule[1], jitter_schedule[2])
            used, drop = _collect(post, prior, cols, logical, D0, D1, D.shape[1], lam, lam2,
                                  vecs, costs, logs, hashes, used, gap, index)
            dropped += drop
        if exact_order3:
            used, drop = _collect_exact3(
                D0, D1, cols, logical, prior, heads, nxt, ii, jj,
                vecs, costs, logs, hashes, used, gap, index)
            dropped += drop
        if closure_seeds > 0 and closure_rounds > 0:
            used, drop = expand_pair_closure(
                cols, logical, prior, heads, nxt, ii, jj,
                vecs, costs, logs, hashes, used, gap, index,
                closure_seeds, closure_rounds, closure_per_class)
            dropped += drop
        best = costs[:used].min()
        mass = np.zeros(16, np.float64)
        for i in range(used):
            if costs[i] <= best + gap:
                mass[logs[i]] += np.exp(best - costs[i])
        cp[s] = mass / mass.sum()
        retained[s], overflow[s] = used, dropped
    return cp, retained, overflow


def decode_union(graph, detector_events, schedules=((3, 0.8, 0.5), (5, 0.8, 0.5)),
                 lam=400, lam2=100, capacity=512, gap=16.0,
                 return_classp=False, return_diagnostics=False, exact_order3=False,
                 share_prefix=True, hashed_pool=True, jitter_trials=0,
                 jitter_strength=1.0, jitter_seed=860001,
                 jitter_schedule=(5, 0.8, 0.5), closure_seeds=0,
                 closure_rounds=1, closure_per_class=False):
    if capacity < 1 or gap < 0:
        raise ValueError("capacity must be positive and gap nonnegative")
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    if closure_seeds < 0 or closure_rounds < 0:
        raise ValueError("closure seeds/rounds must be nonnegative")
    if closure_seeds and not hashed_pool:
        raise ValueError("pair closure requires the exact hashed pool")
    index = (get_index(graph["cols"]) if exact_order3 or closure_seeds
             else (np.empty(0, np.int32),) * 4)
    if jitter_trials < 0 or jitter_strength < 0:
        raise ValueError("jitter trials and strength must be nonnegative")
    rng = np.random.default_rng(jitter_seed)
    jitter_priors = np.maximum(
        .05, graph["llr0"][None, :] +
        rng.normal(0, jitter_strength, (jitter_trials, len(graph["llr0"])))
    ).astype(np.float32)
    cp, retained, overflow = _batch(
        D, graph["ec"], graph["ev"], graph["check_ptr"], graph["cols"],
        graph["logical"], graph["llr0"], np.asarray(schedules, dtype=np.float64),
        lam, lam2, capacity, gap, *index, share_prefix, hashed_pool,
        jitter_priors, np.asarray(jitter_schedule, dtype=np.float64),
        exact_order3, closure_seeds, closure_rounds, closure_per_class)
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)
    out = (cp @ bits > 0.5).astype(np.uint8)
    if return_diagnostics:
        return out, cp, retained, overflow
    return (out, cp) if return_classp else out
