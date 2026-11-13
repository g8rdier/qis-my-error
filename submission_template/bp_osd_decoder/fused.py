"""Speed-only fused implementation of the frozen balanced configuration.

BP runs to iteration five once. Easy shots stop there. Uncertain shots reuse
the iteration-three/five snapshots and continue to eight for the same union.
The initial and candidate-pool settings are deliberately explicit arguments.
No learned weights, new decision rule, labels, or repeated-call caches.
"""
from numba import njit, prange
import numpy as np
from bp_osd_decoder.enhanced import _osd
from bp_osd_decoder.union import _collect
from bp_osd_decoder.pool import create as create_pool_index

BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)


@njit(cache=True)
def _step(D, ev, ptr, prior, c2v, post, acc, xs, alpha, damping):
    a, damp, big = np.float32(alpha), np.float32(damping), np.float32(1e30)
    for i in range(len(D)):
        lo, hi = ptr[i], ptr[i + 1]
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
    for e in range(len(ev)):
        post[ev[e]] += c2v[e]
    acc += post


@njit(parallel=True, cache=True)
def _batch(D, ev, ptr, prior, cols, logical, initial_lam, initial_lam2,
           union_lam, union_lam2, capacity, gap, threshold, alpha, damping, hashed_pool):
    cp = np.zeros((len(D), 16), np.float32)
    masks = np.zeros(len(D), np.uint8)
    routed = np.zeros(len(D), np.uint8)
    n = len(prior)
    for s in prange(len(D)):
        d = D[s]
        c2v = np.zeros(len(ev), np.float32)
        post, acc = prior.copy(), np.zeros(n, np.float32)
        xs = np.empty(len(ev), np.float32)
        snapshot3 = np.empty(n, np.float32)
        for it in range(1, 6):
            _step(d, ev, ptr, prior, c2v, post, acc, xs, alpha, damping)
            if it == 3:
                snapshot3[:] = acc
        d0, d1 = np.uint64(0), np.uint64(0)
        for r in range(len(d)):
            if d[r]:
                if r < 64:
                    d0 |= np.uint64(1) << np.uint64(r)
                else:
                    d1 |= np.uint64(1) << np.uint64(r - 64)
        masks[s] = _osd(acc, prior, cols, logical, d0, d1, len(d),
                         initial_lam, initial_lam2, 2, cp[s], 1.0, 0, 0)
        route = False
        for bit in range(4):
            marginal = np.float32(0)
            for cls in range(16):
                if (cls >> bit) & 1:
                    marginal += cp[s, cls]
            if marginal > threshold and marginal < 1 - threshold:
                route = True
        if not route:
            continue
        routed[s] = 1
        vectors = np.empty((capacity, (n + 63) // 64), np.uint64)
        costs = np.empty(capacity, np.float64)
        logs = np.empty(capacity, np.uint8)
        hashes = np.empty(capacity, np.uint64)
        index = create_pool_index(capacity, hashed_pool)
        used, _ = _collect(snapshot3, prior, cols, logical, d0, d1, len(d),
                            union_lam, union_lam2, vectors, costs, logs, hashes, 0, gap, index)
        used, _ = _collect(acc, prior, cols, logical, d0, d1, len(d),
                            union_lam, union_lam2, vectors, costs, logs, hashes, used, gap, index)
        for _ in range(3):
            _step(d, ev, ptr, prior, c2v, post, acc, xs, alpha, damping)
        used, _ = _collect(acc, prior, cols, logical, d0, d1, len(d),
                            union_lam, union_lam2, vectors, costs, logs, hashes, used, gap, index)
        best = costs[:used].min()
        mass = np.zeros(16, np.float64)
        for j in range(used):
            if costs[j] <= best + gap:
                mass[logs[j]] += np.exp(best - costs[j])
        cp[s] = mass / mass.sum()
    return masks, cp, routed


def decode_fused(graph, detector_events, initial=None, refiner=None,
                 route_threshold=0.05, return_classp=False):
    initial = initial or {"max_iter": 5, "refine_thr": 0}
    refiner = refiner or {"union": True, "lam": 600, "lam2": 200}
    schedules = refiner.get("schedules", [[3, .8, .5], [5, .8, .5], [8, .8, .5]])
    alpha, damping = initial.get("alpha", .8), initial.get("damping", .5)
    expected = [[3, alpha, damping], [5, alpha, damping], [8, alpha, damping]]
    # Fail closed: this specialization must never silently change other configs.
    if (initial.get("max_iter", 3) != 5 or initial.get("refine_thr", .3) != 0
            or not initial.get("avg_post", True) or not initial.get("always_osd", True)
            or initial.get("coset", 2) != 2 or initial.get("temp", 1.) != 1.
            or not refiner.get("union", True) or schedules != expected
            or refiner.get("exact_order3", False) or refiner.get("jitter_trials", 0)
            or refiner.get("closure_seeds", 0)):
        raise ValueError("fused backend supports only the frozen balanced schedule")
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    masks, cp, route = _batch(
        D, graph["ev"], graph["check_ptr"], graph["llr0"], graph["cols"], graph["logical"],
        initial.get("lam", 400), initial.get("lam2", 100),
        refiner.get("lam", 600), refiner.get("lam2", 200),
        refiner.get("capacity", 512), refiner.get("gap", 16.),
        route_threshold, alpha, damping, refiner.get("hashed_pool", True))
    pred = np.unpackbits(masks[:, None], axis=1, count=4, bitorder="little")
    use = route.astype(bool)
    pred[use] = (cp[use] @ BITS > .5).astype(np.uint8)
    return (pred, cp) if return_classp else pred
