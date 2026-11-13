"""Bit-identical BP posterior snapshots with shared iteration prefixes.

Schedules with the same alpha/damping follow exactly the same message
trajectory. Save its cumulative posteriors at iterations 3, 5 and 8 rather
than restarting and computing 3+5+8 iterations. No probabilities or candidate
selection are changed. Different alpha/damping groups remain independent.
"""
from numba import njit
import numpy as np


@njit(cache=True)
def posterior_checkpoints(D, ev, check_ptr, prior, schedules):
    E, n, count = len(ev), len(prior), len(schedules)
    output = np.empty((count, n), np.float32)
    done = np.zeros(count, np.uint8)
    big = np.float32(1e30)
    for group in range(count):
        if done[group]:
            continue
        alpha, damping = schedules[group, 1], schedules[group, 2]
        stop = 0
        for j in range(count):
            if schedules[j, 1] == alpha and schedules[j, 2] == damping:
                stop = max(stop, int(schedules[j, 0]))
                if schedules[j, 0] == 0:
                    output[j] = 0
                    done[j] = 1
        c2v = np.zeros(E, np.float32)
        post = prior.copy()
        acc = np.zeros(n, np.float32)
        xs = np.empty(E, np.float32)
        a, damp = np.float32(alpha), np.float32(damping)
        for it in range(1, stop + 1):
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
            for j in range(count):
                if (int(schedules[j, 0]) == it and schedules[j, 1] == alpha
                        and schedules[j, 2] == damping):
                    output[j] = acc
                    done[j] = 1
    return output
