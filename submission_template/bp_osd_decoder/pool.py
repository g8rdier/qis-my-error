"""Exact, collision-safe indexing and max-heap for a bounded candidate pool.

The retained arrays and first-index argmax tie break match the original linear
implementation. Tombstones are periodically rebuilt; full-vector equality,
not hashes, determines duplication. Pool updates use no labels or randomness.
"""
from numba import njit
import numpy as np


@njit(cache=True)
def create(capacity, enabled=True):
    size = 1
    while size < 4 * capacity:
        size *= 2
    return (np.full(size if enabled else 0, -1, np.int64),
            np.zeros(1, np.int64), np.empty(capacity, np.int64),
            np.empty(capacity, np.int64))


@njit(cache=True)
def _bucket(h, mask):
    # Avalanche is essential: FNV over uint64 words has poor LOW-bit dispersion.
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return np.int64(h & np.uint64(mask))


@njit(cache=True)
def _greater(i, j, costs):
    return costs[i] > costs[j] or (costs[i] == costs[j] and i < j)


@njit(cache=True)
def insert(vector, cost, logical, vectors, costs, logs, hashes, used, h, index):
    slots, tombs, positions, heap = index
    mask = len(slots) - 1
    if tombs[0] > len(costs):
        slots[:] = -1
        for i in range(used):
            pos = _bucket(hashes[i], mask)
            while slots[pos] >= 0:
                pos = (pos + 1) & mask
            slots[pos], positions[i] = i, pos
        tombs[0] = 0
    pos = _bucket(h, mask)
    first_tomb = -1
    while slots[pos] != -1:
        slot = slots[pos]
        if slot == -2:
            if first_tomb < 0:
                first_tomb = pos
        elif hashes[slot] == h and np.all(vectors[slot] == vector):
            return used, 0
        pos = (pos + 1) & mask
    if first_tomb >= 0:
        pos = first_tomb
    if used < len(costs):
        slot = used
        used += 1
        dropped = 0
    else:
        slot = heap[0]
        if cost >= costs[slot]:
            return used, 1
        slots[positions[slot]] = -2
        tombs[0] += 1
        dropped = 1
    if slots[pos] == -2:
        tombs[0] -= 1
    slots[pos], positions[slot] = slot, pos
    vectors[slot] = vector
    costs[slot], logs[slot], hashes[slot] = cost, logical, h
    if not dropped:
        k = used - 1
        while k:
            parent = (k - 1) // 2
            if not _greater(slot, heap[parent], costs):
                break
            heap[k] = heap[parent]
            k = parent
        heap[k] = slot
    else:
        k = 0
        while 2 * k + 1 < used:
            child = 2 * k + 1
            if child + 1 < used and _greater(heap[child + 1], heap[child], costs):
                child += 1
            if not _greater(heap[child], slot, costs):
                break
            heap[k] = heap[child]
            k = child
        heap[k] = slot
    return used, dropped
