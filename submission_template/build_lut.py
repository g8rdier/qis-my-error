"""Build an empirical lookup table (LUT) decoder from stim samples.

For every syndrome D seen while sampling, count how often each logical observable flipped. The prediction for D is
the per-bit majority, which is the Bayes-optimal answer for the per-bit LER metric once the counts are large.

Run from the repository root:

    python submission_template/build_lut.py --shots 200_000_000 --min-count 1

Writes submission_template/lut_X.npz and lut_Z.npz.
"""
import argparse
import hashlib
import os
import time

import numba
import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
EMPTY = np.int16(-1)


@numba.njit(cache=True, inline="always")
def _slot(lo, hi, mask):
    h = (lo ^ (np.uint64(hi) * np.uint64(0xC2B2AE3D27D4EB4F))) * np.uint64(0x9E3779B97F4A7C15)
    return np.int64((h >> np.uint64(29)) & mask)


@numba.njit(cache=True)
def _insert(keys_lo, keys_hi, counts, lo, hi, obs, max_fill, fill):
    """Add shots to the open-addressing table. New keys are only added while fill < max_fill."""
    mask = np.uint64(len(keys_lo) - 1)
    for i in range(len(lo)):
        s = _slot(lo[i], hi[i], mask)
        while True:
            if keys_hi[s] == EMPTY:
                if fill >= max_fill:
                    break
                keys_hi[s] = hi[i]
                keys_lo[s] = lo[i]
                fill += 1
            if keys_hi[s] == hi[i] and keys_lo[s] == lo[i]:
                counts[s, 4] += 1
                for b in range(4):
                    counts[s, b] += (obs[i] >> b) & 1
                break
            s = (s + 1) & np.int64(mask)
    return fill


def pack(detectors_bitpacked: np.ndarray):
    """(N, 9) uint8 bit-packed detectors -> (lo uint64 = D0..D63, hi int16 = D64..D69)."""
    d = np.ascontiguousarray(detectors_bitpacked)
    lo = d[:, :8].copy().view(np.uint64).ravel()
    hi = d[:, 8].astype(np.int16)
    return lo, hi


def build(circuit: stim.Circuit, shots: int, chunk: int, log2_cap: int, seed: int):
    cap = 1 << log2_cap
    keys_lo = np.zeros(cap, np.uint64)
    keys_hi = np.full(cap, EMPTY, np.int16)
    counts = np.zeros((cap, 5), np.uint32)
    fill, max_fill = 0, int(cap * 0.7)
    sampler = circuit.compile_detector_sampler(seed=seed)
    t0 = time.time()
    for done in range(0, shots, chunk):
        n = min(chunk, shots - done)
        det, obs = sampler.sample(n, separate_observables=True, bit_packed=True)
        lo, hi = pack(det)
        fill = _insert(keys_lo, keys_hi, counts, lo, hi, obs[:, 0], max_fill, fill)
        print(f"  {done + n:>12,} shots  {fill:>11,} keys  {time.time() - t0:6.1f} s", flush=True)
    used = keys_hi != EMPTY
    return keys_lo[used], keys_hi[used], counts[used]


def to_table(keys_lo, keys_hi, counts, min_count):
    keep = counts[:, 4] >= min_count
    keys_lo, keys_hi, counts = keys_lo[keep], keys_hi[keep], counts[keep]
    pred = (2 * counts[:, :4] > counts[:, 4:5]).astype(np.uint8)  # per-bit majority
    pred_packed = (pred * (1 << np.arange(4, dtype=np.uint8))).sum(1).astype(np.uint8)
    order = np.lexsort((keys_lo, keys_hi))
    keys_lo, keys_hi, pred_packed, total = keys_lo[order], keys_hi[order], pred_packed[order], counts[order, 4]
    offsets = np.searchsorted(keys_hi, np.arange(65)).astype(np.int64)  # bucket per hi value (6 bits)
    return dict(keys_lo=keys_lo, offsets=offsets, pred=pred_packed, count=total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="data/train")
    ap.add_argument("--shots", type=lambda s: int(s.replace("_", "")), default=100_000_000)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--log2-cap", type=int, default=25)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    for i, b in enumerate("XZ"):
        print(f"basis {b}")
        circuit = stim.Circuit.from_file(os.path.join(a.circuits, f"{b}_circuit.stim"))
        table = to_table(*build(circuit, a.shots, a.chunk, a.log2_cap, a.seed + i), a.min_count)
        path = os.path.join(a.out, f"lut_{b}.npz")
        sha = hashlib.sha256(str(circuit).encode()).hexdigest()
        np.savez(path, circuit_sha=np.array(sha), **table)
        print(f"  wrote {path}: {len(table['pred']):,} entries")
