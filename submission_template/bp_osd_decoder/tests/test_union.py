"""Exact small-model and collision tests for candidate-union scoring."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numba
import numpy as np
from bp_osd_decoder.enhanced import build_graph
from bp_osd_decoder.union import decode_union, _hash, _insert


class TestUnion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(4)

    def test_exact_small_models_and_duplicate_schedules(self):
        rng = np.random.default_rng(4922)
        for _ in range(4):
            H = np.concatenate([np.eye(4, dtype=np.uint8),
                                rng.integers(0, 2, (4, 2), dtype=np.uint8)], axis=1)
            L = rng.integers(0, 2, (4, 6), dtype=np.uint8)
            p = rng.uniform(0.01, 0.4, 6)
            graph = build_graph(H, L, p)
            D = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
            e = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
            syn = e @ H.T % 2
            label = (e @ L.T % 2) @ (1 << np.arange(4))
            probability = np.prod(np.where(e, p, 1 - p), axis=1)
            pred, cp = decode_union(graph, D, lam=2, lam2=2, gap=1000,
                                    capacity=32, return_classp=True)
            for i, target in enumerate(D):
                valid = (syn == target).all(1)
                exact = np.bincount(label[valid], weights=probability[valid], minlength=16)
                exact /= exact.sum()
                np.testing.assert_allclose(cp[i], exact, atol=1e-6)
            _, repeated = decode_union(
                graph, D, schedules=((3, .8, .5),) * 3, lam=2, lam2=2,
                gap=1000, capacity=32, return_classp=True)
            np.testing.assert_allclose(cp, repeated, atol=1e-7)

    def test_hash_collision_compares_full_vectors(self):
        mask = (1 << 64) - 1
        start, mul = 0xCBF29CE484222325, 0x100000001B3
        ha, hb = (start * mul) & mask, ((start ^ 1) * mul) & mask
        a = np.array([0, 0], np.uint64)
        b = np.array([1, ha ^ hb], np.uint64)
        self.assertEqual(_hash(a), _hash(b))
        vectors = np.empty((8, 2), np.uint64)
        costs, logs, hashes = np.empty(8), np.empty(8, np.uint8), np.empty(8, np.uint64)
        used, _ = _insert(a, 1.0, np.uint8(1), vectors, costs, logs, hashes, 0)
        used, _ = _insert(b, 2.0, np.uint8(2), vectors, costs, logs, hashes, used)
        self.assertEqual(used, 2)
        used, _ = _insert(a, 1.0, np.uint8(1), vectors, costs, logs, hashes, used)
        self.assertEqual(used, 2)

    def test_zero_jitter_duplicates_do_not_multiply_probability(self):
        H = np.array([[1, 0, 1, 1], [0, 1, 1, 0]], np.uint8)
        L = np.eye(4, dtype=np.uint8)
        graph = build_graph(H, L, np.array([.1, .2, .3, .15]))
        D = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
        _, base = decode_union(graph, D, lam=2, lam2=2, capacity=64,
                                gap=1000, return_classp=True)
        _, extra = decode_union(graph, D, lam=2, lam2=2, capacity=64,
                                 gap=1000, return_classp=True,
                                 jitter_trials=8, jitter_strength=0)
        np.testing.assert_array_equal(base, extra)

    def test_global_order_three_recovers_all_three_variable_states(self):
        # Only two detectors: OSD pair-only lists may omit the three-error state.
        # The global pair index adds it regardless of the OSD information set.
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
        p = np.array([0.1, 0.2, 0.3])
        graph = build_graph(H, L, p)
        D = ((np.arange(4)[:, None] >> np.arange(2)) & 1).astype(np.uint8)
        _, cp = decode_union(graph, D, lam=1, lam2=1, capacity=32, gap=1000,
                              exact_order3=True, return_classp=True)
        e = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(np.uint8)
        syn = e @ H.T % 2
        label = (e @ L.T % 2) @ (1 << np.arange(4))
        prob = np.prod(np.where(e, p, 1 - p), axis=1)
        for i, target in enumerate(D):
            valid = (syn == target).all(1)
            exact = np.bincount(label[valid], weights=prob[valid], minlength=16)
            exact /= exact.sum()
            np.testing.assert_allclose(cp[i], exact, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
