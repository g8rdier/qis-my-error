"""Exhaustive small-model checks, padded to the kernel's 70-detector format."""
from pathlib import Path
import sys
import unittest

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mld import build_mld, marginals_logw, mld_logw
from mld_improvements.runtime import build_tables, infer, unique_events


class TestMLD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        rng = np.random.default_rng(42010)
        cls.H = np.zeros((70, 9), np.uint8)
        # Include detectors across the uint64 boundary and an invisible fault.
        cls.H[[0, 2, 63, 64, 69], :] = rng.integers(0, 2, (5, 9), dtype=np.uint8)
        cls.H[:, 0] = 0
        cls.L = rng.integers(0, 2, (4, 9), dtype=np.uint8)
        cls.p = rng.uniform(0.02, 0.2, 9)
        errors = ((np.arange(1 << 9)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
        syndromes = (errors @ cls.H.T) % 2
        cls.D, inverse = np.unique(syndromes, axis=0, return_inverse=True)
        logical = ((errors @ cls.L.T) % 2) @ (1 << np.arange(4))
        weight = np.prod(np.where(errors, cls.p / (1 - cls.p), 1), axis=1)
        cls.exact = np.zeros((len(cls.D), 16))
        np.add.at(cls.exact, (inverse, logical), weight)

    def test_exact_posteriors_for_all_heuristics_and_orders(self):
        for heuristic in ("max", "parity", "hybrid"):
            for order in ("span", "probability", "span_probability", "last"):
                for reverse in (False, True):
                    with self.subTest(heuristic=heuristic, order=order, reverse=reverse):
                        H = self.H[::-1].copy() if reverse else self.H
                        D = self.D[:, ::-1] if reverse else self.D
                        graph = build_mld(H, self.L, self.p, heuristic=heuristic,
                                          mechanism_order=order)
                        actual = np.exp(mld_logw(graph, D, eps=1e-100, max_states=1024))
                        np.testing.assert_allclose(actual, self.exact, atol=1e-13, rtol=2e-12)

    def test_pruned_masses_are_lower_bounds(self):
        for heuristic in ("max", "parity", "hybrid"):
            for merge_factor in (1.0, 32.0, 1e100):
                graph = build_mld(self.H, self.L, self.p, heuristic=heuristic,
                                  merge_factor=merge_factor)
                actual = np.exp(mld_logw(graph, self.D, eps=0.1, max_states=8))
                self.assertTrue(np.all(actual <= self.exact + 1e-12))

    def test_empty_failed_and_repeated_batches(self):
        graph = build_mld(self.H, self.L, self.p)
        self.assertEqual(mld_logw(graph, self.D[:0]).shape, (0, 16))
        D = np.repeat(self.D[:1], 3, axis=0)
        first = mld_logw(graph, D)
        np.testing.assert_array_equal(first, mld_logw(graph, D))
        np.testing.assert_array_equal(first, np.repeat(first[:1], 3, axis=0))
        impossible = np.zeros((1, 70), bool)
        impossible[0, 1] = 1
        np.testing.assert_array_equal(marginals_logw(mld_logw(graph, impossible)),
                                      np.full((1, 4), 0.5))

    def test_defaults_unchanged(self):
        a = build_mld(self.H, self.L, self.p)
        b = build_mld(self.H, self.L, self.p, heuristic="max", mechanism_order="span")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_runtime_routing_and_deduplication(self):
        graph = build_mld(self.H, self.L, self.p)
        graphs = (graph, build_mld(self.H[::-1].copy(), self.L, self.p))
        config = dict(compact=32, tiers=[[0, 1e-10, 1024, 0], [1, 1e-10, 1024, 0]])
        D = np.concatenate([self.D, self.D[:5], self.D[:5]])
        np.random.default_rng(341).shuffle(D)
        direct, counts = infer(graphs, D, config)
        unique, inverse = unique_events(D)
        _, first = np.unique(D, axis=0, return_index=True)
        np.testing.assert_array_equal(unique, D[np.sort(first)])
        dedup, _ = infer(graphs, unique, config)
        np.testing.assert_array_equal(direct, dedup[inverse])
        self.assertEqual(counts[0], len(D))
        self.assertEqual(infer(graphs, D[:0], config)[0].shape, (0, 16))
        self.assertEqual(len(unique_events(D[:0])[0]), 0)

    def test_within_round_detector_orders(self):
        for mode in ("time", "reverse_checks", "shuffle_checks"):
            config = dict(heuristic="max", hexp=0.5, order="span", compact=32,
                          detector_order=mode, tiers=[[0, 1e-100, 1024, 0]])
            graphs = build_tables(dict(H=self.H, L=self.L, p=self.p), config)
            logw, _ = infer(graphs, self.D, config)
            np.testing.assert_allclose(np.exp(logw), self.exact, atol=1e-13, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
