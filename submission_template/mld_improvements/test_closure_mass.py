"""Absolute closure weights must agree with exhaustive physical likelihoods."""
from pathlib import Path
import sys
import unittest

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closure_decoder.bp_osd import make_graph, solve
from mld_improvements.closure_mass import closure_logw


class TestClosureMass(unittest.TestCase):
    def test_absolute_scale_and_no_duplicate_mass(self):
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        H = np.zeros((70, 4), np.uint8)
        H[0] = [1, 0, 1, 0]
        H[64] = [0, 1, 0, 1]
        L = np.eye(4, dtype=np.uint8)
        p = np.array([0.03, 0.05, 0.07, 0.09])
        E = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
        D, inv = np.unique((E @ H.T) % 2, axis=0, return_inverse=True)
        exact = np.zeros((len(D), 16))
        np.add.at(exact, (inv, np.arange(16)),
                  np.prod(np.where(E, p / (1 - p), 1), axis=1))
        graph = make_graph(H, L, p)
        params = dict(schedules=[[3, 0.8, 0.1]], lam=4, lam2=4, capacity=64,
                      gap=100, closure_seeds=2)
        actual = np.exp(closure_logw(graph, D, params, p))
        np.testing.assert_allclose(actual, exact, atol=1e-13, rtol=1e-12)
        # Identical proposal repetitions must not count a vector twice.
        params["schedules"] *= 2
        np.testing.assert_allclose(np.exp(closure_logw(graph, D, params, p)), exact,
                                   atol=1e-13, rtol=1e-12)
        _, cp = solve(graph, D, params, return_probabilities=True)
        np.testing.assert_allclose(cp, actual / actual.sum(1, keepdims=True), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
