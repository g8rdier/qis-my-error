"""Routing tests use only syndromes/probabilities, never labels."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numba
import numpy as np
from bp_osd_decoder.adaptive import BITS, decode_adaptive, run_refiner
from bp_osd_decoder.enhanced import build_graph, decode_enhanced


class TestAdaptive(unittest.TestCase):
    def test_route_mask_and_zero_routing(self):
        numba.set_num_threads(4)
        rng = np.random.default_rng(914)
        H = np.concatenate([np.eye(4, dtype=np.uint8),
                            rng.integers(0, 2, (4, 4), dtype=np.uint8)], 1)
        L = rng.integers(0, 2, (4, 8), dtype=np.uint8)
        graph = build_graph(H, L, rng.uniform(.02, .2, 8))
        D = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
        initial = {"max_iter": 5, "refine_thr": 0, "lam": 4, "lam2": 2}
        refiner = {"union": True, "lam": 4, "lam2": 4}
        pred, cp = decode_enhanced(graph, D, return_classp=True, **initial)
        no_route = decode_adaptive(graph, D, initial, refiner, route_threshold=.5)
        np.testing.assert_array_equal(no_route, pred)
        marg = cp @ BITS
        use = ((marg > .1) & (marg < .9)).any(1)
        expected = pred.copy()
        if use.any():
            expected[use] = run_refiner(graph, D[use], refiner)[0]
        actual = decode_adaptive(graph, D, initial, refiner, route_threshold=.1)
        np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
