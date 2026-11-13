"""Run with python -m unittest discover -s submission_template/bp_osd_decoder/tests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numba
import numpy as np
import stim
from hackathon_2026 import problem_info

from bp_osd_decoder import bp_osd as reference
from bp_osd_decoder import enhanced


class TestEnhanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(4)

    def test_triples_against_exhaustive_exact_marginals(self):
        """Rank 4, seven variables: at most three nonbasis flips = ALL solutions."""
        rng = np.random.default_rng(923)
        for _ in range(12):
            H = np.concatenate([np.eye(4, dtype=np.uint8),
                                rng.integers(0, 2, (4, 3), dtype=np.uint8)], 1)
            L = rng.integers(0, 2, (4, 7), dtype=np.uint8)
            p = rng.uniform(0.01, 0.4, 7)
            graph = enhanced.build_graph(H, L, p)
            errors = ((np.arange(128)[:, None] >> np.arange(7)) & 1).astype(np.uint8)
            syndromes = (errors @ H.T) % 2
            outcomes = ((errors @ L.T) % 2) @ (1 << np.arange(4))
            mass = np.prod(np.where(errors, p, 1 - p), axis=1)
            for target in (0, 5, 15):
                D = ((target >> np.arange(4)) & 1).astype(np.uint8)
                valid = (syndromes == D).all(1)
                exact = np.bincount(outcomes[valid], weights=mass[valid], minlength=16)
                exact /= exact.sum()
                for order in (0, 1):
                    cp = np.zeros(16, dtype=np.float32)
                    enhanced._osd(
                        rng.normal(size=7).astype(np.float32), graph["llr0"],
                        graph["cols"], graph["logical"], np.uint64(target),
                        np.uint64(0), 4, 3, 3, 2, cp, 1.0, 3, order)
                    np.testing.assert_allclose(cp, exact, atol=2e-6)

    def test_order_zero_extension_matches_reference(self):
        repo = Path(__file__).resolve().parents[3]
        for b in "XZ":
            circuit = stim.Circuit.from_file(str(repo / "data" / "train" / f"{b}_circuit.stim"))
            graph = reference.build_graph(**problem_info(circuit))
            D = circuit.compile_detector_sampler(seed=310030).sample(100)
            p0, cp0 = reference.decode_bp_osd(graph, D, return_classp=True)
            p1, cp1 = enhanced.decode_enhanced(graph, D, return_classp=True)
            np.testing.assert_array_equal(p0, p1)
            np.testing.assert_allclose(cp0, cp1, rtol=0, atol=0)

    def test_ctz_every_bit(self):
        for i in range(64):
            self.assertEqual(enhanced._ctz(np.uint64(1 << i)), i)


if __name__ == "__main__":
    unittest.main()
