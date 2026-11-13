from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from bp_osd_decoder.enhanced import build_graph
from bp_osd_decoder.pair_index import build
from bp_osd_decoder.experiments.low_weight_bound import compute


class TestLowerBound(unittest.TestCase):
    def test_exact_restricted_probability_not_conditional_risk(self):
        rng = np.random.default_rng(7453)
        H = ((np.arange(1, 7)[None, :] >> np.arange(3)[:, None]) & 1).astype(np.uint8)
        L = rng.integers(0, 2, (4, 6), dtype=np.uint8)
        p = rng.uniform(.02, .2, 6)
        graph = build_graph(H, L, p)
        bound, _, _ = compute(graph["cols"], graph["logical"], p, *build(graph["cols"]))
        errors = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
        syndromes = errors @ H.T % 2
        syndromes = syndromes @ (1 << np.arange(3))
        outcomes = errors @ L.T % 2
        mass = np.prod(np.where(errors, p, 1 - p), axis=1)
        exact = np.zeros(4)
        full = np.zeros(4)
        for d in range(8):
            for bit in range(4):
                sums, all_sums = np.zeros(2), np.zeros(2)
                for i in range(len(errors)):
                    if syndromes[i] == d:
                        all_sums[outcomes[i, bit]] += mass[i]
                        if errors[i].sum() <= 2:
                            sums[outcomes[i, bit]] += mass[i]
                exact[bit] += sums.min()
                full[bit] += all_sums.min()
        np.testing.assert_allclose(bound, exact, atol=1e-12)
        self.assertTrue((bound <= full + 1e-12).all())


if __name__ == "__main__":
    unittest.main()
