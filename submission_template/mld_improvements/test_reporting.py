"""Check pooled paired intervals against direct shot-level calculation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mld_improvements.final_report import pool


class TestReporting(unittest.TestCase):
    def test_pooled_shot_variance(self):
        rng = np.random.default_rng(724)
        a = rng.integers(-4, 5, 80) / 4
        b = rng.integers(-4, 5, 120) / 4

        def record(delta):
            mean = float(delta.mean())
            radius = float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta)))
            return dict(delta=mean, ci95=[mean - radius, mean + radius],
                        net_bits_saved=int(-4 * delta.sum()))

        actual = pool(record(a), len(a), record(b), len(b))
        expected = record(np.concatenate([a, b]))
        self.assertAlmostEqual(actual["delta"], expected["delta"])
        np.testing.assert_allclose(actual["ci95"], expected["ci95"], atol=1e-14)
        self.assertEqual(actual["net_bits_saved"], expected["net_bits_saved"])


if __name__ == "__main__":
    unittest.main()
