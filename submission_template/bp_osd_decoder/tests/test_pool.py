"""Randomized replacements, ties, duplicates and collisions match the linear pool."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from bp_osd_decoder.pool import create, insert
from bp_osd_decoder.union import _hash, _insert


class TestPool(unittest.TestCase):
    def test_identical_updates(self):
        rng = np.random.default_rng(9153)
        for capacity in (1, 8, 64):
            vv = [np.empty((capacity, 3), np.uint64) for _ in range(2)]
            cc = [np.empty(capacity) for _ in range(2)]
            ll = [np.empty(capacity, np.uint8) for _ in range(2)]
            hh = [np.empty(capacity, np.uint64) for _ in range(2)]
            index = create(capacity)
            used = [0, 0]
            for step in range(2000):
                vector = rng.integers(0, 1 << 63, 3, dtype=np.uint64)
                cost = float(rng.integers(-step // 20, 20))  # ties and many replacements
                logical = np.uint8(rng.integers(0, 16))
                if used[0] and step % 5 == 0:
                    slot = rng.integers(used[0])
                    vector = vv[0][slot].copy()
                    cost, logical = cc[0][slot], ll[0][slot]
                # Force collision buckets on some updates; full equality still wins.
                h = np.uint64(17) if step % 13 == 0 else np.uint64(_hash(vector))
                # Linear implementation computes the real hash, so use an equivalent
                # reference loop with the same externally supplied hash here.
                a = used[0]
                duplicate = next((i for i in range(a) if hh[0][i] == h
                                  and np.array_equal(vv[0][i], vector)), None)
                if duplicate is not None:
                    drop0 = 0
                else:
                    slot = a if a < capacity else int(np.argmax(cc[0]))
                    if a < capacity or cost < cc[0][slot]:
                        vv[0][slot], cc[0][slot], ll[0][slot], hh[0][slot] = vector, cost, logical, h
                        used[0] = min(capacity, a + 1)
                    drop0 = int(a >= capacity)
                used[1], drop1 = insert(vector, cost, logical, vv[1], cc[1], ll[1], hh[1],
                                       used[1], h, index)
                self.assertEqual(used[0], used[1])
                self.assertEqual(drop0, drop1)
                for old, new in zip((vv[0], cc[0], ll[0], hh[0]), (vv[1], cc[1], ll[1], hh[1])):
                    np.testing.assert_array_equal(old[:used[0]], new[:used[1]])


if __name__ == "__main__":
    unittest.main()
