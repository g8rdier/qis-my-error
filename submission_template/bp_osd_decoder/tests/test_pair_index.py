"""The precomputed index preserves full 70-bit syndromes and collision chains."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from bp_osd_decoder.pair_index import build, find


class TestPairIndex(unittest.TestCase):
    def test_all_pairs_and_both_words(self):
        rng = np.random.default_rng(1103)
        cols = np.zeros((20, 2), dtype=np.uint64)
        cols[:, 0] = rng.integers(0, 1 << 63, len(cols), dtype=np.uint64)
        cols[:, 1] = rng.integers(0, 64, len(cols), dtype=np.uint64)
        # Two repeated column effects deliberately create equal-syndrome pairs.
        cols[17], cols[18] = cols[2], cols[3]
        heads, nxt, ii, jj = build(cols)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                lo, hi = cols[a, 0] ^ cols[b, 0], cols[a, 1] ^ cols[b, 1]
                k = find(lo, hi, cols, heads, ii, jj)
                chain = set()
                while k >= 0:
                    chain.add((int(ii[k]), int(jj[k])))
                    k = nxt[k]
                expected = {(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))
                            if (cols[i, 0] ^ cols[j, 0]) == lo
                            and (cols[i, 1] ^ cols[j, 1]) == hi}
                self.assertEqual(chain, expected)


if __name__ == "__main__":
    unittest.main()
