"""Pair replacement recovers an exact toy posterior, preserving syndromes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from bp_osd_decoder.enhanced import build_graph
from bp_osd_decoder.pair_index import build
from bp_osd_decoder.pool import create, insert
from bp_osd_decoder.local_closure import expand, vector_hash
from bp_osd_decoder.union import decode_union


class TestLocalClosure(unittest.TestCase):
    def test_integrated_decoder_posterior(self):
        H = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], np.uint8)
        p = np.array([.1, .2, .3, .4])
        graph = build_graph(H, np.eye(4, dtype=np.uint8), p)
        _, cp = decode_union(
            graph, np.array([[1, 1]], np.uint8), lam=1, lam2=0,
            capacity=64, gap=1000, closure_seeds=8, return_classp=True)
        e = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
        valid = (e @ H.T % 2 == [1, 1]).all(1)
        probability = np.where(valid, np.prod(np.where(e, p, 1 - p), 1), 0)
        probability /= probability.sum()
        np.testing.assert_allclose(cp[0], probability, atol=1e-6)

    def test_exact_pair_alternatives(self):
        H = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], np.uint8)
        L = np.eye(4, dtype=np.uint8)
        p = np.array([.1, .2, .3, .4])
        graph = build_graph(H, L, p)
        heads, nxt, ii, jj = build(graph["cols"])
        for per_class in (False, True):
            vecs = np.empty((64, 1), np.uint64)
            costs = np.empty(64)
            logs, hashes = np.empty(64, np.uint8), np.empty(64, np.uint64)
            index = create(64)
            base = np.array([3], np.uint64)  # mechanisms 0 and 1
            used, _ = insert(
                base, float(graph["llr0"][0]) + float(graph["llr0"][1]),
                np.uint8(3), vecs, costs, logs, hashes, 0, vector_hash(base), index)
            used, _ = expand(
                graph["cols"], graph["logical"], graph["llr0"], heads, nxt, ii, jj,
                vecs, costs, logs, hashes, used, 1000., index, 32, 2, per_class)
            e = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8)
            valid = (e @ H.T % 2 == np.array([1, 1])).all(1)
            mass = np.prod(np.where(e, p, 1 - p), axis=1)
            exact = np.where(valid, mass, 0)
            exact /= exact.sum()
            found = np.zeros(16)
            for k in range(used):
                bits = ((int(vecs[k, 0]) >> np.arange(4)) & 1).astype(np.uint8)
                np.testing.assert_array_equal(bits @ H.T % 2, [1, 1])
                self.assertEqual(int(logs[k]), int(vecs[k, 0]))
                found[logs[k]] += np.exp(-costs[k])
            found /= found.sum()
            np.testing.assert_allclose(found, exact, atol=1e-6)

    def test_every_generated_vector_has_correct_cost_syndrome_and_logicals(self):
        rng = np.random.default_rng(2944)
        H = ((np.arange(1, 13)[None, :] >> np.arange(4)[:, None]) & 1).astype(np.uint8)
        L = rng.integers(0, 2, (4, 12), dtype=np.uint8)
        p = rng.uniform(.01, .3, 12)
        graph = build_graph(H, L, p)
        heads, nxt, ii, jj = build(graph["cols"])
        error = np.zeros(12, np.uint8)
        error[[0, 3, 7, 9]] = 1
        mask = int(error @ (1 << np.arange(12)))
        logical = int((L @ error % 2) @ (1 << np.arange(4)))
        vectors = np.empty((1024, 1), np.uint64)
        costs, logs, hashes = np.empty(1024), np.empty(1024, np.uint8), np.empty(1024, np.uint64)
        index = create(1024)
        base = np.array([mask], np.uint64)
        used, _ = insert(
            base, float(graph["llr0"] @ error), np.uint8(logical),
            vectors, costs, logs, hashes, 0, vector_hash(base), index)
        used, _ = expand(
            graph["cols"], graph["logical"], graph["llr0"], heads, nxt, ii, jj,
            vectors, costs, logs, hashes, used, 1000., index, 8, 2, True)
        self.assertGreater(used, 1)
        for k in range(used):
            e = ((int(vectors[k, 0]) >> np.arange(12)) & 1).astype(np.uint8)
            np.testing.assert_array_equal(H @ e % 2, H @ error % 2)
            self.assertEqual(int(logs[k]), int((L @ e % 2) @ (1 << np.arange(4))))
            self.assertAlmostEqual(costs[k], float(graph["llr0"].astype(float) @ e), places=5)


if __name__ == "__main__":
    unittest.main()
