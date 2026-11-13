from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numba
import numpy as np
import stim
from hackathon_2026 import problem_info
from bp_osd_decoder.enhanced import build_graph
from bp_osd_decoder.adaptive import decode_adaptive
from bp_osd_decoder.fused import decode_fused


class TestFused(unittest.TestCase):
    def test_frozen_balanced_kernel(self):
        numba.set_num_threads(4)
        repo = Path(__file__).resolve().parents[3]
        for b in "XZ":
            c = stim.Circuit.from_file(str(repo / "data" / "train" / f"{b}_circuit.stim"))
            graph = build_graph(**problem_info(c))
            D = c.compile_detector_sampler(seed=319005).sample(400)
            settings = dict(initial={"max_iter": 5, "refine_thr": 0}, route_threshold=.05)
            p0, cp0 = decode_adaptive(graph, D, return_classp=True, **settings)
            p1, cp1 = decode_fused(graph, D, return_classp=True, **settings)
            np.testing.assert_array_equal(p0, p1)
            np.testing.assert_array_equal(cp0, cp1)


if __name__ == "__main__":
    unittest.main()
