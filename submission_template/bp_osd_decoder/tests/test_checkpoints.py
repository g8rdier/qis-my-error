"""Shared BP trajectories must reproduce independently run ranks bit for bit."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import stim
from hackathon_2026 import problem_info
from bp_osd_decoder.enhanced import build_graph
from bp_osd_decoder.union import _rank_post
from bp_osd_decoder.bp_checkpoints import posterior_checkpoints


class TestCheckpoints(unittest.TestCase):
    def test_all_snapshots_and_separate_hyperparameter_groups(self):
        repo = Path(__file__).resolve().parents[3]
        schedules = np.array([[3, .8, .5], [5, .8, .5], [8, .8, .5],
                              [3, 1., .3], [5, 1., .3], [0, .8, .5]])
        for basis in "XZ":
            c = stim.Circuit.from_file(str(repo / "data" / "train" / f"{basis}_circuit.stim"))
            graph = build_graph(**problem_info(c))
            for D in c.compile_detector_sampler(seed=319004).sample(50):
                snapshots = posterior_checkpoints(
                    D.astype(np.uint8), graph["ev"], graph["check_ptr"],
                    graph["llr0"], schedules)
                for i, (iterations, alpha, damping) in enumerate(schedules):
                    original = _rank_post(
                        D.astype(np.uint8), graph["ec"], graph["ev"],
                        graph["check_ptr"], graph["llr0"],
                        int(iterations), alpha, damping)
                    np.testing.assert_array_equal(original, snapshots[i])


if __name__ == "__main__":
    unittest.main()
