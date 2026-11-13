"""Contract/determinism tests for the isolated hand-in entry point."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numba
import numpy as np
import stim

BP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BP.parent))


class TestCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(4)
        spec = importlib.util.spec_from_file_location(
            "isolated_bposd_candidate", BP / "candidate" / "decoder.py")
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_repeated_and_empty_batches(self):
        repo = BP.parent.parent
        for basis in "XZ":
            circuit = stim.Circuit.from_file(str(repo / "data" / "train" / f"{basis}_circuit.stim"))
            D = circuit.compile_detector_sampler(seed=391002).sample(120)
            repeated = np.concatenate([D, D[::-1], D[:20]])
            pred = self.mod.decode(circuit, repeated)
            np.testing.assert_array_equal(pred, self.mod.decode(circuit, repeated))
            np.testing.assert_array_equal(pred[:120], pred[120:240][::-1])
            np.testing.assert_array_equal(pred[:20], pred[240:])
            self.assertEqual(pred.dtype, np.uint8)
            self.assertTrue(np.isin(pred, (0, 1)).all())
            self.assertEqual(self.mod.decode(circuit, D[:0]).shape, (0, 4))

    def test_reference_routing_when_using_reference_configuration(self):
        from bp_osd_decoder.decoder_bposd import decode_bposd
        old = os.environ.get("BPOSD_CONFIG")
        try:
            os.environ["BPOSD_CONFIG"] = "reference.json"
            spec = importlib.util.spec_from_file_location(
                "isolated_bposd_reference", BP / "candidate" / "decoder.py")
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
        finally:
            if old is None:
                os.environ.pop("BPOSD_CONFIG", None)
            else:
                os.environ["BPOSD_CONFIG"] = old
        repo = BP.parent.parent
        for basis in "XZ":
            circuit = stim.Circuit.from_file(str(repo / "data" / "train" / f"{basis}_circuit.stim"))
            D = circuit.compile_detector_sampler(seed=391003).sample(300)
            np.testing.assert_array_equal(
                decode_bposd(circuit, D), reference.decode(circuit, D))

    def test_promotion_routing_matches_unoptimized_backend(self):
        from bp_osd_decoder.decoder_bposd import lookup
        repo = BP.parent.parent
        for basis in "XZ":
            circuit = stim.Circuit.from_file(str(repo / "data" / "train" / f"{basis}_circuit.stim"))
            D = circuit.compile_detector_sampler(seed=391004).sample(200)
            expected, found = lookup(circuit, D)
            miss = ~(found & (D.sum(1) <= self.mod.CONFIG["lut_cutoff"]))
            import hashlib
            sha = hashlib.sha256(str(circuit).encode()).hexdigest()
            graph = self.mod.STATE[sha][0]
            settings = self.mod.CONFIG.get("basis_parameters", {}).get(
                basis, self.mod.CONFIG["parameters"])
            expected[miss] = self.mod._compute(graph, D[miss], settings)
            np.testing.assert_array_equal(expected, self.mod.decode(circuit, D))


if __name__ == "__main__":
    unittest.main()
