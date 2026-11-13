"""Check exported entry points against audit caches, without reading labels."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numba
import numpy as np
import stim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from mld import marginals_logw


def main():
    numba.set_num_threads(4)
    results = []
    old = os.environ.get("MLD_IMPROVEMENTS_CONFIG")
    try:
        for model in ("candidate", "accuracy"):
            os.environ["MLD_IMPROVEMENTS_CONFIG"] = str(HERE / f"{model}.json")
            spec = importlib.util.spec_from_file_location(f"verify_mld_{model}", HERE / "decoder.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for family in range(3):
                for offset, basis in enumerate("XZ"):
                    circuit = stim.Circuit.from_file(str(ROOT / f"data/train/{basis}_circuit.stim"))
                    events = circuit.compile_detector_sampler(seed=954100 + 2 * family + offset).sample(50000)
                    D = events[:512]
                    names = ["hybrid075_tight"] if model == "candidate" else ["reference", "shuffle_checks"]
                    logw = None
                    for name in names:
                        with np.load(HERE / "cache" / f"audit_954100_50000_{basis}_{family}_{name}.npz") as z:
                            values = z["logw"][:len(D)]
                            logw = values if logw is None else np.maximum(logw, values)
                    expected = (marginals_logw(logw) > 0.5).astype(np.uint8)
                    actual = module.decode(circuit, D)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(module.decode(circuit, D), actual)
                    self_empty = module.decode(circuit, D[:0])
                    assert self_empty.shape == (0, 4) and self_empty.dtype == np.uint8
                    results.append(dict(model=model, basis=basis, family=family,
                                        shots=len(D), changed_bits=0))
        os.environ.pop("MLD_IMPROVEMENTS_CONFIG", None)
        spec = importlib.util.spec_from_file_location("verify_mld_default", HERE / "decoder.py")
        default = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(default)
        assert default.COMPONENTS == json.loads((HERE / "accuracy.json").read_text())["components"]
    finally:
        if old is None:
            os.environ.pop("MLD_IMPROVEMENTS_CONFIG", None)
        else:
            os.environ["MLD_IMPROVEMENTS_CONFIG"] = old
    result = dict(records=results, hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in [HERE / "decoder.py", HERE / "runtime.py",
                                                    HERE / "candidate.json", HERE / "accuracy.json"]})
    (HERE / "results/candidate_equivalence.json").write_text(json.dumps(result, indent=2))
    print("Verified", sum(r["shots"] for r in results), "exported shot predictions; zero changed bits.")


if __name__ == "__main__":
    main()
