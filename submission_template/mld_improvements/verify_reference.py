"""Regression against the pre-study kernel from git, without changing files."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numba
import numpy as np
import stim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from hackathon_2026 import problem_info
import mld

REVISION = "962f067efdfe5ffd7dd8f43f1e4faf99ee42ac1e"


def main():
    numba.set_num_threads(4)
    source = subprocess.check_output(
        ["git", "show", f"{REVISION}:submission_template/mld.py"], cwd=ROOT).decode()
    reference = types.ModuleType("pre_study_mld")
    sys.modules[reference.__name__] = reference
    # No file writes. Numba's disk cache requires a physical source file.
    exec(compile(source.replace("cache=True", "cache=False"), "<pre-study-mld>", "exec"),
         reference.__dict__)
    records = []
    for basis in "XZ":
        circuit = stim.Circuit.from_file(str(ROOT / f"data/train/{basis}_circuit.stim"))
        info = problem_info(circuit)
        D = circuit.compile_detector_sampler(seed=976001 + (basis == "Z")).sample(1000)
        for reverse in (False, True):
            H, events = (info["H"][::-1].copy(), D[:, ::-1]) if reverse else (info["H"], D)
            old = reference.mld_logw(reference.build_mld(H, info["L"], info["p"]), events,
                                     eps=1e-4, max_states=1024, compact_every=32)
            new = mld.mld_logw(mld.build_mld(H, info["L"], info["p"]), events,
                               eps=1e-4, max_states=1024, compact_every=32)
            np.testing.assert_array_equal(old, new)
            records.append(dict(basis=basis, reverse=reverse, shots=len(D), logweight_changes=0))
    result = dict(revision=REVISION, original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  current_sha256=hashlib.sha256((HERE.parent / "mld.py").read_bytes()).hexdigest(),
                  records=records)
    (HERE / "results/reference_equivalence.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
