"""Build artifacts/basis_map.npz: fingerprints to identify X vs Z circuit in decode().

decode() receives only (circuit, detector_events) — no basis flag. This stores:
  sha[basis]  = sha256 of str(circuit)  (fast path, ~ms)
  ncol[basis] = # error mechanisms from problem_info (fallback path)
Run once from repo root:
  .venv/Scripts/python.exe submission_template/experiments/build_basis_map.py
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import stim

from models.common import REPO_ROOT, ensure_artifacts_dir

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from hackathon_2026 import problem_info


def main():
    ensure_artifacts_dir()
    shas, ncols = {}, {}
    for basis in "XZ":
        c = stim.Circuit.from_file(os.path.join(REPO_ROOT, "data", "train", f"{basis}_circuit.stim"))
        shas[basis] = hashlib.sha256(str(c).encode()).hexdigest()
        ncols[basis] = problem_info(c)["H"].shape[1]
        print(basis, "sha", shas[basis][:16], "ncol", ncols[basis])
    np.savez_compressed(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "artifacts", "basis_map.npz"),
        **{f"sha_{b}": np.array(s) for b, s in shas.items()},
        **{f"ncol_{b}": np.array(n) for b, n in ncols.items()},
    )
    print("saved basis_map.npz")


if __name__ == "__main__":
    main()
