"""Generate reproducible, circuit-fingerprinted graph artifacts before hand-in."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BP_DIR = HERE.parent
SUBMISSION = BP_DIR.parent
REPO = SUBMISSION.parent
sys.path.insert(0, str(SUBMISSION))

import numpy as np
import stim
from hackathon_2026 import problem_info
from bp_osd_decoder.enhanced import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuits", type=Path, default=REPO / "data" / "train")
    args = parser.parse_args()
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    lut_manifest = json.loads((BP_DIR / "lut" / "manifest.json").read_text())
    manifest = {}
    for basis in "XZ":
        path = args.circuits / f"{basis}_circuit.stim"
        c = stim.Circuit.from_file(str(path))
        sha = hashlib.sha256(str(c).encode()).hexdigest()
        info = problem_info(c)
        graph = build_graph(**info)
        file = f"graph_{basis}.npz"
        np.savez_compressed(out / file, **graph)
        try:
            source = str(path.relative_to(REPO))
        except ValueError:
            source = str(path)
        manifest[sha] = dict(
            basis=basis, file=file, lut_file=lut_manifest[sha]["file"],
            source_circuit=source,
            H_sha256=hashlib.sha256(info["H"].tobytes()).hexdigest(),
            L_sha256=hashlib.sha256(info["L"].tobytes()).hexdigest(),
            p_sha256=hashlib.sha256(info["p"].tobytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote circuit-fingerprinted graphs (no detector samples or labels).")


if __name__ == "__main__":
    main()
