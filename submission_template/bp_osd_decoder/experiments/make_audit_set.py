"""Export 100k/basis from the sealed holdout cache into run_local's CSV format.

Run ONLY after freezing a finalist and running search.py --split holdout
--shots 50000 --folds 3. This exports already evaluated held-out samples; it
does not generate a different distribution or touch Kaggle test data.
Refuses to overwrite an existing output folder.
"""
import csv
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bp_osd_decoder.experiments.search import CACHE, REPO
import numpy as np
import stim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-base", type=int, default=720100)
    parser.add_argument("--out", default="data/bposd_audit")
    args = parser.parse_args()
    out = (REPO / args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    # Check all inputs before making any output files.
    inputs = {}
    for basis in "XZ":
        circuit = stim.Circuit.from_file(str(REPO / "data" / "train" / f"{basis}_circuit.stim"))
        import hashlib
        sha = hashlib.sha256(str(circuit).encode()).hexdigest()
        paths = [CACHE / f"sample_{sha[:12]}_{args.seed_base + 2*f + (basis == 'Z')}_50000.npz"
                 for f in range(3)]
        if not all(p.exists() for p in paths):
            raise SystemExit("sealed holdout caches missing; evaluate frozen finalist first")
        inputs[basis] = circuit, paths
    out.mkdir()
    with (out / f"{out.name}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "basis"] + [f"D{i}" for i in range(70)]
                        + [f"L{i}" for i in range(4)])
        for basis in "XZ":
            circuit, paths = inputs[basis]
            circuit.to_file(str(out / f"{basis}_circuit.stim"))
            idx = 0
            for fold, path in enumerate(paths):
                n = 33334 if fold == 0 else 33333
                with np.load(path) as z:
                    for row, label in zip(z["D"][:n], z["O"][:n]):
                        writer.writerow([f"audit_{basis}_{idx:06}", basis]
                                        + row.astype(int).tolist()
                                        + label.astype(int).tolist())
                        idx += 1
    (out / "args.json").write_text(json.dumps(dict(
        role="final independent holdout; NOT training or parameter tuning",
        source_seeds=list(range(args.seed_base, args.seed_base + 6)),
        shots_per_basis=100000, source_shots_per_seed=50000), indent=2))
    print(out)


if __name__ == "__main__":
    main()
