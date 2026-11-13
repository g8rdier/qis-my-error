"""Choose uncertainty routing on DEVELOPMENT caches, not holdout labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bp_osd_decoder.experiments.search import CACHE, RESULTS, report
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--name", default="routes")
    args = parser.parse_args()
    assert all("development" in p.name for p in args.logs)
    raw = [json.loads(s) for p in args.logs for s in p.read_text().splitlines()]
    index = {(r["basis"], r["fold"], r["config"]): r for r in raw}
    rows = []
    for basis, fold in sorted({(r["basis"], r["fold"]) for r in raw}):
        ref = index[basis, fold, "baseline"]
        ident = ref["identity"]
        sample = CACHE / f"sample_{ident['circuit'][:12]}_{ref['seed']}_{ref['shots']}.npz"
        with np.load(sample) as z:
            O = z["O"]

        def load(name):
            row = index[basis, fold, name]
            key = hashlib.sha256(json.dumps(row["identity"], sort_keys=True).encode()).hexdigest()
            with np.load(CACHE / f"prediction_{key}.npz") as z:
                return z["pred"].copy(), z["marg"].copy(), float(z["seconds"])

        baseline, _, _ = load("baseline")
        initial, marg, t0 = load("iterations_5")
        refloss = (baseline != O).mean(1)
        for method in ("union358_pairs200", "triple5_bp60", "triple5_bp100"):
            pp, _, tr = load(method)
            for threshold in (0.05, 0.1, 0.2, 0.3, 0.4):
                route = ((marg > threshold) & (marg < 1 - threshold)).any(1)
                pred = initial.copy()
                pred[route] = pp[route]
                loss = (pred != O).mean(1)
                delta = loss - refloss
                rows.append(dict(
                    config=f"route_{method}_{threshold}", basis=basis, fold=fold,
                    seed=ref["seed"], ler=float(loss.mean()),
                    delta_ler=float(delta.mean()),
                    delta_se=float(delta.std(ddof=1) / np.sqrt(len(O))),
                    seconds=t0 + route.mean() * tr,
                    timing_note="approximation only; runtime depends on routed shot difficulty",
                    routed_fraction=float(route.mean()), initial="iterations_5",
                    refiner=method, threshold=threshold))
    path = RESULTS / f"{args.name}_development.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = report(rows, path.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
