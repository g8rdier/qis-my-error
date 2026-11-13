"""Test probability ensembles from existing development predictions, no redecoding.

This averages separately normalized truncated posteriors. It is a heuristic,
not exact union-of-candidates marginalization. Models and seeds are recorded.
Only development logs may be supplied, never the final holdout for tuning.
"""
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
    parser.add_argument("--name", default="ensembles")
    args = parser.parse_args()
    assert all("development" in p.name for p in args.logs), "do not tune on holdout"
    raw = [json.loads(line) for path in args.logs
           for line in path.read_text().splitlines()]
    index = {(r["basis"], r["fold"], r["config"]): r for r in raw}
    bank = {
        "iter3_5_equal": (["baseline", "iterations_5"], [0.5, 0.5]),
        "iter3_5_8_equal": (["baseline", "iterations_5", "iterations_8"], [1/3] * 3),
        "iter3_5_alpha_equal": (["baseline", "iterations_5", "alpha_0.9"], [1/3] * 3),
    }
    if any(r["config"] == "triple_bp40" for r in raw):
        bank["iter5_triple_equal"] = (["iterations_5", "triple_bp40"], [0.5, 0.5])
    rows = []
    for basis, fold in sorted({(r["basis"], r["fold"]) for r in raw}):
        ref = index[basis, fold, "baseline"]
        # Samples have the circuit hash prefix, seed, and shot count in the filename.
        ident = ref["identity"]
        sample = CACHE / f"sample_{ident['circuit'][:12]}_{ref['seed']}_{ref['shots']}.npz"
        with np.load(sample) as z:
            truth = z["O"]

        def prediction(name):
            row = index[basis, fold, name]
            h = hashlib.sha256(json.dumps(row["identity"], sort_keys=True).encode()).hexdigest()
            with np.load(CACHE / f"prediction_{h}.npz") as z:
                return z["marg"].copy(), float(z["seconds"])

        base, basetime = prediction("baseline")
        refloss = ((base > 0.5) != truth).mean(1)
        for name, (members, weights) in bank.items():
            ps = [prediction(member) for member in members]
            marg = sum(w * p[0] for w, p in zip(weights, ps))
            pred = marg > 0.5
            loss = (pred != truth).mean(1)
            delta = loss - refloss
            rows.append(dict(
                config=name, basis=basis, fold=fold, seed=ref["seed"],
                shots=len(truth), ler=float(loss.mean()),
                baseline_ler=float(refloss.mean()), delta_ler=float(delta.mean()),
                delta_se=float(delta.std(ddof=1) / np.sqrt(len(truth))),
                seconds=sum(p[1] for p in ps), members=members, weights=weights))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.name}_development.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = report(rows, path.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
