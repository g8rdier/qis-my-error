"""Paired comparison of two predeclared, already frozen audit configurations.

Descriptive evaluation only, not a parameter-selection loop. Per-shot averaging
retains dependence between the four logical bits. Cached predictions keep this
comparison independent of subsequent speed-only implementation changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bp_osd_decoder.experiments.search import CACHE
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--challenger", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    index = {(r["basis"], r["fold"], r["config"]): r for r in rows}
    all_delta, all_a, all_b, checks = [], [], [], []
    corrected = introduced = 0
    for basis, fold in sorted({(r["basis"], r["fold"]) for r in rows}):
        a, b = index[basis, fold, args.reference], index[basis, fold, args.challenger]
        assert a["seed"] == b["seed"] and a["shots"] == b["shots"]
        ident = a["identity"]
        sample = CACHE / f"sample_{ident['circuit'][:12]}_{a['seed']}_{a['shots']}.npz"
        with np.load(sample) as z:
            truth = z["O"]
        preds = []
        for row in (a, b):
            sha = hashlib.sha256(json.dumps(row["identity"], sort_keys=True).encode()).hexdigest()
            with np.load(CACHE / f"prediction_{sha}.npz") as z:
                preds.append(z["pred"])
        error_a, error_b = preds[0] != truth, preds[1] != truth
        delta = error_b.mean(1) - error_a.mean(1)
        corrected += int((error_a & ~error_b).sum())
        introduced += int((~error_a & error_b).sum())
        all_delta.append(delta)
        all_a.append(error_a.mean(1))
        all_b.append(error_b.mean(1))
        checks.append(dict(basis=basis, seed=a["seed"], delta=float(delta.mean())))
    delta = np.concatenate(all_delta)
    mean = float(delta.mean())
    # Stratified paired variance: fixed equal basis/seed weights.
    se = float(np.sqrt(sum(d.var(ddof=1) / len(d) for d in all_delta)) / len(all_delta))
    record = dict(reference=args.reference, challenger=args.challenger,
                  shots=len(delta), reference_ler=float(np.concatenate(all_a).mean()),
                  challenger_ler=float(np.concatenate(all_b).mean()), delta=mean,
                  paired_95ci=[mean - 1.96 * se, mean + 1.96 * se],
                  corrected_bits=corrected, introduced_bits=introduced,
                  net_corrected_bits=corrected - introduced, checks=checks)
    path = args.log.with_name(
        f"{args.log.stem}_paired_{args.challenger}_vs_{args.reference}.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
