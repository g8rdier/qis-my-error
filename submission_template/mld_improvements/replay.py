"""Evaluate predeclared outcome-weight combinations from development caches."""
import argparse
import json
from pathlib import Path

import numpy as np

from search import HERE, marginals_logw, score


PAIRS = [
    ("reference", "probability"),
    ("reference", "span_probability"),
    ("reference", "max065"),
    ("reference", "hybrid05"),
    ("reference", "hybrid075"),
    ("reference", "parity075"),
    ("reference", "merge32"),
    ("reference", "merge_all"),
    ("tight", "probability"),
    ("tight", "span_probability"),
    ("reference", "last_detector"),
    ("reference", "reverse_checks"),
    ("reference", "shuffle_checks"),
    ("reference", "parity075_tight"),
    ("reference", "hybrid075_tight"),
    ("tight", "last_detector"),
    ("tight", "reverse_checks"),
    ("tight", "shuffle_checks"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="development_943100_3000")
    args = parser.parse_args()
    metadata = json.loads((HERE / "results" / f"{args.prefix}.metadata.json").read_text())
    families = metadata["arguments"]["families"]
    summary = json.loads((HERE / "results" / f"{args.prefix}.summary.json").read_text())
    times = {r["name"]: r["seconds"] for r in summary}
    truth, reference = [], []
    outputs = {}
    for family in range(families):
        for basis in "XZ":
            cached = {}
            for name in metadata["configurations"]:
                path = HERE / "cache" / f"{args.prefix}_{basis}_{family}_{name}.npz"
                with np.load(path) as z:
                    cached[name] = {k: z[k] for k in z.files}
            labels = cached["reference"]["labels"]
            ref = cached["reference"]["probability"] > 0.5
            truth.append(labels)
            reference.append(ref)
            for a, b in PAIRS:
                if a not in cached or b not in cached:
                    continue
                for operation in ("max", "average"):
                    name = f"{a}+{b}/{operation}"
                    if operation == "max":
                        prob = marginals_logw(np.maximum(cached[a]["logw"], cached[b]["logw"]))
                    else:
                        prob = 0.5 * (cached[a]["probability"] + cached[b]["probability"])
                    outputs.setdefault(name, []).append(prob)
    labels, ref = np.concatenate(truth), np.concatenate(reference)
    report = []
    for name, probs in outputs.items():
        a, b = name.split("/")[0].split("+")
        row = score(name, np.concatenate(probs), ref, labels, times[a] + times[b])
        row["groups"] = [score(name, prob, r, y, 0) for prob, r, y in zip(probs, reference, truth)]
        report.append(row)
    report.sort(key=lambda r: r["ler"])
    path = HERE / "results" / f"{args.prefix}.replay.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2))
    for row in report:
        print(row["name"], f"{row['ler']:.8f}", row["paired"])


if __name__ == "__main__":
    main()
