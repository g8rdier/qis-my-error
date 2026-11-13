"""Aggregate the two frozen holdouts and official-shaped timing logs."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent


def pool(first, n1, second, n2):
    """Reconstruct pooled shot-difference variance from recorded normal CIs."""
    mean = (n1 * first["delta"] + n2 * second["delta"]) / (n1 + n2)
    variance = []
    for row, n in [(first, n1), (second, n2)]:
        radius = (row["ci95"][1] - row["ci95"][0]) / 2
        variance.append((radius / 1.96) ** 2 * n)
    var = ((n1 - 1) * variance[0] + (n2 - 1) * variance[1] +
           n1 * (first["delta"] - mean) ** 2 +
           n2 * (second["delta"] - mean) ** 2) / (n1 + n2 - 1)
    radius = 1.96 * np.sqrt(var / (n1 + n2))
    return dict(shots=n1 + n2, delta=mean, ci95=[mean - radius, mean + radius],
                net_bits_saved=first["net_bits_saved"] + second["net_bits_saved"])


def main():
    result_dir = HERE / "results"
    audited = json.loads((result_dir / "audit_954100_50000.summary.json").read_text())
    audit = {r["name"]: r for r in audited}
    secondary = json.loads((result_dir / "audit_954100_50000.secondary.json").read_text())
    benchmark = json.loads((result_dir / "benchmark_paired.json").read_text())
    fresh = benchmark["fresh"]
    timings = {}
    initial_timings = {}
    for name in ("reference", "candidate", "accuracy"):
        text = (result_dir / f"benchmark_fresh_{name}.txt").read_text()
        match = re.search(r"runtime \(median of (\d+)\): ([0-9.]+) s", text)
        timings[name] = dict(runs=int(match[1]), seconds=float(match[2]))
        initial_timings[name] = timings[name]
        if name != "reference":
            text = (result_dir / f"benchmark_ordered_fresh_{name}.txt").read_text()
            match = re.search(r"runtime \(median of (\d+)\): ([0-9.]+) s", text)
            timings[name] = dict(runs=int(match[1]), seconds=float(match[2]))
    result = dict(
        fresh=benchmark["fresh"], train=benchmark["train"], local_timings=timings,
        sorted_dedup_timings=initial_timings,
        pooled_candidate=pool(audit["hybrid075_tight"]["paired"], 300000,
                              fresh["candidate_paired"], 200000),
        pooled_accuracy=pool(secondary["paired"], 300000,
                             fresh["accuracy_paired"], 200000),
        versions_note="See experiment metadata and pinned parent requirements.txt",
        parameter_hashes={
            name: hashlib.sha256(json.dumps(json.loads((HERE / f"{name}.json").read_text())["components"],
                                           sort_keys=True).encode()).hexdigest()
            for name in ("candidate", "accuracy")},
        source_hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [HERE / "decoder.py", HERE / "runtime.py", HERE.parent / "mld.py"]})
    target = result_dir / "final_report.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
