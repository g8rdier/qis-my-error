"""Run the repository's actual run_local.py and retain its output/configuration.

Example:
  python .../benchmark.py --config config.json --problem-set data/bposd_audit
No substitute microbenchmark: this launches the unchanged official-shaped
runner in a fresh process, sets thread counts before numpy/numba import,
and records all five runs plus the exact chosen configuration/source hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BP = Path(__file__).resolve().parents[1]
SUBMISSION = BP.parent
REPO = SUBMISSION.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json",
                        choices=[p.name for p in (BP / "candidate").glob("*.json")])
    parser.add_argument("--problem-set", default="data/bposd_audit")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--label", default="", help="optional version label for result filenames")
    args = parser.parse_args()
    problem = (REPO / args.problem_set).resolve()
    if problem == (REPO / "data" / "test").resolve():
        raise SystemExit("this is a labelled evaluation tool, not a test-data submission tool")
    cache = BP / "experiments" / "cache"
    results = BP / "experiments" / "results"
    cache.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    tag = f"benchmark_{Path(args.config).stem}_{problem.name}_t{args.threads}"
    if args.label:
        tag += f"_{args.label}"
    env = dict(os.environ, BPOSD_CONFIG=args.config)
    cmd = [sys.executable, str(SUBMISSION / "run_local.py"), "--decoder",
           str(BP / "candidate"), "--problem-set", str(problem), "--runs", str(args.runs),
           "--threads", str(args.threads), "--out", str(cache / f"{tag}.csv")]
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    (results / f"{tag}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    m = re.search(r"runtime \(median of \d+\): ([\d.]+) s", result.stdout)
    ler = re.search(r"both: LER ([\d.]+)", result.stdout)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(command=cmd, config=json.loads((BP / "candidate" / args.config).read_text()),
                  threads=args.threads, runs=args.runs, problem_set=str(problem),
                  median_seconds=float(m.group(1)) if m else None,
                  ler=float(ler.group(1)) if ler else None,
                  source_hashes={str(p.relative_to(BP)): sha(p) for p in
                                 [BP / "enhanced.py", BP / "union.py", BP / "adaptive.py",
                                  BP / "pair_index.py", BP / "candidate" / "decoder.py"]})
    for p in (BP / "fused.py", BP / "bp_checkpoints.py", BP / "pool.py", BP / "local_closure.py"):
        record["source_hashes"][p.name] = sha(p)
    if m and ler:
        record["illustrative_local_score"] = (
            float(ler.group(1)) * max(1, float(m.group(1)) / 15.4) ** .2)
        record["score_note"] = "local timing only; not a judging-server score"
    (results / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
