"""Evaluate the ported BP+OSD decoder on a LABELLED problem set, run_local-style.

Refuses data/test (unlabelled): evaluation needs labels, and test stays decode-only.
Usage (repo root):
  .venv/Scripts/python.exe submission_template/bp_osd_decoder/run_bposd.py --problem-set data/eval_fresh --runs 5
"""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Time decode_bposd and report LER on a labelled problem set.")
parser.add_argument("--problem-set", default="data/eval_fresh")
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--threads", type=int, default=None)
parser.add_argument("--shots", type=int, default=None)
args = parser.parse_args()

if "test" in os.path.normpath(args.problem_set).split(os.sep):
    raise SystemExit("refusing data/test: evaluation needs labels and test stays decode-only")

if args.threads:
    for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)

import numpy as np  # noqa: E402
import stim  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
from hackathon_2026 import load_kaggle_csv, logical_error_rate  # noqa: E402
from bp_osd_decoder import decode_bposd  # noqa: E402

data_path = os.path.join(args.problem_set, os.path.basename(os.path.normpath(args.problem_set)) + ".csv")
print(f"loading {data_path} ...")
data = load_kaggle_csv(data_path)
assert "observable_flips" in data, "problem set has no labels; use a labelled set (train, eval_fresh)"
circuits, ids, det, obs = {}, {}, {}, {}
for b in "XZ":
    circuits[b] = stim.Circuit.from_file(os.path.join(args.problem_set, f"{b}_circuit.stim"))
    rows = np.flatnonzero(data["basis"] == b)[: args.shots]
    det[b] = data["detector_events"][rows]
    obs[b] = data["observable_flips"][rows]
    print(f"{args.problem_set} {b}: {len(rows)} shots")


def run():
    pred, t0 = {}, time.perf_counter()
    for b in "XZ":
        out = np.asarray(decode_bposd(circuits[b], det[b]))
        assert out.shape == (len(det[b]), 4) and np.isin(out, (0, 1)).all()
        pred[b] = out
    return pred, time.perf_counter() - t0


times, first = [], None
for i in range(args.runs):
    pred, secs = run()
    times.append(secs)
    print(f"run {i + 1}: {secs:.2f} s")
    if first is None:
        first = pred
    elif not all(np.array_equal(first[b], pred[b]) for b in "XZ"):
        print("  WARNING: predictions differ between runs; check seeding")
print(f"runtime (median of {args.runs}): {statistics.median(times):.2f} s")
for b in "XZ":
    print(f"{b}: LER {logical_error_rate(first[b], obs[b]):.5f}")
both_p = np.concatenate([first[b] for b in "XZ"])
both_t = np.concatenate([obs[b] for b in "XZ"])
print(f"both: LER {logical_error_rate(both_p, both_t):.5f}")
