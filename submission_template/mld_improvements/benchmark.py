"""Official-shaped serial timing of the frozen candidate and train references.

Never overwrites a data folder. Generated predictions stay in the ignored cache.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(command, name, model=None):
    log = HERE / "results" / f"benchmark_{name}.txt"
    if log.exists():
        raise FileExistsError(log)
    print("RUN", subprocess.list2cmdline(command), flush=True)
    env = os.environ.copy()
    env.pop("MLD_IMPROVEMENTS_CONFIG", None)
    if model is not None:
        env["MLD_IMPROVEMENTS_CONFIG"] = str(model)
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"benchmark failed; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--fresh", type=Path, default=ROOT / "data/mld_audit_runtime")
    args = parser.parse_args()
    fresh = args.fresh.resolve()
    if fresh.exists():
        raise FileExistsError(f"refusing to overwrite or reuse {fresh}")
    (HERE / "cache").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "data/generate_data.py"), str(fresh),
                    "--shots", "100000", "--seed", "965100"], cwd=ROOT, check=True)
    for data_name, folder, name, decoder, runs, model in [
        ("fresh", fresh, "reference", HERE.parent, 1, None),
        ("fresh", fresh, "candidate", HERE, 5, HERE / "candidate.json"),
        ("fresh", fresh, "accuracy", HERE, 5, HERE / "accuracy.json"),
        ("train", ROOT / "data/train", "reference", HERE.parent, 1, None),
        ("train", ROOT / "data/train", "candidate", HERE, 1, HERE / "candidate.json"),
        ("train", ROOT / "data/train", "accuracy", HERE, 1, HERE / "accuracy.json"),
    ]:
        tag = f"{data_name}_{name}"
        command = [sys.executable, str(HERE.parent / "run_local.py"), "--decoder", str(decoder),
                   "--problem-set", str(folder), "--threads", str(args.threads),
                   "--runs", str(runs), "--out", str(HERE / "cache" / f"{tag}.csv")]
        run(command, tag, model)
    # Paired intervals on the exact run_local predictions, not just rounded LER.
    import numpy as np
    sys.path.insert(0, str(HERE.parent))
    from hackathon_2026 import load_kaggle_csv
    from mld_improvements.search import paired
    summary = {}
    for data_name, folder in [("fresh", fresh), ("train", ROOT / "data/train")]:
        data = load_kaggle_csv(folder / f"{folder.name}.csv")
        labels = np.concatenate([data["observable_flips"][data["basis"] == b] for b in "XZ"])
        pred = {}
        for name in ("reference", "candidate", "accuracy"):
            pred[name] = np.loadtxt(HERE / "cache" / f"{data_name}_{name}.csv",
                                     delimiter=",", skiprows=1, usecols=(1, 2, 3, 4),
                                     dtype=np.uint8).astype(bool)
        summary[data_name] = dict(
            reference_ler=float(np.mean(pred["reference"] != labels)),
            candidate_ler=float(np.mean(pred["candidate"] != labels)),
            accuracy_ler=float(np.mean(pred["accuracy"] != labels)),
            candidate_paired=paired(pred["candidate"], pred["reference"], labels),
            accuracy_paired=paired(pred["accuracy"], pred["reference"], labels))
    (HERE / "results/benchmark_paired.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
