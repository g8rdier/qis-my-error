"""Fresh-seed MLD ablations, paired shot-level intervals, and cached ensembles.

Run from the repo root with .venv/Scripts/python.exe:
  submission_template/mld_improvements/search.py --stage development

Labels are used only after inference. No test CSV or leaderboard access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numba
import numpy as np
import stim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from hackathon_2026 import load_kaggle_csv, problem_info
from mld import marginals_logw
from mld_improvements.runtime import build_tables as tables, infer

BASE = dict(heuristic="max", hexp=0.5, order="span", compact=32,
            tiers=[[0, 1e-4, 1024, 0.0], [1, 3e-5, 4096, 0.15]])
CONFIGS = {"reference": BASE}
for name, changes in {
    "broader_route": dict(tiers=[[0, 1e-4, 1024, 0], [1, 3e-5, 4096, 0.02]]),
    "tight": dict(tiers=[[0, 3e-5, 4096, 0], [1, 1e-5, 8192, 0.05]]),
    "third_pass": dict(tiers=BASE["tiers"] + [[0, 3e-6, 8192, 0.15]]),
    "parity05": dict(heuristic="parity"),
    "parity075": dict(heuristic="parity", hexp=0.75),
    "parity10": dict(heuristic="parity", hexp=1.0),
    "hybrid05": dict(heuristic="hybrid"),
    "hybrid075": dict(heuristic="hybrid", hexp=0.75),
    "probability": dict(order="probability"),
    "span_probability": dict(order="span_probability"),
    "max065": dict(hexp=0.65),
    "merge32": dict(merge_factor=32.0),
    "merge_all": dict(merge_factor=1e100),
    "last_detector": dict(order="last"),
    "reverse_checks": dict(detector_order="reverse_checks"),
    "shuffle_checks": dict(detector_order="shuffle_checks"),
    "parity075_tight": dict(heuristic="parity", hexp=0.75,
                            tiers=[[0, 3e-5, 4096, 0], [1, 1e-5, 8192, 0.05]]),
    "hybrid075_tight": dict(heuristic="hybrid", hexp=0.75,
                            tiers=[[0, 3e-5, 4096, 0], [1, 1e-5, 8192, 0.05]]),
}.items():
    CONFIGS[name] = dict(BASE, **changes)


def paired(pred, reference, labels):
    delta = (np.mean(pred != labels, axis=1) -
             np.mean(reference != labels, axis=1))
    mean = float(delta.mean())
    radius = float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0
    return dict(delta=mean, ci95=[mean - radius, mean + radius],
                net_bits_saved=int(-delta.sum() * 4))


def score(name, probability, reference, labels, seconds, **extra):
    pred = probability > 0.5
    return dict(name=name, ler=float(np.mean(pred != labels)),
                wrong_per_bit=np.count_nonzero(pred != labels, axis=0).tolist(),
                seconds=seconds, paired=paired(pred, reference, labels), **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "audit", "train"], default="development")
    parser.add_argument("--shots", type=int, default=3000, help="shots per basis per seed family")
    parser.add_argument("--families", type=int, default=3)
    parser.add_argument("--seed", type=int, help="default: 943100 development, 954100 audit")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--names", nargs="+", default=list(CONFIGS))
    parser.add_argument("--config", type=Path, help="frozen named configurations instead of built-in ablations")
    parser.add_argument("--closure", action="store_true")
    args = parser.parse_args()
    if args.stage == "audit" and args.config is None:
        parser.error("audit requires --config with settings frozen before opening its labels")
    if args.shots < 2 or args.families < 1:
        parser.error("--shots must be at least 2 and --families must be positive")
    if args.seed is None:
        args.seed = 954100 if args.stage == "audit" else 943100
    numba.set_num_threads(args.threads)
    configs = json.loads(args.config.read_text()) if args.config else {n: CONFIGS[n] for n in args.names}
    configs = {"reference": BASE, **configs}
    result_dir = HERE / "results"
    cache_dir = HERE / "cache"
    result_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    prefix = f"{args.stage}_{args.seed}_{args.shots}"
    log_path = result_dir / f"{prefix}.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite {log_path}")
    metadata = dict(arguments=vars(args) | {"config": str(args.config) if args.config else None},
                    configurations=configs, versions=dict(numpy=np.__version__, numba=numba.__version__,
                                                           stim=stim.__version__),
                    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), HERE / "runtime.py", HERE.parent / "mld.py",
                                      ROOT / "data/train/X_circuit.stim",
                                      ROOT / "data/train/Z_circuit.stim"]})
    if args.closure:
        for path in (HERE.parent / "closure_decoder").glob("*.py"):
            metadata["hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("model.json", "assets.npz"):
            path = HERE.parent / "closure_decoder" / name
            metadata["hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (result_dir / f"{prefix}.metadata.json").write_text(json.dumps(metadata, indent=2))
    records = []
    pooled = {}
    all_labels = []
    all_reference = []
    graph_cache = {}
    train = load_kaggle_csv(ROOT / "data/train/train.csv") if args.stage == "train" else None
    if args.closure:
        from closure_decoder.decoder import _STATES
        from closure_decoder.bp_osd import solve
        from closure_decoder.decoder import PARAMETERS
        closure_graphs = {s["basis"]: s["graph"] for s in _STATES.values()}
    for family in range(1 if train else args.families):
        for offset, basis in enumerate("XZ"):
            seed = args.seed + 2 * family + offset
            circuit = stim.Circuit.from_file(str(ROOT / f"data/train/{basis}_circuit.stim"))
            info = problem_info(circuit)
            if train:
                rows = np.flatnonzero(train["basis"] == basis)
                D, labels = train["detector_events"][rows], train["observable_flips"][rows]
            else:
                D, labels = circuit.compile_detector_sampler(seed=seed).sample(
                    args.shots, separate_observables=True)
            all_labels.append(labels)
            outputs = {}
            times = {}
            for name, config in configs.items():
                graph_key = (basis, config["heuristic"], config["hexp"], config["order"],
                             config.get("merge_factor", 1.0), config.get("detector_order", "time"))
                if graph_key not in graph_cache:
                    graph_cache[graph_key] = tables(info, config)
                    infer(graph_cache[graph_key], D[:1], config)  # exclude JIT/setup
                start = time.perf_counter()
                logw, routed = infer(graph_cache[graph_key], D, config)
                seconds = time.perf_counter() - start
                prob = marginals_logw(logw)
                outputs[name] = prob
                times[name] = seconds
                if name == "reference":
                    reference = prob > 0.5
                    all_reference.append(reference)
                record = score(name, prob, reference, labels, seconds, basis=basis,
                               family=family, seed=seed, routed=routed, config=config)
                records.append(record)
                pooled.setdefault(name, []).append(prob)
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                np.savez_compressed(cache_dir / f"{prefix}_{basis}_{family}_{name}.npz",
                                    logw=logw, probability=prob, labels=labels)
                print(f"{basis}/{family} {name:20} LER {record['ler']:.7f} "
                      f"delta {record['paired']['delta']:+.7f} {seconds:.2f}s {routed}", flush=True)
            if args.closure:
                start = time.perf_counter()
                _, cp = solve(closure_graphs[basis], D, PARAMETERS, return_probabilities=True)
                seconds = time.perf_counter() - start
                # solve returns 16 logical-outcome probabilities.
                from mld import marginals
                prob = marginals(cp)
                pooled.setdefault("closure", []).append(prob)
                records.append(score("closure", prob, reference, labels, seconds,
                                     basis=basis, family=family, seed=seed))
                np.savez_compressed(cache_dir / f"{prefix}_{basis}_{family}_closure.npz",
                                    probability=prob, labels=labels)
                for weight in (0.1, 0.25, 0.5):
                    name = f"closure_blend{weight}"
                    blend = (1 - weight) * outputs["reference"] + weight * prob
                    pooled.setdefault(name, []).append(blend)
                    records.append(score(name, blend, reference, labels,
                                         seconds + times["reference"],
                                         basis=basis, family=family, seed=seed))
    truth, ref = np.concatenate(all_labels), np.concatenate(all_reference)
    summary = []
    for name, values in pooled.items():
        seconds = sum(r["seconds"] for r in records if r["name"] == name)
        row = score(name, np.concatenate(values), ref, truth, seconds)
        row["groups"] = [r for r in records if r["name"] == name]
        summary.append(row)
    summary.sort(key=lambda r: r["ler"])
    (result_dir / f"{prefix}.summary.json").write_text(json.dumps(summary, indent=2))
    # Held-out family selection diagnostics, not an independent final audit.
    if args.stage == "development" and args.families > 1:
        cv = []
        for held in range(args.families):
            means = {name: np.mean([r["ler"] for r in records
                                    if r["name"] == name and r["family"] != held])
                     for name in pooled}
            selected = min(means, key=lambda name: (means[name], name))
            cv.append(dict(family=held, selected=selected, ler=float(np.mean(
                [r["ler"] for r in records if r["name"] == selected and r["family"] == held]))))
        (result_dir / f"{prefix}.cv.json").write_text(json.dumps(cv, indent=2))
    print("\nAGGREGATE", flush=True)
    for row in summary:
        print(f"{row['name']:24} {row['ler']:.8f} {row['paired']} {row['seconds']:.2f}s")


if __name__ == "__main__":
    main()
