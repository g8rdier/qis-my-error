"""Sequential ablations with independent-seed cross-validation and paired errors.

Run from the repository root:
  .venv/Scripts/python.exe submission_template/bp_osd_decoder/experiments/search.py

Development: three newly sampled seeds per basis. Holdout: a different, sealed
seed family, evaluated only after choosing a configuration. No data/test input.
The immutable imported bp_osd.py is the reference. Each result records code,
circuit, LUT and (when relevant) MLP hashes. Timings here exclude setup/JIT:
use run_local.py with the isolated hand-in entry point for final timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numba
import numpy as np
import stim
from hackathon_2026 import problem_info

from bp_osd_decoder.bp_osd import build_graph, decode_bp_osd
from bp_osd_decoder.decoder_bposd import _load_table

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
REPO = ROOT.parent
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)
DEFAULT = dict(max_iter=3, alpha=0.8, damping=0.5, avg_post=True,
               always_osd=True, lam=400, lam2=100, coset=2, temp=1.0,
               refine_thr=0.3, refine_lam=1000, refine_lam2=300)


def configs():
    out = {"baseline": {}}
    for t in (0.7, 0.85, 1.15, 1.3):
        out[f"temperature_{t}"] = {"temp": t}
    for n in (2, 4, 5, 8):
        out[f"iterations_{n}"] = {"max_iter": n}
    for a in (0.6, 0.9, 1.0):
        out[f"alpha_{a}"] = {"alpha": a}
    for d in (0.3, 0.7):
        out[f"damping_{d}"] = {"damping": d}
    out["last_posterior"] = {"avg_post": False}
    for singles, pairs in ((600, 200), (1000, 300), (1500, 500), (2300, 800)):
        out[f"sweep_{singles}_{pairs}"] = {
            "lam": singles, "lam2": pairs, "refine_thr": 0}
    out["no_refinement"] = {"refine_thr": 0}
    out["refine_more"] = {"refine_thr": 0.15}
    out["refine_wider"] = {"refine_lam": 1500, "refine_lam2": 500}
    for w in (0.05, 0.1, 0.2, 0.35):
        out[f"mlp_blend_{w}"] = {"mlp_weight": w}
    for cutoff in (-1, 2, 3, 5, 6):
        out[f"lut_cutoff_{cutoff}"] = {"lut_cutoff": cutoff}
    return out


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sample(circuit, seed, shots):
    CACHE.mkdir(parents=True, exist_ok=True)
    chash = hashlib.sha256(str(circuit).encode()).hexdigest()
    path = CACHE / f"sample_{chash[:12]}_{seed}_{shots}.npz"
    if not path.exists():
        D, O = circuit.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        np.savez_compressed(path, D=D, O=O)
    with np.load(path) as z:
        return z["D"], z["O"]


def table_lookup(table, D):
    keys, labels = table
    q = np.ascontiguousarray(np.packbits(D, axis=1, bitorder="little"))
    q = q.view("V9").ravel()
    idx = np.searchsorted(keys, q)
    safe = np.minimum(idx, len(keys) - 1)
    found = (idx < len(keys)) & (keys[safe] == q)
    pred = np.unpackbits(labels[safe, None], axis=1, count=4, bitorder="little")
    return pred, found


def cached_decode(basis, circuit, graph, table, D, seed, changes):
    cutoff = changes.get("lut_cutoff", 4)
    weight = changes.get("mlp_weight", 0.0)
    params = dict(DEFAULT)
    params.update({k: v for k, v in changes.items()
                   if k not in ("lut_cutoff", "mlp_weight", "enhanced", "union", "adaptive")})
    enhanced = changes.get("enhanced", False)
    union = changes.get("union", False)
    adaptive = changes.get("adaptive", False)
    core = Path(__file__).resolve().parents[1] / (
        "adaptive.py" if adaptive else
        ("union.py" if union else ("enhanced.py" if enhanced else "bp_osd.py")))
    fingerprint = dict(
        circuit=hashlib.sha256(str(circuit).encode()).hexdigest(),
        source=digest_file(core), adapter=digest_file(__file__),
        lut=digest_file(core.parent / "lut" / f"lut_{basis}.npz"),
        params=params, seed=seed, shots=len(D), cutoff=cutoff)
    if union or adaptive:
        fingerprint["pair_index_source"] = digest_file(core.parent / "pair_index.py")
        fingerprint["enhanced_source"] = digest_file(core.parent / "enhanced.py")
        fingerprint["checkpoint_source"] = digest_file(core.parent / "bp_checkpoints.py")
        fingerprint["pool_source"] = digest_file(core.parent / "pool.py")
        fingerprint["closure_source"] = digest_file(core.parent / "local_closure.py")
    if adaptive:
        fingerprint["union_source"] = digest_file(core.parent / "union.py")
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    path = CACHE / f"prediction_{key}.npz"
    if path.exists():
        with np.load(path) as z:
            marg, pred = z["marg"], z["pred"]
            seconds = float(z["seconds"])
    else:
        start = time.perf_counter()
        lut_pred, found = table_lookup(table, D)
        use_lut = found & (D.sum(1) <= cutoff)
        marg = lut_pred.astype(np.float32)
        pred = lut_pred.copy()
        if (~use_lut).any():
            if adaptive:
                from bp_osd_decoder.adaptive import decode_adaptive
                settings = {k: v for k, v in params.items()
                            if k in ("initial", "refiner", "route_threshold")}
                p, cp = decode_adaptive(graph, D[~use_lut],
                                        return_classp=True, **settings)
            elif union:
                from bp_osd_decoder.union import decode_union
                union_params = {k: v for k, v in params.items()
                                if k in ("lam", "lam2", "schedules", "capacity", "gap", "exact_order3",
                                         "jitter_trials", "jitter_strength", "jitter_seed", "jitter_schedule",
                                         "closure_seeds", "closure_rounds", "closure_per_class")}
                p, cp = decode_union(graph, D[~use_lut],
                                     return_classp=True, **union_params)
            elif enhanced:
                from bp_osd_decoder.enhanced import decode_enhanced
                p, cp = decode_enhanced(graph, D[~use_lut],
                                        return_classp=True, **params)
            else:
                p, cp = decode_bp_osd(graph, D[~use_lut],
                                      return_classp=True, **params)
            assert np.isfinite(cp).all(), "non-finite candidate probabilities"
            assert np.allclose(cp.sum(1), 1, atol=1e-5)
            pred[~use_lut] = p
            marg[~use_lut] = cp @ BITS
        seconds = time.perf_counter() - start
        np.savez_compressed(path, pred=pred, marg=marg, seconds=seconds,
                            fingerprint=json.dumps(fingerprint, sort_keys=True))
    if weight:
        from models.model_mlp import numpy_forward
        filename = f"{basis}_mlp_1024x512_2600k_s4.npz"
        wpath = ROOT / "models" / "artifacts" / filename
        fingerprint["mlp"] = digest_file(wpath)
        fingerprint["mlp_weight"] = weight
        with np.load(wpath) as z:
            w = {k: z[k] for k in z.files if k != "arch_tag"}
        start = time.perf_counter()
        pm = np.concatenate([numpy_forward(D[i:i + 4096], w)
                             for i in range(0, len(D), 4096)])
        marg = (1 - weight) * marg + weight * pm
        pred = (marg > 0.5).astype(np.uint8)
        seconds += time.perf_counter() - start
    return pred, marg, seconds, fingerprint


def report(rows, output):
    holdout = "holdout" in output.name
    names = list(dict.fromkeys(r["config"] for r in rows))
    summary = []
    for name in names:
        rr = [r for r in rows if r["config"] == name]
        by_basis = {b: float(np.mean([r["ler"] for r in rr if r["basis"] == b]))
                    for b in "XZ"}
        delta = float(np.mean([r["delta_ler"] for r in rr]))
        se = float(np.sqrt(sum(r["delta_se"] ** 2 for r in rr)) / len(rr))
        summary.append(dict(config=name, ler=np.mean(list(by_basis.values())),
                            X=by_basis["X"], Z=by_basis["Z"], delta=delta,
                            paired_95ci=[delta - 1.96 * se, delta + 1.96 * se],
                            seconds=np.mean([r["seconds"] for r in rr]),
                            folds=len(rr)))
    summary.sort(key=lambda r: r["ler"])
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = [("# BP+OSD frozen independent holdout" if holdout else
             "# BP+OSD development ablations"), "",
            "Independent seed folds; paired CIs use per-shot bit-error differences.",
            ("Configurations were frozen before opening these seeds; no parameter selection here."
             if holdout else
             "CIs do not correct for model selection. These are NOT final holdout scores."),
            "Setup/JIT excluded; use run_local.py for official-shaped timing.", "",
            "| Configuration | Both LER | X | Z | Delta vs reference | Paired 95% CI |",
            "|---|---:|---:|---:|---:|---|"]
    for r in summary:
        lo, hi = r["paired_95ci"]
        text.append(f"| {r['config']} | {r['ler']:.6f} | {r['X']:.6f} | "
                    f"{r['Z']:.6f} | {r['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    # Leave one entire independently sampled seed family out when selecting.
    seed_folds = sorted(set(r["fold"] for r in rows))
    if not holdout and len(seed_folds) >= 3 and len(names) > 1:
        text += ["", "## Leave-one-seed-family-out selection check", ""]
        for fold in seed_folds:
            train = [r for r in rows if r["fold"] != fold]
            means = {name: np.mean([r["ler"] for r in train if r["config"] == name])
                     for name in names}
            winner = min(means, key=means.get)
            test = [r for r in rows if r["fold"] == fold and r["config"] == winner]
            text.append(f"- Fold {fold}: choose `{winner}` without this seed; "
                        f"left-out LER {np.mean([r['ler'] for r in test]):.6f}.")
    output.with_suffix(".md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=8000, help="per basis per seed")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--config-file", type=Path, help="named configurations (JSON)")
    parser.add_argument("--name", default="stage1")
    parser.add_argument("--seed-base", type=int,
                        help="new sealed seed family for subsequent audits; never retune on an opened audit")
    args = parser.parse_args()
    assert args.shots > 0 and args.folds > 0
    numba.set_num_threads(args.threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    bank = configs()
    if args.config_file:
        bank.update(json.loads(args.config_file.read_text(encoding="utf-8")))
    selected = args.ids or list(bank)
    assert all(name in bank for name in selected)
    rows = []
    output = RESULTS / f"{args.name}_{args.split}.jsonl"
    seedbase = args.seed_base if args.seed_base is not None else (
        610100 if args.split == "development" else 720100)
    # Compile/cache before measuring, including both refinement stages.
    for basis in "XZ":
        circuit = stim.Circuit.from_file(str(REPO / "data" / "train" / f"{basis}_circuit.stim"))
        info = problem_info(circuit)
        graph = build_graph(**info)
        table = _load_table(circuit)
        warm_D, _ = load_sample(circuit, seedbase + 900, 8)
        decode_bp_osd(graph, warm_D, **DEFAULT)
        if any(bank[name].get("enhanced", False) for name in selected):
            from bp_osd_decoder.enhanced import decode_enhanced
            decode_enhanced(graph, warm_D, lam3=8, **DEFAULT)
        if any(bank[name].get("union", False) for name in selected):
            from bp_osd_decoder.union import decode_union
            decode_union(graph, warm_D)
            if any(bank[name].get("exact_order3", False) for name in selected):
                decode_union(graph, warm_D, exact_order3=True)
            if any(bank[name].get("closure_seeds", 0) for name in selected):
                decode_union(graph, warm_D, closure_seeds=1)
        if any(bank[name].get("adaptive", False) for name in selected):
            from bp_osd_decoder.adaptive import decode_adaptive
            decode_adaptive(graph, warm_D)
        for fold in range(args.folds):
            seed = seedbase + 2 * fold + (basis == "Z")
            D, O = load_sample(circuit, seed, args.shots)
            baseline, _, _, _ = cached_decode(
                basis, circuit, graph, table, D, seed, {})
            ref_loss = (baseline != O).mean(1)
            for name in selected:
                pred, marg, seconds, identity = cached_decode(
                    basis, circuit, graph, table, D, seed, bank[name])
                loss = (pred != O).mean(1)
                delta = loss - ref_loss
                row = dict(config=name, basis=basis, fold=fold, seed=seed,
                           shots=len(D), ler=float(loss.mean()),
                           bit_ler=(pred != O).mean(0).tolist(),
                           baseline_ler=float(ref_loss.mean()),
                           delta_ler=float(delta.mean()),
                           delta_se=float(delta.std(ddof=1) / np.sqrt(len(D))),
                           seconds=seconds, identity=identity)
                rows.append(row)
                # Atomic rewrite: a rerun does not duplicate measurements.
                output.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                  encoding="utf-8")
                print(f"{basis} seed={seed} {name}: {row['ler']:.6f} "
                      f"delta={row['delta_ler']:+.6f} {seconds:.2f}s", flush=True)
    summary = report(rows, output.with_suffix(".summary.json"))
    print("\nBest:", json.dumps(summary[0], indent=2), flush=True)


if __name__ == "__main__":
    main()
