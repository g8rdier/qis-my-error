"""Release-only evidence kept on the research branch, not copied onto main.

qualify: bit-check the lean package on the archived 300k audit, then choose
         10M versus 50M LUT using new development seeds ONLY.
audit:   evaluate the frozen choice on 2M new shots, never retune from it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numba
import numpy as np
import stim

RESEARCH = Path(__file__).resolve().parents[3]
EXPERIMENTS = Path(__file__).resolve().parent
CACHE = EXPERIMENTS / "cache"
RELEASE = CACHE / "release_lut_50m"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def load_package(folder):
    sys.path.insert(0, str(folder.parent))
    from closure_decoder import decoder
    return decoder


def archived_equivalence(module, folder):
    rows = [json.loads(s) for s in (EXPERIMENTS / "results" / "closure_audit_holdout.jsonl")
            .read_text().splitlines() if json.loads(s)["config"] == "frozen_pair_closure"]
    checks = []
    for row in rows:
        ident, basis = row["identity"], row["basis"]
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
        with np.load(CACHE / f"prediction_{key}.npz") as z:
            expected = z["pred"]
        sample = CACHE / f"sample_{ident['circuit'][:12]}_{row['seed']}_{row['shots']}.npz"
        with np.load(sample) as z:
            D = z["D"]  # Do not read truth labels for a packaging-equivalence check.
        state = module._STATES[ident["circuit"]]
        old_graph = RESEARCH / "submission_template" / "bp_osd_decoder" / "candidate" / "artifacts" / f"graph_{basis}.npz"
        with np.load(old_graph) as z:
            for old, new in (("ev", "ev"), ("check_ptr", "ptr"), ("cols", "cols"),
                             ("logical", "logical"), ("llr0", "prior")):
                np.testing.assert_array_equal(z[old], state["graph"][new])
        pred = module._decode_state(state, D)
        changed = int((pred != expected).sum())
        check = dict(basis=basis, seed=row["seed"], shots=len(D), changed_bits=changed)
        checks.append(check)
        print("equivalence", json.dumps(check), flush=True)
        if changed:
            raise AssertionError("lean packaging changed the frozen model's decisions")
    write(RELEASE / "packaging_equivalence.json", {
        "checks": checks, "truth_labels_read": False,
        "runtime_source_sha256": {p.name: digest(p) for p in folder.glob("*.py")}})


def sample(state, seed, shots, role):
    fingerprint = hashlib.sha256(state["circuit"].encode()).hexdigest()
    path = RELEASE / f"{role}_{state['basis']}_{seed}_{shots}.npz"
    if not path.exists():
        c = stim.Circuit(state["circuit"])
        D, O = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        np.savez_compressed(path, D=D, O=O, seed=np.array(seed),
                            circuit_sha256=np.array(fingerprint))
    with np.load(path) as z:
        assert int(z["seed"]) == seed and str(z["circuit_sha256"]) == fingerprint
        return z["D"], z["O"]


def revised_lookup(module, original, challenger, D, original_pred):
    """Same physics: only re-decode rows that lose a formerly trusted LUT hit."""
    old_head, old_found, _ = module._lookup(original, D)
    new_head, new_found, _ = module._lookup(challenger, D)
    easy = D.sum(1) <= module.MODEL["lut_max_fired"]
    old_use, new_use = old_found & easy, new_found & easy
    result = original_pred.copy()
    result[new_use] = new_head[new_use]
    lost = old_use & ~new_use
    if lost.any():
        result[lost] = module.solve(challenger["graph"], D[lost], module.PARAMETERS)
    affected = (old_use != new_use) | (old_use & new_use & (old_head != new_head).any(1))
    if affected.any():
        direct = module._decode_state(challenger, D[affected])
        np.testing.assert_array_equal(result[affected], direct)
    return result, old_use, new_use, int(affected.sum())


def measure(pred, truth):
    return (pred != truth).mean(1)


def aggregate(rows):
    n = sum(r["shots"] for r in rows)
    mean = sum(r["loss_sum"] for r in rows) / n
    variance = sum(r["shots"] * r["loss_var"] for r in rows) / n ** 2
    delta = sum(r.get("delta_sum", 0) for r in rows) / n
    delta_variance = sum(r["shots"] * r.get("delta_var", 0) for r in rows) / n ** 2
    return {"shots": n, "logical_bits": 4 * n, "ler": mean,
            "ler_95ci": [mean - 1.96 * np.sqrt(variance), mean + 1.96 * np.sqrt(variance)],
            "delta_vs_original": delta,
            "delta_95ci": [delta - 1.96 * np.sqrt(delta_variance),
                           delta + 1.96 * np.sqrt(delta_variance)],
            "per_basis_ler": {b: sum(r["loss_sum"] for r in rows if r["basis"] == b)
                              / sum(r["shots"] for r in rows if r["basis"] == b) for b in "XZ"}}


def develop(module, folder):
    original = module._load_states(RELEASE / "assets_10m.npz")
    challenger = module._load_states(RELEASE / "assets_50m.npz")
    rows = []
    for fingerprint, old in original.items():
        new = challenger[fingerprint]
        for name in ("ev", "ptr", "cols", "logical", "prior"):
            np.testing.assert_array_equal(old["graph"][name], new["graph"][name])
        for fold in range(3):
            seed = 920100 + 2 * fold + (old["basis"] == "Z")
            D, O = sample(old, seed, 50000, "development")
            start = time.perf_counter()
            base = module._decode_state(old, D)
            pred, old_head, new_head, affected = revised_lookup(module, old, new, D, base)
            base_loss, loss = measure(base, O), measure(pred, O)
            delta = loss - base_loss
            row = {"basis": old["basis"], "seed": seed, "shots": len(D),
                   "original_ler": float(base_loss.mean()), "ler": float(loss.mean()),
                   "loss_sum": float(loss.sum()), "loss_var": float(loss.var(ddof=1)),
                   "delta_sum": float(delta.sum()), "delta_var": float(delta.var(ddof=1)),
                   "affected_rows": affected, "original_head_shots": int(old_head.sum()),
                   "challenger_head_shots": int(new_head.sum()),
                   "original_head_wrong_bits": int((base[old_head] != O[old_head]).sum()),
                   "original_search_wrong_bits": int((base[~old_head] != O[~old_head]).sum()),
                   "seconds": time.perf_counter() - start}
            rows.append(row)
            print("development", json.dumps(row), flush=True)
            write(RELEASE / "development.json", {"rows": rows})
    result = aggregate(rows)
    choose_new = result["delta_vs_original"] < 0 and result["delta_95ci"][1] < 0
    result["selected_shots_per_basis"] = 50000000 if choose_new else 10000000
    result["selection_rule"] = "50M must improve paired development LER with 95% CI below zero; otherwise retain frozen 10M."
    result["rows"] = rows
    write(RELEASE / "development.json", result)
    model_path = folder / "model.json"
    model = json.loads(model_path.read_text())
    if choose_new:
        shutil.copyfile(RELEASE / "assets_50m.npz", folder / "assets.npz")
        model["lookup_training"] = {
            "shots_per_basis": 50000000, "seeds": {"X": 961001, "Z": 961002},
            "generator": "prepare.py train --shots 50000000 --chunk 1000000 --seed 961001"}
    else:
        model["lookup_training"]["selection"] = "Retained: 50M challenger did not demonstrate a paired development improvement."
    model["assets_sha256"] = digest(folder / "assets.npz")
    model["development_selection"] = {k: v for k, v in result.items() if k != "rows"}
    model["additional_training"] = {
        "challenger_shots_per_basis": 50000000, "seeds": [961001, 961002],
        "selected": bool(choose_new), "trained_samples_committed": False}
    model["release_validation"] = {
        "status": "frozen_before_final_audit",
        "seeds": list(range(940100, 940108)), "shots_per_seed": 250000,
        "total_shots": 2000000, "hyperparameters_retuned_after_audit": False}
    write(model_path, model)
    print("FROZEN SELECTION", json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


def audit(module, folder):
    metadata = json.loads((folder / "model.json").read_text())
    assert metadata["release_validation"]["status"] == "frozen_before_final_audit"
    assert digest(folder / "assets.npz") == metadata["assets_sha256"]
    selected = module._STATES
    original = module._load_states(RELEASE / "assets_10m.npz")
    rows = []
    for fingerprint, state in selected.items():
        old = original[fingerprint]
        for fold in range(4):
            seed = 940100 + 2 * fold + (state["basis"] == "Z")
            D, O = sample(state, seed, 250000, "final_audit")
            start = time.perf_counter()
            pred = module._decode_state(state, D)
            if metadata["additional_training"]["selected"]:
                base, _, _, _ = revised_lookup(module, state, old, D, pred)
            else:
                base = pred
            loss, base_loss = measure(pred, O), measure(base, O)
            delta = loss - base_loss
            _, found, _ = module._lookup(state, D)
            head = found & (D.sum(1) <= module.MODEL["lut_max_fired"])
            row = {"basis": state["basis"], "seed": seed, "shots": len(D),
                   "loss_sum": float(loss.sum()), "loss_var": float(loss.var(ddof=1)),
                   "delta_sum": float(delta.sum()), "delta_var": float(delta.var(ddof=1)),
                   "ler": float(loss.mean()), "per_bit_ler": (pred != O).mean(0).tolist(),
                   "head_shots": int(head.sum()),
                   "head_wrong_bits": int((pred[head] != O[head]).sum()),
                   "search_wrong_bits": int((pred[~head] != O[~head]).sum()),
                   "seconds": time.perf_counter() - start}
            rows.append(row)
            print("FINAL AUDIT", json.dumps(row), flush=True)
            write(RELEASE / "final_audit.json", {"rows": rows})
    result = aggregate(rows)
    result["rows"] = rows
    result["status"] = "completed_no_retuning"
    result["stim_version"] = stim.__version__
    result["numpy_version"] = np.__version__
    result["numba_version"] = numba.__version__
    result["assets_sha256"] = metadata["assets_sha256"]
    write(RELEASE / "final_audit.json", result)
    metadata["release_validation"] = {k: v for k, v in result.items() if k != "rows"}
    metadata["release_validation"]["seeds"] = [r["seed"] for r in rows]
    metadata["release_validation"]["shots_per_seed"] = 250000
    metadata["release_validation"]["hyperparameters_retuned_after_audit"] = False
    write(folder / "model.json", metadata)
    print("FINAL AUDIT SUMMARY", json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["qualify", "audit"])
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    module = load_package(args.package.resolve())
    if args.phase == "qualify":
        archived_equivalence(module, args.package)
        develop(module, args.package)
    else:
        audit(module, args.package)


if __name__ == "__main__":
    main()
