"""Per-observable threshold calibration for the pinned MLP weights.

Per-bit LER decomposes over observables, so each of the 4 thresholds is swept
independently. TUNE on the seed-2 val split of the 600k pool (same split the
model was selected on); VERIFY on data/eval_fresh. Applies to current.json
only if fresh LER improves by >0.0005 (else keeps 0.5s).

Usage (repo root): .venv/Scripts/python.exe submission_template/experiments/calibrate_thresholds.py
"""
import os
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), "src"))

import numpy as np

from models import model_mlp
from models.common import REPO_ROOT, current_load, current_save, ler, load_train, train_val_split

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-big2", action="store_true")
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()


def pool(basis):
    d = load_train(basis)
    D, O = [d["detector_events"]], [d["observable_flips"]]
    g = np.load(os.path.join(REPO_ROOT, "data", "gen_big", f"big_{basis}.npz"))
    D.append(g["D"].astype(bool))
    O.append(g["O"].astype(bool))
    if args.use_big2:
        g2 = np.load(os.path.join(REPO_ROOT, "data", "gen_big2", f"big_{basis}.npz"))
        D.append(g2["D"].astype(bool))
        O.append(g2["O"].astype(bool))
    return np.concatenate(D), np.concatenate(O)


def main():
    from hackathon_2026 import load_kaggle_csv

    fd = load_kaggle_csv(os.path.join(REPO_ROOT, "data", "eval_fresh", "eval_fresh.csv"))
    cur = current_load()
    for basis in "XZ":
        z = np.load(os.path.join(os.path.dirname(EXP_DIR), "models", "artifacts",
                                 cur[basis]["weights"]), allow_pickle=True)
        params = {k: z[k] for k in z.files if k != "arch_tag"}
        D, O = pool(basis)
        _, _, Dva, Ova = train_val_split(D, O, 0.1, seed=args.seed)
        Pva = np.concatenate([model_mlp.numpy_forward(Dva[i:i + 16384], params)
                              for i in range(0, len(Dva), 16384)])
        m = fd["basis"] == basis
        Df, Of = fd["detector_events"][m], fd["observable_flips"][m]
        Pf = np.concatenate([model_mlp.numpy_forward(Df[i:i + 16384], params)
                             for i in range(0, len(Df), 16384)])
        base_val = ler(Pva >= 0.5, Ova)
        base_fresh = ler(Pf >= 0.5, Of)
        thr = []
        for j in range(4):
            grid = np.arange(0.30, 0.701, 0.01)
            errs = [float(np.mean((Pva[:, j] >= t) != Ova[:, j])) for t in grid]
            t_best = float(grid[int(np.argmin(errs))])
            thr.append(t_best)
        thr = np.array(thr)
        val_new = ler(Pva >= thr, Ova)
        fresh_new = ler(Pf >= thr, Of)
        print(f"{basis}: baseline val={base_val:.5f} fresh={base_fresh:.5f}", flush=True)
        print(f"{basis}: thr={[round(t, 2) for t in thr]} val={val_new:.5f} fresh={fresh_new:.5f}", flush=True)
        if base_fresh - fresh_new > 0.0005:
            cur[basis]["threshold"] = [round(float(t), 3) for t in thr]
            current_save(cur)
            print(f"{basis}: applied to current.json")
        else:
            print(f"{basis}: kept 0.5 (gain too small)")


if __name__ == "__main__":
    main()
