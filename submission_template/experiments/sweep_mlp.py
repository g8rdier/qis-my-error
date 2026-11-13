"""Architecture/parameter sweep with k-fold cross-validation.

Compares MLP configs fairly: identical fold splits for every config, one shared
CSV log, weights tagged + registered. Winner is picked by mean CV val LER only;
data/eval_fresh stays untouched until a single final validation of the winner.

Pool: train-full 100k + gen_big 500k (600k/basis) unless --use-big2 (adds
gen_big2). A fixed rng(11) subset (default 300k/basis) and fixed rng(11) folds
are shared by all configs.

Usage (repo root):
  smoke (CPU, ~1 min):  .venv/.../python.exe submission_template/experiments/sweep_mlp.py --smoke
  full (GPU, ~2h):      .venv/.../python.exe submission_template/experiments/sweep_mlp.py
Results: submission_template/experiments/sweep_results.csv + console summary.
"""
import argparse
import csv
import os
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), "src"))

import numpy as np

from models import model_mlp
from models.common import REPO_ROOT, load_train

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="2 configs x 2 folds, 20k rows, 2 epochs, CPU")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--use-big2", action="store_true", help="include gen_big2 in the pool")
parser.add_argument("--folds", type=int, default=3)
parser.add_argument("--subset", type=int, default=300_000)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--patience", type=int, default=8)
parser.add_argument("--batch", type=int, default=8192)
args = parser.parse_args()

CONFIGS = [
    {"name": "256x128_do10", "hidden": [256, 128], "dropout": 0.1, "lr": 1e-3},
    {"name": "512x256_do10", "hidden": [512, 256], "dropout": 0.1, "lr": 1e-3},
    {"name": "1024x512_do10", "hidden": [1024, 512], "dropout": 0.1, "lr": 1e-3},
    {"name": "512x256_do20", "hidden": [512, 256], "dropout": 0.2, "lr": 1e-3},
    {"name": "512x256_do00", "hidden": [512, 256], "dropout": 0.0, "lr": 1e-3},
    {"name": "512x1_do10", "hidden": [512], "dropout": 0.1, "lr": 1e-3},
    {"name": "512x256_lr3e4", "hidden": [512, 256], "dropout": 0.1, "lr": 3e-4},
]
CSV_PATH = os.path.join(EXP_DIR, "sweep_results.csv")


def pool(basis, n):
    d = load_train(basis)
    D = [d["detector_events"]]
    O = [d["observable_flips"]]
    g = np.load(os.path.join(REPO_ROOT, "data", "gen_big", f"big_{basis}.npz"))
    D.append(g["D"].astype(bool))
    O.append(g["O"].astype(bool))
    if args.use_big2:
        g2 = np.load(os.path.join(REPO_ROOT, "data", "gen_big2", f"big_{basis}.npz"))
        D.append(g2["D"].astype(bool))
        O.append(g2["O"].astype(bool))
    D, O = np.concatenate(D), np.concatenate(O)
    rng = np.random.default_rng(11)
    idx = rng.choice(len(D), size=min(n, len(D)), replace=False)
    return D[idx], O[idx]


def main():
    import torch

    from models.common import registry_add

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    cfgs = CONFIGS[:2] if args.smoke else CONFIGS
    folds = 2 if args.smoke else args.folds
    n = 20_000 if args.smoke else args.subset
    epochs = 2 if args.smoke else args.epochs
    new_file = not os.path.exists(CSV_PATH)
    csvf = open(CSV_PATH, "a", newline="")
    wr = csv.writer(csvf)
    if new_file:
        wr.writerow(["time", "basis", "config", "hidden", "dropout", "lr", "fold",
                      "val_ler", "epochs_run", "seconds", "weights", "pool_n"])
    for basis in "XZ":
        D, O = pool(basis, n)
        print(f"{basis}: sweep pool {len(D)} rows", flush=True)
        perm = np.random.default_rng(11).permutation(len(D))
        parts = np.array_split(perm, folds)
        for ci, cfg in enumerate(cfgs):
            vals = []
            for f in range(folds):
                va = parts[f]
                tr = np.concatenate([parts[k] for k in range(folds) if k != f])
                torch.manual_seed(1000 + ci)  # same init per config across folds
                model = model_mlp.build_torch_model(tuple(cfg["hidden"]), cfg["dropout"]).to(device)
                t0 = time.perf_counter()
                best_state, best, ep_run, secs = model_mlp.fit_torch(
                    model, D[tr], O[tr], D[va], O[va], device, lr=cfg["lr"],
                    batch=args.batch, epochs=epochs, patience=args.patience,
                    seed=1234, tag=f"{basis}[{cfg['name']}/f{f}]", log_every=10**9)
                wname = f"{basis}_mlp_sweep_{cfg['name']}_f{f}.npz"
                params = model_mlp.torch_state_to_numpy(best_state)
                model_mlp.save(params, basis, arch_tag=f"sweep-{cfg['name']}", filename=wname)
                registry_add(wname, {"basis": basis, "arch": [70] + cfg["hidden"] + [4],
                                     "sweep": cfg["name"], "fold": f, "seed_init": 1000 + ci,
                                     "epochs": epochs, "batch": args.batch, "lr": cfg["lr"],
                                     "dropout": cfg["dropout"], "pool_n": len(D),
                                     "val_ler": round(float(best), 5),
                                     "date": time.strftime("%Y-%m-%d")})
                wr.writerow([time.strftime("%Y-%m-%dT%H:%M"), basis, cfg["name"],
                             "x".join(map(str, cfg["hidden"])), cfg["dropout"], cfg["lr"],
                             f, round(float(best), 5), ep_run, round(secs, 1), wname, len(D)])
                csvf.flush()
                vals.append(best)
                print(f"{basis} {cfg['name']} fold {f}: val {best:.5f} ({ep_run}ep {secs:.0f}s)", flush=True)
            print(f"{basis} {cfg['name']}: mean {np.mean(vals):.5f} std {np.std(vals):.5f}", flush=True)
    csvf.close()
    print(f"logged -> {CSV_PATH}")


if __name__ == "__main__":
    main()
