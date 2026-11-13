"""Train one MLP config (thin wrapper around models/model_mlp.py torch core).

Protocol: TRAIN on train(+pools) with a seeded holdout split; EVAL on held-out
split + data/eval_fresh. NEVER touches data/test/test.csv. Weights are tagged
and registered in artifacts/manifest.json; pinning to current.json (the decoder)
is explicit via --pin, only after fresh validation.

Usage examples (repo root):
  .venv/.../python.exe submission_template/experiments/train_mlp.py --use-big --hidden 512 256 --epochs 120 --patience 20 --batch 16384 --seed 2
  ... --use-big --use-big2 --hidden 512 256 --epochs 60 --patience 10 --batch 32768 --seed 3
"""
import argparse
import os
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))  # submission_template/

import numpy as np

from models import model_mlp
from models.common import REPO_ROOT, load_train, train_val_split

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=8192)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--basis", type=str, default="XZ", help="which bases to train, e.g. X, Z, or XZ")
parser.add_argument("--patience", type=int, default=6)
parser.add_argument("--use-big", action="store_true",
                    help="append data/gen_big 500k rows to the 100k train rows")
parser.add_argument("--use-big2", action="store_true",
                    help="append data/gen_big2 2000k rows as well")
parser.add_argument("--device", type=str, default=None, help="cuda/cpu (default: cuda if available)")
parser.add_argument("--tag", type=str, default=None, help="weight tag (default: auto)")
parser.add_argument("--pin", action="store_true",
                    help="point current.json at the new weights (only after fresh validation)")
args = parser.parse_args()


def run_basis(basis, device):
    import torch

    d = load_train(basis)
    D, O = d["detector_events"], d["observable_flips"]
    if args.use_big:
        g = np.load(os.path.join(REPO_ROOT, "data", "gen_big", f"big_{basis}.npz"))
        D = np.concatenate([D, g["D"].astype(bool)])
        O = np.concatenate([O, g["O"].astype(bool)])
    if args.use_big2:
        g2 = np.load(os.path.join(REPO_ROOT, "data", "gen_big2", f"big_{basis}.npz"))
        D = np.concatenate([D, g2["D"].astype(bool)])
        O = np.concatenate([O, g2["O"].astype(bool)])
    if args.use_big or args.use_big2:
        print(f"{basis}: pooled {len(D)} rows", flush=True)
    Dtr, Otr, Dva, Ova = train_val_split(D, O, 0.1, seed=args.seed)

    torch.manual_seed(args.seed)
    model = model_mlp.build_torch_model(tuple(args.hidden), args.dropout).to(device)
    tag = args.tag or f"{'x'.join(map(str, args.hidden))}_{len(D) // 1000}k_s{args.seed}"
    best_state, best, ep_run, secs = model_mlp.fit_torch(
        model, Dtr, Otr, Dva, Ova, device, lr=args.lr, batch=args.batch,
        epochs=args.epochs, patience=args.patience, seed=args.seed,
        tag=f"{basis}[{tag}]")
    params = model_mlp.torch_state_to_numpy(best_state)
    path, name = model_mlp.save(params, basis, arch_tag=f"mlp{'x'.join(map(str, args.hidden))}",
                                filename=f"{basis}_mlp_{tag}.npz")
    data = ("train100k" + ("+gen_big500k" if args.use_big else "")
            + ("+gen_big2_2000k" if args.use_big2 else "")
            if (args.use_big or args.use_big2) else "train90k-split")
    entry = {
        "basis": basis, "arch": [70] + [int(h) for h in args.hidden] + [4],
        "data": data, "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "dropout": args.dropout, "patience": args.patience,
        "val_ler": round(float(best), 5), "fresh_ler": None,
        "date": time.strftime("%Y-%m-%d"),
        "cmd": f"train_mlp.py --hidden {' '.join(map(str, args.hidden))} --epochs {args.epochs} "
               f"--patience {args.patience} --batch {args.batch} --seed {args.seed}"
               + (" --use-big" if args.use_big else "") + (" --use-big2" if args.use_big2 else ""),
    }
    from models.common import registry_add

    registry_add(name, entry)
    print(f"{basis}: best val LER {best:.5f} ({ep_run}ep, {secs:.0f}s) -> {path}")
    if args.pin:
        from models.common import current_load, current_save

        cur = current_load()
        thr = cur.get(basis, {}).get("threshold", [0.5, 0.5, 0.5, 0.5])
        cur[basis] = {"weights": name, "threshold": thr}
        current_save(cur)
        print(f"{basis}: pinned in current.json")
    return best


def main():
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    for basis in args.basis:
        assert basis in "XZ"
        run_basis(basis, device)


if __name__ == "__main__":
    main()
