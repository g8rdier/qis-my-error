"""Train + evaluate the logistic-regression baseline (model_logreg).

Protocol: TRAIN on 90% of data/train/train.csv per basis; EVAL on held-out 10%
+ fresh seed set. NEVER touches data/test/test.csv.

Usage (from repo root):
  .venv/Scripts/python.exe submission_template/experiments/train_logreg.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import numpy as np

from models.common import FRESH_EVAL_DIR, ler, load_train, train_val_split
from models import model_logreg


def main():
    for basis in "XZ":
        d = load_train(basis)
        D, O = d["detector_events"], d["observable_flips"]
        Dtr, Otr, Dva, Ova = train_val_split(D, O, 0.1, seed=0)
        print(f"{basis}: train {len(Dtr)} val {len(Ova)}", flush=True)
        W, b = model_logreg.train(Dtr, Otr, seed=0, epochs=30, batch=4096, lr=0.5)
        path = model_logreg.save(W, b, basis)
        print(f"{basis}: saved -> {path}")
        p = model_logreg.predict(Dva, W, b)
        print(f"  val:   LER={ler(p, Ova):.5f} (all-zero={ler(np.zeros_like(Ova), Ova):.5f})")
        try:
            from hackathon_2026 import load_kaggle_csv

            fresh = os.path.join(FRESH_EVAL_DIR, "eval_fresh.csv")
            if os.path.exists(fresh):
                fd = load_kaggle_csv(fresh)
                m = fd["basis"] == basis
                pf = model_logreg.predict(fd["detector_events"][m], W, b)
                print(f"  fresh: LER={ler(pf, fd['observable_flips'][m]):.5f} n={m.sum()}")
            else:
                print(f"  fresh: {fresh} not found, skipping")
        except Exception as e:
            print(f"  fresh: skipped ({e})")


if __name__ == "__main__":
    main()
