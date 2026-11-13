"""Train + evaluate the LUT decoder (model_lut).

Protocol (do NOT deviate — keeps train/test separation clean):
  TRAIN on: data/train/train.csv (90% split, seed 0), labels used.
  EVAL on:  (a) held-out 10% of train, (b) data/eval_fresh (fresh seed).
  NEVER touches data/test/test.csv.

Usage (from repo root):
  .venv/Scripts/python.exe submission_template/experiments/train_lut.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import numpy as np

from models.common import FRESH_EVAL_DIR, load_train, train_val_split
from models import model_lut

RUNS = [
    ("zeros", {}),
    ("dist1", {"fallback": "dist1"}),
]


def main():
    for basis in "XZ":
        d = load_train(basis)
        D, O = d["detector_events"], d["observable_flips"]
        Dtr, Otr, Dva, Ova = train_val_split(D, O, 0.1, seed=0)
        table = model_lut.fit_table(Dtr, Otr)
        path = model_lut.save_table(table, basis)
        table = model_lut.load_table(basis)  # evaluate the saved (sorted) format
        print(f"{basis}: table patterns={len(table['keys'])} train_shots={len(Dtr)} -> {path}")
        for name, kw in RUNS:
            r = model_lut.evaluate(table, Dva, Ova, **kw)
            print(f"  val[{name}]: LER={r['ler']:.5f} hit_rate={r['hit_rate']:.4f} n={r['n']}")
        # fresh-seed check
        try:
            from hackathon_2026 import load_kaggle_csv

            fresh = os.path.join(FRESH_EVAL_DIR, "eval_fresh.csv")
            if os.path.exists(fresh):
                fd = load_kaggle_csv(fresh)
                m = fd["basis"] == basis
                for name, kw in RUNS:
                    r = model_lut.evaluate(table, fd["detector_events"][m], fd["observable_flips"][m], **kw)
                    print(f"  fresh[{name}]: LER={r['ler']:.5f} hit_rate={r['hit_rate']:.4f} n={r['n']}")
            else:
                print(f"  fresh: {fresh} not found, skipping")
        except Exception as e:
            print(f"  fresh: skipped ({e})")


if __name__ == "__main__":
    main()
