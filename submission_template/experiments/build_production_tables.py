"""Rebuild PRODUCTION LUT artifacts from train-full + gen_big (600k rows/basis).

This overwrites models/artifacts/{X,Z}_lut.npz consumed by decoder.py.
Nothing here touches data/test. Validate after with run_local on eval_fresh.

Usage (repo root): .venv/Scripts/python.exe submission_template/experiments/build_production_tables.py
"""
import os
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), "src"))

import numpy as np

from models import model_lut
from models.common import REPO_ROOT, load_train


def main():
    for basis in "XZ":
        d = load_train(basis)  # full 100k train, labels allowed
        g = np.load(os.path.join(REPO_ROOT, "data", "gen_big", f"big_{basis}.npz"))
        D = np.concatenate([d["detector_events"], g["D"].astype(bool)])
        O = np.concatenate([d["observable_flips"], g["O"].astype(bool)])
        print(f"{basis}: fitting {len(D)} rows...", flush=True)
        table = model_lut.fit_table(D, O)
        path = model_lut.save_table(table, basis)  # production location
        t = model_lut.load_table(basis)
        print(f"{basis}: distinct={len(t['keys'])} -> {path}", flush=True)


if __name__ == "__main__":
    main()
