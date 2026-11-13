"""Scaling experiment: does a HUGE sorted lookup table solve the problem?

Question: if we generate millions of labelled shots and store every D->O mapping
in a sorted table (binary search at decode time), does exact-match coverage get
high enough that "best match" lookup beats the MLP?

Protocol (keeps train/test separation):
  SAMPLE: fresh Stim sample, seeds 1000/1001 — disjoint from data/train (seed 42)
          AND data/eval_fresh (seed 999). Saved to data/gen_big/*.npz for reuse.
  TABLES: build LUTs from the first n rows for n in SIZES (same code as model_lut).
  EVAL:   hit rate + hybrid (LUT-exact + existing MLP miss-fallback) LER on the
          untouched data/eval_fresh set. data/test/test.csv never touched.

Why "sorted" doesn't give nearest-match for free: integer sort order does not
place Hamming neighbours adjacently (adjacent integers can differ in many bits),
so binary search finds EXACT matches in C speed but cannot find "closest" rows.
True distance-1 best-match needs 70 flipped-key lookups (model_lut dist1 path).

Usage (repo root): .venv/Scripts/python.exe submission_template/experiments/measure_coverage.py
"""
import os
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))          # submission_template/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), "src"))

import numpy as np

from models import model_lut, model_mlp
from models.common import REPO_ROOT, ler

BIG_DIR = os.path.join(REPO_ROOT, "data", "gen_big")
TOTAL = 500_000
CHUNK = 50_000
SIZES = [10_000, 30_000, 90_000, 200_000, 500_000]
SEEDS = {"X": 1000, "Z": 1001}


def sample_big(basis):
    from hackathon_2026 import apply_circuit_noise, create_circuit, test_dataset

    os.makedirs(BIG_DIR, exist_ok=True)
    path = os.path.join(BIG_DIR, f"big_{basis}.npz")
    if os.path.exists(path):
        z = np.load(path)
        print(f"{basis}: loaded {len(z['D'])} sampled rows from {path}", flush=True)
        return z["D"].astype(bool), z["O"].astype(bool)
    circuit = apply_circuit_noise(create_circuit(basis, 5), p=0.002)
    Ds, Os = [], []
    for i in range(TOTAL // CHUNK):
        D, O = test_dataset(circuit, shots=CHUNK, seed=SEEDS[basis] + i)
        Ds.append(np.asarray(D, dtype=bool))
        Os.append(np.asarray(O, dtype=bool))
    D, O = np.concatenate(Ds), np.concatenate(Os)
    np.savez_compressed(path, D=D.astype(np.uint8), O=O.astype(np.uint8),
                         seed=np.array(SEEDS[basis]), shots=np.array(TOTAL))
    print(f"{basis}: sampled {len(D)} rows -> {path}", flush=True)
    return D, O


def main():
    from hackathon_2026 import load_kaggle_csv

    fd = load_kaggle_csv(os.path.join(REPO_ROOT, "data", "eval_fresh", "eval_fresh.csv"))
    print("n, distinct, hit_rate, hybrid_LER, t_build, t_decode_20k")
    for basis in "XZ":
        D, O = sample_big(basis)
        m = fd["basis"] == basis
        Df, Of = fd["detector_events"][m], fd["observable_flips"][m]
        mlp_params, _ = model_mlp.load(basis)
        fb = lambda Dm, _p=mlp_params: model_mlp.predict(Dm, _p)
        for n in SIZES:
            t0 = time.perf_counter()
            raw = model_lut.fit_table(D[:n], O[:n])
            path = model_lut.save_table(raw, basis, out_dir=BIG_DIR)
            table = model_lut.load_table(basis, in_dir=BIG_DIR)
            t_build = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = model_lut.predict(table, Df, fallback="model", fallback_model=fb)
            t_dec = time.perf_counter() - t0
            print(f"{basis} n={n}: distinct={len(table['keys'])} "
                  f"hit={model_lut.hit_rate(table, Df):.4f} "
                  f"hybrid_LER={ler(p, Of):.5f} build={t_build:.1f}s dec20k={t_dec:.2f}s",
                  flush=True)


if __name__ == "__main__":
    main()
