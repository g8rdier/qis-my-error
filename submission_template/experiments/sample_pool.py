"""Sample a large labelled pool straight to compressed .npz (no CSV bottleneck).

Writes data/<name>/big_<BASIS>.npz with D (uint8), O (uint8), seed, shots.
Disjoint seeds from train (42), eval_fresh (999) and gen_big (1000/1001).

Usage (repo root):
  .venv/Scripts/python.exe submission_template/experiments/sample_pool.py --out data/gen_big2 --shots 2000000 --seed 2000
"""
import argparse
import os
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--shots", type=int, default=2_000_000)
parser.add_argument("--seed", type=int, default=2000)
parser.add_argument("--chunk", type=int, default=50_000)
args = parser.parse_args()

import numpy as np

from hackathon_2026 import apply_circuit_noise, create_circuit, test_dataset

os.makedirs(args.out, exist_ok=True)
for k, basis in enumerate("XZ"):
    circuit = apply_circuit_noise(create_circuit(basis, 5), p=0.002)
    Ds, Os = [], []
    for i in range(args.shots // args.chunk):
        D, O = test_dataset(circuit, shots=args.chunk, seed=args.seed + k + 2 * i)
        Ds.append(np.asarray(D, dtype=np.uint8))
        Os.append(np.asarray(O, dtype=np.uint8))
        if (i + 1) % 10 == 0:
            print(f"{basis}: {((i + 1) * args.chunk) // 1000}k / {args.shots // 1000}k", flush=True)
    path = os.path.join(args.out, f"big_{basis}.npz")
    np.savez_compressed(path, D=np.concatenate(Ds), O=np.concatenate(Os),
                        seed=np.array(args.seed), shots=np.array(args.shots))
    print(f"{basis}: wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)
