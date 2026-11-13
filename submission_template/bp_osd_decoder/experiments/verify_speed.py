"""Verify speed-only kernels against ALL archived frozen-audit predictions.

Reads detector bits and archived predictions, NEVER ground-truth labels.
Source hashes differ because implementation changed; configurations and every
predicted bit must match. Refuses promotion on any difference.
"""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numba
import numpy as np
import stim
from hackathon_2026 import problem_info
from bp_osd_decoder.bp_osd import build_graph
from bp_osd_decoder.decoder_bposd import _load_table
from bp_osd_decoder.experiments.search import CACHE, REPO, RESULTS, table_lookup
from bp_osd_decoder.fused import decode_fused
from bp_osd_decoder.union import decode_union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    numba.set_num_threads(8)
    path = RESULTS / "frozen_audit_holdout.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()
            if json.loads(s)["config"] in ("accuracy_union", "routed_replace_005")]
    states = {}
    records = []
    for row in rows:
        basis, ident = row["basis"], row["identity"]
        if basis not in states:
            circuit = stim.Circuit.from_file(str(REPO / "data" / "train" / f"{basis}_circuit.stim"))
            states[basis] = build_graph(**problem_info(circuit)), _load_table(circuit)
        graph, table = states[basis]
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
        with np.load(CACHE / f"prediction_{key}.npz") as z:
            expected = z["pred"].copy()
        sample = CACHE / f"sample_{ident['circuit'][:12]}_{row['seed']}_{row['shots']}.npz"
        with np.load(sample) as z:
            D = z["D"]  # deliberately never access O
        pred, found = table_lookup(table, D)
        miss = ~(found & (D.sum(1) <= 4))
        start = time.perf_counter()
        if row["config"] == "accuracy_union":
            pred[miss] = decode_union(
                graph, D[miss], schedules=((3, .8, .5), (5, .8, .5), (8, .8, .5)),
                lam=600, lam2=200)
        else:
            pred[miss] = decode_fused(graph, D[miss])
        seconds = time.perf_counter() - start
        differences = int((pred != expected).sum())
        record = dict(config=row["config"], basis=basis, seed=row["seed"],
                      shots=len(D), bit_differences=differences, seconds=seconds)
        records.append(record)
        print(json.dumps(record), flush=True)
        if differences:
            raise AssertionError("speed-only implementation changed frozen predictions")
    bp = Path(__file__).resolve().parents[1]
    proof = dict(checks=records, labels_read=False,
                 source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (bp / "union.py", bp / "fused.py",
                                          bp / "pool.py", bp / "bp_checkpoints.py")})
    (RESULTS / f"speed_equivalence_{args.label}.json").write_text(
        json.dumps(proof, indent=2) + "\n")
    print("All 600,000 shot predictions (2.4M bits) match the archived frozen audit.")


if __name__ == "__main__":
    main()
