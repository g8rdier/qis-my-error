"""Compare MLD with absolute, rather than normalized, closure candidate masses.

Uses the existing closure candidate generator without changing its release.
Retained vectors are rescored with float64 physical priors on MLD's scale.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numba
from numba import njit, prange
import numpy as np
import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from closure_decoder.bp_osd import collect, posteriors
from closure_decoder.candidates import create, ctz
from closure_decoder.closure import expand, get_index
from mld import marginals_logw


@njit(parallel=True, cache=True)
def _batch_logw(D, ev, ptr, cols, logical, prior, exact_prior, schedules, lam,
                lam2, capacity, gap, heads, nxt, ii, jj, closure_seeds):
    out = np.full((len(D), 16), -np.inf)
    for s in prange(len(D)):
        D0, D1 = np.uint64(0), np.uint64(0)
        for r in range(D.shape[1]):
            if D[s, r]:
                if r < 64:
                    D0 |= np.uint64(1) << np.uint64(r)
                else:
                    D1 |= np.uint64(1) << np.uint64(r - 64)
        vectors = np.empty((capacity, (len(prior) + 63) // 64), np.uint64)
        costs = np.empty(capacity, np.float64)
        logs = np.empty(capacity, np.uint8)
        hashes = np.empty(capacity, np.uint64)
        index = create(capacity)
        used = 0
        snapshots = posteriors(D[s], ev, ptr, prior, schedules)
        for i in range(len(schedules)):
            used = collect(snapshots[i], prior, cols, logical, D0, D1, D.shape[1],
                           lam, lam2, vectors, costs, logs, hashes, used, gap, index)
        used, _ = expand(cols, logical, prior, heads, nxt, ii, jj,
                          vectors, costs, logs, hashes, used, gap, index, closure_seeds)
        if not used:
            continue
        # Float32 BP rankings/candidate selection are unchanged. Absolute
        # likelihoods must use the same float64 p/(1-p) as MLD.
        for i in range(used):
            cost = 0.0
            for word in range(vectors.shape[1]):
                bits = vectors[i, word]
                while bits:
                    cost += exact_prior[64 * word + ctz(bits)]
                    bits &= bits - np.uint64(1)
            costs[i] = cost
        best = costs[:used].min()
        mass = np.zeros(16)
        for i in range(used):
            if costs[i] <= best + gap:
                mass[logs[i]] += np.exp(best - costs[i])
        for j in range(16):
            if mass[j] > 0:
                out[s, j] = np.log(mass[j]) - best
    return out


def closure_logw(graph, D, parameters, p):
    return _batch_logw(
        np.ascontiguousarray(D, dtype=np.uint8), graph["ev"], graph["ptr"],
        graph["cols"], graph["logical"], graph["prior"], np.log((1 - p) / p),
        np.asarray(parameters["schedules"], dtype=np.float64),
        parameters["lam"], parameters["lam2"], parameters["capacity"],
        parameters["gap"], *get_index(graph["cols"]), parameters["closure_seeds"])


def main():
    from closure_decoder.decoder import _STATES, PARAMETERS
    from hackathon_2026 import problem_info
    from mld_improvements.search import ROOT, score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="development_943100_3000")
    args = parser.parse_args()
    numba.set_num_threads(16)
    meta = json.loads((HERE / "results" / f"{args.prefix}.metadata.json").read_text())
    outputs, truths, refs, records = {}, [], [], []
    graphs = {state["basis"]: state["graph"] for state in _STATES.values()}
    seed0 = meta["arguments"]["seed"]
    shots = meta["arguments"]["shots"]
    for family in range(meta["arguments"]["families"]):
        for offset, basis in enumerate("XZ"):
            circuit = stim.Circuit.from_file(str(ROOT / f"data/train/{basis}_circuit.stim"))
            D, labels = circuit.compile_detector_sampler(seed=seed0 + 2 * family + offset).sample(
                shots, separate_observables=True)
            info = problem_info(circuit)
            closure_logw(graphs[basis], D[:1], PARAMETERS, info["p"])
            start = time.perf_counter()
            cw = closure_logw(graphs[basis], D, PARAMETERS, info["p"])
            seconds = time.perf_counter() - start
            with np.load(HERE / "cache" / f"{args.prefix}_{basis}_{family}_reference.npz") as z:
                np.testing.assert_array_equal(labels, z["labels"])
                baseline = z["logw"]
            ref = marginals_logw(baseline) > 0.5
            refs.append(ref)
            truths.append(labels)
            for name, weights in [("closure_absolute", cw),
                                  ("reference+closure_absolute", np.maximum(baseline, cw))]:
                prob = marginals_logw(weights)
                outputs.setdefault(name, []).append(prob)
                row = score(name, prob, ref, labels, seconds, family=family, basis=basis)
                records.append(row)
                print(basis, family, name, row["ler"], row["paired"], flush=True)
            np.savez_compressed(HERE / "cache" / f"{args.prefix}_{basis}_{family}_closure_absolute.npz",
                                logw=cw, labels=labels)
    truth, ref = np.concatenate(truths), np.concatenate(refs)
    summary = [score(name, np.concatenate(values), ref, truth,
                     sum(r["seconds"] for r in records if r["name"] == name))
               for name, values in outputs.items()]
    target = HERE / "results" / f"{args.prefix}.closure_absolute.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(dict(summary=summary, records=records), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
