"""Exact weight<=2 lower bound on Bayes bit error under the independent DEM.

For each syndrome, let A_b be joint probability mass with logical bit b
and mechanism count <=2. Full-model masses can only add nonnegative terms:
min(A_0+C_0, A_1+C_1) >= min(A_0,A_1). Summing these minima gives a lower
bound, NOT the error rate conditional on low weight. No labelled samples.

The current circuits have unique, nonzero detector columns. This simplifies
accounting of zero/single errors; the implementation asserts that property.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from numba import njit
import numpy as np
import stim
from hackathon_2026 import problem_info
from bp_osd_decoder.enhanced import build_graph
from bp_osd_decoder.pair_index import build, find


@njit(cache=True)
def compute(cols, logical, p, heads, nxt, ii, jj):
    odds = p / (1 - p)
    additions = np.zeros(len(ii), np.float64)
    single_class = np.zeros(len(ii), np.uint8)
    for j in range(len(p)):
        k = find(cols[j, 0], cols[j, 1], cols, heads, ii, jj)
        if k >= 0:
            additions[k] = odds[j]
            single_class[k] = logical[j]
    risk = np.zeros(4)
    joint = np.zeros(16)
    ambiguous_mass = 0.
    ambiguous_groups = 0
    for head in heads:
        if head < 0 or (nxt[head] < 0 and additions[head] == 0):
            continue
        joint[:] = 0
        k = head
        while k >= 0:
            joint[logical[ii[k]] ^ logical[jj[k]]] += odds[ii[k]] * odds[jj[k]]
            k = nxt[k]
        if additions[head]:
            joint[single_class[head]] += additions[head]
        total = joint.sum()
        any_ambiguous = False
        for bit in range(4):
            one, zero = 0., 0.
            for cls in range(16):
                if (cls >> bit) & 1:
                    one += joint[cls]
                else:
                    zero += joint[cls]
            risk[bit] += min(one, zero)
            any_ambiguous |= one > 0 and zero > 0
        if any_ambiguous:
            ambiguous_mass += total
            ambiguous_groups += 1
    zero_mass = np.exp(np.log1p(-p).sum())
    return risk * zero_mass, ambiguous_mass * zero_mass, ambiguous_groups


def main():
    repo = Path(__file__).resolve().parents[3]
    rows = []
    for basis in "XZ":
        c = stim.Circuit.from_file(str(repo / "data" / "train" / f"{basis}_circuit.stim"))
        info = problem_info(c)
        H, L, p = info["H"], info["L"], info["p"]
        assert (H.sum(0) > 0).all()
        assert len(np.unique(H.T, axis=0)) == H.shape[1]
        graph = build_graph(H, L, p)
        index = build(graph["cols"])
        risk, mass, groups = compute(graph["cols"], graph["logical"], p, *index)
        odds, z0 = p / (1 - p), float(np.exp(np.log1p(-p).sum()))
        restricted_mass = z0 * (1 + odds.sum() + (odds.sum() ** 2 - (odds ** 2).sum()) / 2)
        row = dict(basis=basis, lower_bound_ler=float(risk.mean()),
                   bitwise_lower_bounds=risk.tolist(), low_weight_probability=float(restricted_mass),
                   ambiguous_low_weight_mass=mass, ambiguous_syndromes=int(groups),
                   scope="independent DEM, weight<=2 sub-probability; not conditioned/renormalized")
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    out = Path(__file__).resolve().parent / "results" / "low_weight_bound.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
