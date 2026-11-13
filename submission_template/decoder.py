"""Near-maximum-likelihood decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README). Importing the decoder, and anything it builds when imported, is
not timed.

Every shot is decoded by the pruned dynamic programme in mld.py, which computes the weight of each of the 16 logical
outcomes and predicts each logical bit by its marginal (the optimal decision for the per-bit LER). It runs in
passes: the first decodes every shot, and each later pass re-decodes the shots where some logical bit is still
uncertain.

Pruning only removes terms from an outcome's weight, so every pass gives a lower bound on it, and passes are
combined by keeping the largest bound per outcome. Passes alternate between sweeping the error mechanisms forward
and backward in time: the two directions prune different error vectors, so their combination is closer to the exact
weights than one sweep with a tighter threshold that costs the same.

At import we rebuild the competition's two circuits (5 rounds, SI1000 p = 0.002), precompute the decoder tables for
each and compile the numba kernel, so decode() only has to look the tables up by a hash of the circuit text. A
circuit that is not one of these is handled too, by building its tables inside decode().
"""
import hashlib
from re import search
from typing_extensions import runtime

import numpy as np
import stim
from hackathon_2026 import apply_circuit_noise, create_circuit, problem_info

from mld import build_mld, marginals_logw, mld_logw

ROUNDS, NOISE = 5, 0.002
# passes (direction, eps, max_states, thr): the first pass decodes every shot; each later pass re-decodes the shots
# where some logical bit still has a probability between thr and 1 - thr. Direction 0 sweeps the mechanisms forward
# in time, 1 backward.
MLD_TIERS = {
    "X": ((0, 4.4444444444444447e-05, 2816, 0.0), (1, 3e-05, 3328, 0.125)),
    "Z": ((0, 0.0001, 2816, 0.0), (1, 2e-05, 3072, 0.125)),
}
MLD_COMPACT_EVERY = {"X": 32, "Z": 32}
MLD_HEXP = {"X": 0.4, "Z": 0.6}



def _key(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _tables(circuit: stim.Circuit, basis: str):
    m = problem_info(circuit)
    # backward sweep: the same decoder on the detectors in reverse order
    return (
        build_mld(m["H"], m["L"], m["p"], hexp=MLD_HEXP[basis]),
        build_mld(m["H"][::-1].copy(), m["L"], m["p"], hexp=MLD_HEXP[basis]),
    )


_TABLES = {}
for _basis in "XZ":
    _c = apply_circuit_noise(create_circuit(_basis, ROUNDS), p=NOISE)
    _TABLES[_key(_c)] = (_tables(_c, _basis), _basis)
for _g, _basis in _TABLES.values():  # compile (or load from numba's cache) the kernel before the clock starts
    for _dir, _eps, _max_states, _ in MLD_TIERS[_basis]:
        mld_logw(
            _g[_dir],
            np.zeros((2, _g[_dir]["nd"]), bool),
            eps=_eps,
            max_states=_max_states,
            compact_every=MLD_COMPACT_EVERY[_basis],
        )


def decode_marginals(g, detector_events: np.ndarray, basis: str = "X") -> np.ndarray:
    """Per-bit probability of a logical flip, shape (shots, 4)."""
    D = np.asarray(detector_events, dtype=bool)
    events = (np.ascontiguousarray(D), np.ascontiguousarray(D[:, ::-1]))
    logw = None
    for direction, eps, max_states, thr in MLD_TIERS[basis]:
        if logw is None:
            logw = mld_logw(
                g[direction],
                events[direction],
                eps=eps,
                max_states=max_states,
                compact_every=MLD_COMPACT_EVERY[basis],
            )
            continue
        sel = np.flatnonzero((np.abs(marginals_logw(logw) - 0.5) < 0.5 - thr).any(1))
        if len(sel):
            logw[sel] = np.maximum(
                logw[sel],
                mld_logw(
                    g[direction],
                    events[direction][sel],
                    eps=eps,
                    max_states=max_states,
                    compact_every=MLD_COMPACT_EVERY[basis],
                ),
            )
    return marginals_logw(logw)


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    entry = _TABLES.get(_key(circuit))
    if entry is None:
        basis = "X"
        g = _tables(circuit, basis)
    else:
        g, basis = entry
    events = np.asarray(detector_events, dtype=bool)
    assert events.ndim == 2 and events.shape[1] == g[0]["nd"]
    return (decode_marginals(g, events, basis) > 0.5).astype(np.uint8)
    #out, found = lookup(circuit, detector_events)
    ## the table is only trusted for the easiest shots; BP + OSD is more accurate on everything else
    #miss = ~(found & (np.asarray(detector_events).sum(1) <= LUT_MAX_FIRED))
    #if miss.any():
        #m = problem_info(circuit)
        #graph = build_graph(m["H"], m["L"], m["p"])
        #bposd_params = {'max_iter': 8, 'alpha': 0.6, 'always_osd': True, 'damping': 0.1, 'avg_post': True, 'lam': 400, 'lam2': 100, 'coset': 2, 'temp': 0.85, 'refine_thr': 0.3, 'refine_lam': 1000, 'refine_lam2': 300
                        #,'return_stats': False, 'return_classp': False}
        #out[miss] = decode_bp_osd(graph, np.asarray(detector_events)[miss], **bposd_params)
    #return out
