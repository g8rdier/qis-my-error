"""Near-maximum-likelihood decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README). Importing the decoder, and anything it builds when imported, is
not timed.

Every shot is decoded by the pruned dynamic programme in mld.py, which computes the weight of each of the 16 logical
outcomes and predicts each logical bit by its marginal (the optimal decision for the per-bit LER).

Pruning only removes terms from an outcome's weight, so every sweep gives a lower bound on it, and sweeps are
combined by keeping the largest bound per outcome. A sweep over the error mechanisms forward in time and one
backward in time prune different error vectors, so their combination is closer to the exact weights than one sweep
with a tighter threshold that costs the same.

Their disagreement also shows where the pruning mattered. Every shot gets a forward and a backward sweep with a
loose threshold. A shot is swept again in both directions with a tighter threshold if the sweeps (or their
combination) differ on some logical bit's probability by an amount that matters for the decision, that is, relative
to how far that probability is from 1/2; this is repeated once more with a much tighter threshold. About 14% of the
shots get the second pair of sweeps and 2.5% the third. Against an exact (unpruned) decoder the remaining loss is
below one bit in a million.

At import we precompute the decoder tables for the competition's two circuits (circuits/, copied from the Data tab)
and compile the numba kernel, so decode() only has to look the tables up by a hash of the circuit text. A circuit
that is not one of these is handled too, by building its tables inside decode(). The folder needs nothing outside
itself and requirements.txt.
"""
import hashlib
import os

import numpy as np
import stim

from mld import build_mld, marginals_logw, mld_logw

HERE = os.path.dirname(os.path.abspath(__file__))
MLD_TIERS = {
    "X": (
        (0, 2.962962962962963e-05, 2816, 0.0),
        (1, 3e-05, 2048, 0.125),
        (0, 4.5e-05, 3072, 0.25),
    ),
    "Z": (
        (0, 6.666666666666667e-05, 2560, 0.0),
        (1, 6.75e-05, 3072, 0.125),
        (0, 3e-05, 2304, 0.3750000000000001),
    ),
}
MLD_COMPACT_EVERY = {"X": 32, "Z": 32}
MLD_HEXP = {"X": 0.6, "Z": 0.6}

def problem_info(circuit: stim.Circuit):
    """H (detectors x mechanisms), L (observables x mechanisms) and p (probability of each mechanism). Error
    mechanisms with the same detectors and observables are merged into one column."""
    dem = circuit.detector_error_model()
    columns, priors = {}, []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        key, p = (frozenset(detectors), frozenset(observables)), instruction.args_copy()[0]
        if key in columns:  # same effect as an earlier mechanism: either one fires, not both
            q = priors[columns[key]]
            priors[columns[key]] = q * (1 - p) + p * (1 - q)
        else:
            columns[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), dtype=np.uint8)
    for (detectors, observables), j in columns.items():
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return H, L, np.array(priors)


def _key(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _tables(circuit: stim.Circuit, basis: str):
    H, L, p = problem_info(circuit)
    # backward sweep: the same decoder on the detectors in reverse order
    return (
        build_mld(H, L, p, hexp=MLD_HEXP[basis]),
        build_mld(H[::-1].copy(), L, p, hexp=MLD_HEXP[basis]),
    )


_TABLES = {}
for _basis in "XZ":
    _c = stim.Circuit.from_file(os.path.join(HERE, "circuits", f"{_basis}_circuit.stim"))
    _TABLES[_key(_c)] = (_tables(_c, _basis), _basis)
for _g, _basis in _TABLES.values():  # compile (or load from numba's cache) the kernel before the clock starts
    for _dir, _eps, _max_states, _threshold in MLD_TIERS[_basis]:
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
    for direction, eps, max_states, threshold in MLD_TIERS[basis]:
        selected = np.arange(len(D)) if logw is None else np.flatnonzero(
            (np.abs(marginals_logw(logw) - 0.5) < 0.5 - threshold).any(axis=1)
        )
        if not len(selected):
            continue
        refined = mld_logw(
            g[direction],
            events[direction][selected],
            eps=eps,
            max_states=max_states,
            compact_every=MLD_COMPACT_EVERY[basis],
        )
        if logw is None:
            logw = refined
        else:
            logw[selected] = np.maximum(logw[selected], refined)
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
