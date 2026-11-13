"""Adapter: the lut-bp-osd-decoder branch's decode logic, import-safe in this folder.

Original: submission_template/decoder.py on branch lut-bp-osd-decoder
(commits 1f4f334..94f0fc3). Logic unchanged; only the LUT directory and the
import path are adapted so this lives cleanly separated from the ML decoders.
`decode_bposd()` has the same (circuit, events) -> (n,4) contract as decoder.decode.
"""
import hashlib
import json
import os

import numpy as np
import stim
from hackathon_2026 import problem_info

from bp_osd_decoder.bp_osd import build_graph, decode_bp_osd

LUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lut")
N_DET, N_OBS, KEY_BYTES = 70, 4, 9
LUT_MAX_FIRED = 4  # use the lookup table only for shots with at most this many detectors fired


def _load_table(circuit: stim.Circuit):
    with open(os.path.join(LUT_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    entry = manifest.get(hashlib.sha256(str(circuit).encode()).hexdigest())
    if entry is None:
        raise ValueError("no lookup table for this circuit; rebuild with build_lut.py")
    data = np.load(os.path.join(LUT_DIR, entry["file"]))
    return data["keys"].view(f"V{KEY_BYTES}").ravel(), data["predict"]


def lookup(circuit: stim.Circuit, detector_events: np.ndarray):
    """Returns (predictions of shape (shots, 4), bool mask of shots found in the table)."""
    keys, predict = _load_table(circuit)
    events = np.asarray(detector_events, dtype=bool)
    assert events.ndim == 2 and events.shape[1] == N_DET
    query = np.ascontiguousarray(np.packbits(events, axis=1, bitorder="little")).view(f"V{KEY_BYTES}").ravel()
    idx = np.searchsorted(keys, query)
    idx[idx == len(keys)] = 0
    found = keys[idx] == query
    packed = np.where(found, predict[idx], 0).astype(np.uint8)
    out = np.unpackbits(packed[:, None], axis=1, count=N_OBS, bitorder="little")
    return out, found


def decode_bposd(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    out, found = lookup(circuit, detector_events)
    # the table is only trusted for the easiest shots; BP + OSD is more accurate on everything else
    miss = ~(found & (np.asarray(detector_events).sum(1) <= LUT_MAX_FIRED))
    if miss.any():
        m = problem_info(circuit)
        graph = build_graph(m["H"], m["L"], m["p"])
        out[miss] = decode_bp_osd(graph, np.asarray(detector_events)[miss])
    return out
