"""Isolated candidate hand-in entry point; does NOT replace the ML decoder.

Import-time graph/LUT loading and JIT warmup are outside the timed decode calls,
as explicitly permitted by the competition rules. No repeated-call prediction
cache: every decode call computes its own outputs. Same-syndrome deduplication
is confined to the current batch and uses no labels.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BP_DIR = HERE.parent
sys.path.insert(0, str(BP_DIR.parent))

import numpy as np
import stim
from bp_osd_decoder.enhanced import decode_enhanced
from bp_osd_decoder.union import decode_union
from bp_osd_decoder.adaptive import decode_adaptive, run_refiner
from bp_osd_decoder.fused import decode_fused

CONFIG = json.loads((HERE / os.environ.get("BPOSD_CONFIG", "config.json")).read_text())
MANIFEST = json.loads((HERE / "artifacts" / "manifest.json").read_text())
ENGINE = CONFIG.get("engine", "enhanced")


def _compute(graph, D, settings):
    fn = {"enhanced": decode_enhanced, "union": decode_union,
          "adaptive": decode_adaptive, "fused": decode_fused}[ENGINE]
    return fn(graph, D, **settings)


STATE = {}
for sha, entry in MANIFEST.items():
    with np.load(HERE / "artifacts" / entry["file"], allow_pickle=False) as z:
        graph = {k: z[k] for k in z.files}
    graph["m"], graph["n"] = int(graph["m"]), int(graph["n"])
    with np.load(BP_DIR / "lut" / entry["lut_file"], allow_pickle=False) as z:
        keys = np.ascontiguousarray(z["keys"]).view("V9").ravel()
        labels = z["predict"].copy()
    STATE[sha] = (graph, keys, labels)
    # Warm both main/refinement paths without seeing any sample labels.
    cfg = CONFIG.get("basis_parameters", {}).get(entry["basis"], CONFIG["parameters"])
    dummy = np.zeros((1, graph["m"]), dtype=np.uint8)
    _compute(graph, dummy, cfg)
    if ENGINE == "enhanced":
        ref = dict(cfg)
        ref.update(lam=cfg.get("refine_lam", 1000),
                   lam2=cfg.get("refine_lam2", 300),
                   lam3=cfg.get("refine_lam3", 0), refine_thr=0)
        decode_enhanced(graph, dummy, **ref)
    elif ENGINE in ("adaptive", "fused"):
        run_refiner(graph, dummy, cfg["refiner"])


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    D = np.asarray(detector_events, dtype=bool)
    if D.ndim != 2 or D.shape[1] != 70:
        raise ValueError("expected a (shots, 70) detector array")
    if not len(D):
        return np.empty((0, 4), dtype=np.uint8)
    sha = hashlib.sha256(str(circuit).encode()).hexdigest()
    if sha not in STATE:
        raise ValueError("unrecognized circuit; regenerate candidate/artifacts")
    graph, keys, labels = STATE[sha]
    packed = np.ascontiguousarray(np.packbits(D, axis=1, bitorder="little"))
    query = packed.view("V9").ravel()
    idx = np.searchsorted(keys, query)
    safe = np.minimum(idx, len(keys) - 1)
    found = (idx < len(keys)) & (keys[safe] == query)
    use_lut = found & (D.sum(1) <= CONFIG["lut_cutoff"])
    out = np.unpackbits(labels[safe, None], axis=1, count=4, bitorder="little")
    miss = np.flatnonzero(~use_lut)
    if len(miss):
        entry = MANIFEST[sha]
        cfg = CONFIG.get("basis_parameters", {}).get(entry["basis"], CONFIG["parameters"])
        if CONFIG.get("deduplicate", True):
            _, first, inverse = np.unique(query[miss], return_index=True, return_inverse=True)
            pred = _compute(graph, D[miss[first]], cfg)
            out[miss] = pred[inverse]
        else:
            out[miss] = _compute(graph, D[miss], cfg)
    return out.astype(np.uint8, copy=False)
