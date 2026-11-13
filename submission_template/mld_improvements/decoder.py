"""Optional frozen candidate; no CSVs, training, or prediction caches at runtime.

Use run_local.py --decoder submission_template/mld_improvements to benchmark.
The original submission_template/decoder.py remains an independent reference.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import stim
from hackathon_2026 import apply_circuit_noise, create_circuit, problem_info

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mld import marginals_logw
from mld_improvements.runtime import build_tables, infer, unique_events

MODEL_PATH = Path(os.environ.get("MLD_IMPROVEMENTS_CONFIG", str(HERE / "accuracy.json")))
MODEL = json.loads(MODEL_PATH.read_text())
COMPONENTS = MODEL["components"]


def _key(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _tables(circuit):
    info = problem_info(circuit)
    return [build_tables(info, component) for component in COMPONENTS]


_TABLES = {}
for _basis in "XZ":
    _circuit = apply_circuit_noise(create_circuit(_basis, 5), p=0.002)
    _graphs = _tables(_circuit)
    _TABLES[_key(_circuit)] = _graphs
    for _graph, _config in zip(_graphs, COMPONENTS):
        infer(_graph, np.zeros((1, 70), bool), _config)  # import-time JIT


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    raw = np.asarray(detector_events)
    if raw.ndim != 2 or raw.shape[1] != circuit.num_detectors:
        raise ValueError("expected (shots, circuit.num_detectors) detector events")
    if not np.isin(raw, (0, 1)).all():
        raise ValueError("detector events must be binary")
    if not len(raw):
        return np.empty((0, 4), np.uint8)
    graphs = _TABLES.get(_key(circuit))
    if graphs is None:
        graphs = _tables(circuit)
    D, inverse = unique_events(np.asarray(raw, dtype=bool))
    logw = None
    for graph, config in zip(graphs, COMPONENTS):
        extra, _ = infer(graph, D, config)
        logw = extra if logw is None else np.maximum(logw, extra)
    return (marginals_logw(logw) > 0.5).astype(np.uint8)[inverse]
