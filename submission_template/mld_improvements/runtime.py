"""Reusable runtime for frozen MLD configurations (no scoring/data access)."""
import numpy as np

from mld import build_mld, marginals_logw, mld_logw


def detector_permutation(nd, config):
    mode = config.get("detector_order", "time")
    permutation = np.arange(nd)
    if mode == "time":
        return permutation
    if nd != 70:
        raise ValueError("within-round detector permutations require 70 detectors")
    blocks = [(0, 7), (7, 21), (21, 35), (35, 49), (49, 63), (63, 70)]
    if mode == "reverse_checks":
        return np.concatenate([permutation[a:b][::-1] for a, b in blocks])
    if mode == "shuffle_checks":
        rng = np.random.default_rng(42071)
        return np.concatenate([rng.permutation(permutation[a:b]) for a, b in blocks])
    raise ValueError(f"unknown detector order: {mode}")


def build_tables(info, config):
    permutation = detector_permutation(info["H"].shape[0], config)
    return tuple(build_mld(info["H"][permutation[::(-1 if direction else 1)]].copy(),
                           info["L"], info["p"], hexp=config["hexp"],
                           heuristic=config["heuristic"],
                           mechanism_order=config["order"],
                           merge_factor=config.get("merge_factor", 1.0))
                 for direction in (0, 1))


def infer(graphs, D, config):
    """Absolute outcome log weights and per-tier routed shot counts."""
    logw = None
    routed = []
    permutation = detector_permutation(D.shape[1], config)
    for direction, eps, states, threshold in config["tiers"]:
        sel = np.arange(len(D)) if logw is None else np.flatnonzero(
            (np.abs(marginals_logw(logw) - 0.5) < 0.5 - threshold).any(1))
        routed.append(int(len(sel)))
        if not len(sel):
            continue
        events = D[sel][:, permutation[::(-1 if direction else 1)]]
        extra = mld_logw(graphs[direction], events, eps=eps, max_states=states,
                         compact_every=config["compact"])
        if logw is None:
            logw = extra
        else:
            logw[sel] = np.maximum(logw[sel], extra)
    if logw is None:
        logw = np.empty((0, 16), np.float64)
    return logw, routed


def unique_events(D):
    """Deduplicate within a call, retaining first-occurrence shot order.

    Sorting syndromes also sorts easy/hard work into contiguous Numba thread
    ranges. Preserve input order for load balance; do not change any prediction.
    """
    packed = np.ascontiguousarray(np.packbits(D, axis=1, bitorder="little"))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return D[first[order]], remap[inverse]
