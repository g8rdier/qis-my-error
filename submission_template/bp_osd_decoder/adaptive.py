"""Spend extra candidate search only on shots whose first-pass marginals are uncertain.

No data/labels or randomness. Both initial and refinement algorithms compute
their own original-prior candidate probabilities. The second pass replaces
the first on routed shots; no unjustified mixing of normalized masses.
"""
import numpy as np

from bp_osd_decoder.enhanced import decode_enhanced
from bp_osd_decoder.union import decode_union

BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)


def run_refiner(graph, D, settings):
    settings = dict(settings)
    union = settings.pop("union", False)
    settings.pop("enhanced", None)
    fn = decode_union if union else decode_enhanced
    return fn(graph, D, return_classp=True, **settings)


def decode_adaptive(graph, detector_events, initial=None, refiner=None,
                    route_threshold=0.2, return_classp=False):
    if not 0 <= route_threshold <= 0.5:
        raise ValueError("route_threshold must be between 0 and 0.5")
    initial = initial or {"max_iter": 5}
    refiner = refiner or {"union": True, "lam": 600, "lam2": 200,
                          "schedules": [[3, .8, .5], [5, .8, .5], [8, .8, .5]]}
    D = np.ascontiguousarray(detector_events, dtype=np.uint8)
    pred, cp = decode_enhanced(graph, D, return_classp=True, **initial)
    marg = cp @ BITS
    use = ((marg > route_threshold) & (marg < 1 - route_threshold)).any(1)
    if use.any():
        pp, cc = run_refiner(graph, D[use], refiner)
        pred[use], cp[use] = pp, cc
    return (pred, cp) if return_classp else pred
