"""Post-audit diagnostics only: no parameter fitting, selection or thresholds."""
import json
from pathlib import Path

import numpy as np
import stim

from search import HERE, ROOT, marginals_logw


def main():
    values = {key: [] for key in ("risk", "wrong", "weight", "changed", "new_wrong")}
    failed = 0
    for family in range(3):
        for offset, basis in enumerate("XZ"):
            circuit = stim.Circuit.from_file(str(ROOT / f"data/train/{basis}_circuit.stim"))
            D, labels = circuit.compile_detector_sampler(seed=954100 + 2 * family + offset).sample(
                50000, separate_observables=True)
            with np.load(HERE / "cache" / f"audit_954100_50000_{basis}_{family}_reference.npz") as z:
                np.testing.assert_array_equal(labels, z["labels"])
                logw = z["logw"]
            with np.load(HERE / "cache" / f"audit_954100_50000_{basis}_{family}_shuffle_checks.npz") as z:
                combo = np.maximum(logw, z["logw"])
            prob = marginals_logw(logw)
            pred = prob > 0.5
            other = marginals_logw(combo) > 0.5
            failed += int(np.count_nonzero(~np.isfinite(logw).any(1)))
            values["risk"].append(np.minimum(prob, 1 - prob))
            values["wrong"].append(pred != labels)
            values["weight"].append(D.sum(1))
            values["changed"].append(other != pred)
            values["new_wrong"].append(other != labels)
    v = {key: np.concatenate(parts) for key, parts in values.items()}
    risk, wrong = v["risk"], v["wrong"]
    result = dict(
        shots=len(risk), failed_shots=failed,
        mean_posterior_risk=float(risk.mean()), observed_ler=float(wrong.mean()),
        changed_bits=int(v["changed"].sum()),
        changed_shots=int(v["changed"].any(1).sum()),
        correct_to_wrong=int((v["changed"] & v["new_wrong"]).sum()),
        wrong_to_correct=int((v["changed"] & ~v["new_wrong"]).sum()),
        confidence_bins=[], syndrome_weight_bins=[])
    for lo, hi in zip([0, .001, .01, .05, .15, .35], [.001, .01, .05, .15, .35, .500001]):
        sel = (risk >= lo) & (risk < hi)
        result["confidence_bins"].append(dict(
            risk_range=[lo, min(hi, .5)], bits=int(sel.sum()),
            expected_error=float(risk[sel].mean()),
            observed_error=float(wrong[sel].mean())))
    for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 15), (15, 71)]:
        sel = (v["weight"] >= lo) & (v["weight"] < hi)
        result["syndrome_weight_bins"].append(dict(
            weight_range=[lo, hi - 1], shots=int(sel.sum()),
            expected_error=float(risk[sel].mean()), observed_error=float(wrong[sel].mean())))
    target = HERE / "results/audit_diagnostics.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
