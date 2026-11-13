"""Report ONLY the predeclared reference/shuffled-sweep audit combination."""
import json

import numpy as np

from search import HERE, marginals_logw, score


def main():
    prefix = "audit_954100_50000"
    probabilities, labels, references, groups = [], [], [], []
    for family in range(3):
        for basis in "XZ":
            values = {}
            for name in ("reference", "shuffle_checks"):
                with np.load(HERE / "cache" / f"{prefix}_{basis}_{family}_{name}.npz") as z:
                    values[name] = z["logw"]
                    truth = z["labels"]
            ref = marginals_logw(values["reference"]) > 0.5
            prob = marginals_logw(np.maximum(values["reference"], values["shuffle_checks"]))
            probabilities.append(prob)
            labels.append(truth)
            references.append(ref)
            groups.append(score("reference+shuffle_checks/max", prob, ref, truth, 0,
                                basis=basis, family=family))
    result = score("reference+shuffle_checks/max", np.concatenate(probabilities),
                   np.concatenate(references), np.concatenate(labels), None, groups=groups)
    target = HERE / "results" / f"{prefix}.secondary.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
