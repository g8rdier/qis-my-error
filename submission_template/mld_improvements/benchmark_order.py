"""Five-run check after prediction-preserving deduplication order optimization."""
import json
import sys

import numpy as np

from benchmark import HERE, ROOT, run


def main():
    records = []
    for name in ("candidate", "accuracy"):
        tag = f"ordered_fresh_{name}"
        out = HERE / "cache" / f"{tag}.csv"
        command = [sys.executable, str(HERE.parent / "run_local.py"), "--decoder", str(HERE),
                   "--problem-set", str(ROOT / "data/mld_audit_runtime"), "--threads", "16",
                   "--runs", "5", "--out", str(out)]
        run(command, tag, HERE / f"{name}.json")
        options = dict(delimiter=",", skiprows=1, usecols=(1, 2, 3, 4), dtype=np.uint8)
        old = np.loadtxt(HERE / "cache" / f"fresh_{name}.csv", **options)
        new = np.loadtxt(out, **options)
        np.testing.assert_array_equal(new, old)
        records.append(dict(name=name, bits=int(new.size), changed_bits=0))
    (HERE / "results/order_equivalence.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
