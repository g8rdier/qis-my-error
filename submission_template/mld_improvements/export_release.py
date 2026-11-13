"""Export the frozen ensemble's own kernel, not the experimental framework."""
import hashlib
from pathlib import Path
import subprocess
import sys

REVISION = "962f067efdfe5ffd7dd8f43f1e4faf99ee42ac1e"
ORIGINAL_SHA256 = "7a241f47146abb61d0ff9be06efb60bb16f45686e11bcbb57d052f221c0363d7"


def main():
    root = Path(__file__).resolve().parents[2]
    output = Path(sys.argv[1]).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = subprocess.check_output(
        ["git", "show", f"{REVISION}:submission_template/mld.py"], cwd=root).decode()
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("original kernel checksum mismatch")
    ctz = '''# Self-contained unsigned bit index; Python-int construction avoids overflow warnings.
_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_CTZ_TABLE = np.zeros(64, np.int64)
for _i in range(64):
    _CTZ_TABLE[(((1 << _i) * int(_DEBRUIJN)) & ((1 << 64) - 1)) >> 58] = _i


@njit(cache=True)
def _ctz(x):
    return _CTZ_TABLE[((x & (~x + np.uint64(1))) * _DEBRUIJN) >> np.uint64(58)]
'''
    source = source.replace("from bp_osd import _ctz", ctz)
    # Reject impossible unsupported detector rows, as in the verified research kernel.
    source = source.replace("    mx = ninf\n    for i in range(ns):",
                            "    mx = ninf\n"
                            "    detector_hi = (np.uint64(1) << np.uint64(nd - 64)) - np.uint64(1)\n"
                            "    for i in range(ns):")
    source = source.replace("if sc[i] > ninf and lw[i] > mx:",
                            "if ka[i] == 0 and (kb[i] & detector_hi) == 0 and sc[i] > ninf and lw[i] > mx:")
    source = source.replace("            if sc[i] > ninf:\n",
                            "            if ka[i] == 0 and (kb[i] & detector_hi) == 0 and sc[i] > ninf:\n")
    start = source.index("def mld_classp(")
    end = source.index("def mld_logw(", start)
    source = source[:start] + source[end:]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8", newline="\n")
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
