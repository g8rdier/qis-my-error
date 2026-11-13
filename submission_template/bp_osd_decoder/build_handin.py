"""Create a standalone, labelled-data-free decoder folder; refuses overwrites.

Only explicitly listed sources, known graph/LUT artifacts and circuit inputs
are copied. No experiment caches, CSV labels, MLP models or test data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "handin")
    parser.add_argument("--config", default="accuracy_closure.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    source_config = HERE / "candidate" / args.config
    if not source_config.is_file():
        raise SystemExit(f"missing configuration: {source_config}")
    args.out.mkdir(parents=True)
    package = args.out / "bp_osd_decoder"
    package.mkdir()
    sources = [
        "__init__.py", "bp_osd.py", "decoder_bposd.py", "build_lut.py",
        "enhanced.py", "union.py", "adaptive.py", "fused.py", "bp_checkpoints.py",
        "pool.py", "local_closure.py", "pair_index.py", "README.md",
        "EXPERIMENTS.md", "next-steps.md",
    ]
    for file in sources:
        shutil.copy2(HERE / file, package / file)
    candidate = package / "candidate"
    candidate.mkdir()
    for file in ["__init__.py", "decoder.py", "build_graphs.py", "MODELS.md"]:
        shutil.copy2(HERE / "candidate" / file, candidate / file)
    for file in (HERE / "candidate").glob("*.json"):
        shutil.copy2(file, candidate / file.name)
    shutil.copy2(source_config, candidate / "config.json")
    for folder, files in [
        ("lut", ["manifest.json", "lut_X.npz", "lut_Z.npz"]),
        ("candidate/artifacts", ["manifest.json", "graph_X.npz", "graph_Z.npz"]),
    ]:
        dest = package / folder
        dest.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(HERE / folder / file, dest / file)
    circuits = args.out / "circuits"
    circuits.mkdir()
    for basis in "XZ":
        shutil.copy2(REPO / "data" / "train" / f"{basis}_circuit.stim",
                     circuits / f"{basis}_circuit.stim")
    shutil.copy2(HERE / "requirements.txt", args.out / "requirements.txt")
    (args.out / "decoder.py").write_text(
        "from bp_osd_decoder.candidate.decoder import decode\n", encoding="utf-8")
    (args.out / "REPRODUCE.md").write_text(
        "# Standalone BP+OSD hand-in\n\n"
        "Runtime dependencies: requirements.txt. Entry: decoder.decode.\n"
        "No training/test CSVs or experiment caches are included.\n"
        "All runtime graphs/LUTs and their source generators are bundled.\n\n"
        "Rebuild tables from this folder with:\n"
        "`python bp_osd_decoder/build_lut.py --circuits circuits --shots 10000000`\n\n"
        "Rebuild graphs in the provided hackathon_2026 project environment:\n"
        "`python bp_osd_decoder/candidate/build_graphs.py --circuits circuits`\n\n"
        "The organizer's problem_info helper is build-only, not needed to decode.\n"
        "Use 48 CPU threads on the judging machine; no GPU implementation claimed.\n",
        encoding="utf-8")
    manifest = dict(
        selected_configuration=args.config,
        configuration=json.loads(source_config.read_text()),
        files={str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(args.out.rglob("*")) if p.is_file()})
    (args.out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out)
    print(f"{len(manifest['files'])} files; no sample CSVs included")


if __name__ == "__main__":
    main()
