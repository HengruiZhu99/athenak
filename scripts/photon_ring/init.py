"""Create a new photon-ring survey, with all run state outside the AthenaK checkout."""

import argparse
from pathlib import Path
import json
import os
import shutil

from campaign import CODE, executable_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--axis", choices=["width", "spin"], required=True)
    p.add_argument(
        "--name", required=True, help="distinct Slurm job prefix for this campaign"
    )
    p.add_argument("--profile", type=Path, default=CODE / "profiles/slurm.json")
    p.add_argument("--site", type=Path, help="optional batch environment setup script")
    p.add_argument("--athena", type=Path, required=True)
    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="IrisK photon_torus Python directory",
    )
    p.add_argument(
        "--operators",
        type=Path,
        required=True,
        help="IrisK Lobatto executable or cached operator reader",
    )
    p.add_argument("--reference", type=Path, default=CODE / "inputs/reference.json")
    args = p.parse_args()
    source = args.calibration.resolve(strict=True)
    operator = args.operators.resolve(strict=True)
    exe = args.athena.resolve(strict=True)
    if not all(os.access(f, os.X_OK) for f in [operator, exe]):
        p.error("both --athena and --operators must be executable")
    modules = [source / f for f in ("measure.py", "solve.py", "spectral.py")]
    for f in modules:
        f.resolve(strict=True)
    cluster = json.loads(args.profile.read_text())
    cluster["executable"] = str(exe)
    cluster["executable_sha256"] = executable_digest(
        exe, exe.stat().st_mtime_ns, exe.stat().st_size
    )
    scan = json.loads((CODE / f"inputs/{args.axis}.json").read_text())
    scan["name"] = args.name
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases").mkdir()
    (root / "calibration").mkdir()
    for f in modules:
        shutil.copy2(f, root / "calibration" / f.name)
    (root / "calibration/read.sh").symlink_to(operator)
    for name in ["smr.in", "finder-restart.in", "seed.json", "mesh-bounds.json"]:
        shutil.copy2(CODE / "inputs" / name, root / name)
    shutil.copy2(args.reference, root / "reference.json")
    if args.site:
        shutil.copy2(args.site, root / "site.sh")
    for name, data in [("scan", scan), ("cluster", cluster)]:
        (root / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
