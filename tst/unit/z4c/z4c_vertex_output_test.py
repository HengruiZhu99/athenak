#!/usr/bin/env python3
"""Native-VC output sampling and immutable CC-output regression."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess


CC_SHA256 = {
    "bin/z4c_vc_output.z4c_chi.00000.bin":
        "0d76e0756575fc312663c4a5450216f709d97cc32e303a5b327096c441368c57",
    "tab/z4c_vc_output.z4c_chi.00000.tab":
        "a2363cb7f33b6571cadef29936bff2f82c764d1f63ebdb550320ce19150ee9bf",
    "vtk/z4c_vc_output.z4c_chi.00000.vtk":
        "8741361b5a3f7f5f3e236ce3b4aa066810a894cd04aaa209c5eb1c1914b3274e",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], cwd: Path, success: bool) -> str:
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "2", "OMP_PROC_BIND": "false"})
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            check=False, timeout=45,
                            env=environment)
    output = result.stdout + "\n" + result.stderr
    require((result.returncode == 0) == success,
            f"unexpected exit {result.returncode} for {' '.join(command)}:\n{output}")
    return output


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    root = Path(args.work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    vc = root / "vertex"
    vc.mkdir()
    run([args.athena, "-i", args.input, "-d", str(vc)], root, True)

    table = (vc / "tab/z4c_vc_output.z4c_chi.00000.tab").read_text()
    require("# grid_sampling=vertex centering_schema=1" in table,
            "formatted VC output omitted nodal sampling metadata")
    rows = [line.split() for line in table.splitlines() if not line.startswith("#")]
    require(len(rows) == 17, f"VC table has {len(rows)} points, expected 17")
    require(float(rows[0][2]) == 0.0 and float(rows[-1][2]) == 2.0,
            "VC formatted coordinates do not include exact physical endpoints")

    binary = (vc / "bin/z4c_vc_output.z4c_chi.00000.bin").read_bytes()
    binary_header = binary.decode("latin-1", errors="ignore")
    require(re.search(r"(?m)^grid_sampling\s*=\s*vertex\b", binary_header) and
            re.search(r"(?m)^active_n1\s*=\s*17\b", binary_header) and
            re.search(r"(?m)^active_n2\s*=\s*17\b", binary_header),
            "standard binary header omitted native VC sampling/extents")
    for file_id in ("adm", "con", "diag"):
        native_file = vc / f"bin/z4c_vc_output.{file_id}.00000.bin"
        require(native_file.is_file(),
                f"native VC {file_id} output was not produced")
        native_header = native_file.read_bytes().decode("latin-1", errors="ignore")
        require(re.search(r"(?m)^grid_sampling\s*=\s*vertex\b", native_header) and
                re.search(r"(?m)^active_n1\s*=\s*17\b", native_header),
                f"native VC {file_id} output lost point-sampling metadata")

    vtk = (vc / "vtk/z4c_vc_output.z4c_chi.00000.vtk").read_bytes()
    require(b"grid_sampling=vertex" in vtk and b"DIMENSIONS 17 17 1" in vtk and
            b"POINT_DATA 289" in vtk and b"CELL_DATA" not in vtk,
            "legacy VTK did not encode the native VC field as POINT_DATA")

    cc = root / "cell"
    cc.mkdir()
    run([args.athena, "-i", args.input, "-d", str(cc),
         "z4c/grid_centering=cell"], root, True)
    for relative, expected in CC_SHA256.items():
        observed = digest(cc / relative)
        require(observed == expected,
                f"CC output regression for {relative}: {observed} != {expected}")

    rejected = root / "rejected_multiblock_vtk"
    rejected.mkdir()
    output = run([args.athena, "-i", args.input, "-d", str(rejected),
                  "mesh/nx1=32"], root, False)
    require("cannot de-duplicate native VC data across multiple MeshBlocks" in output,
            "unsupported multi-MeshBlock native VC VTK did not fail clearly")

    multiblock = root / "multiblock_table"
    multiblock.mkdir()
    run([args.athena, "-i", args.input, "-d", str(multiblock),
         "mesh/nx1=32", "output3/file_type=bin"],
        root, True)
    multiblock_table = (
        multiblock / "tab/z4c_vc_output.z4c_chi.00000.tab").read_text()
    multiblock_rows = [line.split() for line in multiblock_table.splitlines()
                       if not line.startswith("#")]
    multiblock_coordinates = [float(row[2]) for row in multiblock_rows]
    require(len(multiblock_coordinates) == 33 and
            len(set(multiblock_coordinates)) == 33,
            "native VC table did not de-duplicate a shared MeshBlock endpoint")
    require(multiblock_coordinates[0] == 0.0 and
            multiblock_coordinates[-1] == 2.0,
            "de-duplicated VC table omitted a physical endpoint")

    print("PASS: native VC binary/table/VTK sampling, table de-duplication, "
          "and frozen CC output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
