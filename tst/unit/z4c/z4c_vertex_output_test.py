#!/usr/bin/env python3
"""Native-VC output sampling and immutable CC-output regression."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


CC_SHA256 = {
    "bin/z4c_vc_output.z4c_chi.00000.bin":
        "05d6b3c0b1a4fd2aeaa02934ce97bbe6e9e48c672d08a069ccf786cc328c1193",
    "tab/z4c_vc_output.z4c_chi.00000.tab":
        "a2363cb7f33b6571cadef29936bff2f82c764d1f63ebdb550320ce19150ee9bf",
    "vtk/z4c_vc_output.z4c_chi.00000.vtk":
        "8741361b5a3f7f5f3e236ce3b4aa066810a894cd04aaa209c5eb1c1914b3274e",
}
CC_BINARY_PAYLOAD_SHA256 = (
    "b04b85bbb0b6f4227a1795ba507d7ddc8e19159c080bc4f5e6c4800ab7dc2618"
)
# The shared deterministic-AMR helper deliberately stopped materializing the
# unused ``exercise_dynamic_vc_amr = 0`` default in unrelated runs.  That one
# parameter-header deletion changed only the binary whole-file hash; this is the
# reviewed prior value, retained so the transition cannot be mistaken for a
# numerical golden update.
CC_HISTORICAL_BINARY_FILE_SHA256 = (
    "0d76e0756575fc312663c4a5450216f709d97cc32e303a5b327096c441368c57"
)


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


def binary_payload_digest(path: Path) -> str:
    """Hash MB metadata/data after the embedded parameter header."""
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            require(line, f"binary output {path} omitted header offset")
            if line.startswith(b"  header offset="):
                header_size = int(line.split(b"=", 1)[1])
                break
        require(len(stream.read(header_size)) == header_size,
                f"binary output {path} has a truncated parameter header")
        return hashlib.sha256(stream.read()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=2)
    parser.add_argument("--active-n1", type=int, default=17)
    parser.add_argument("--active-n2", type=int, default=17)
    parser.add_argument("--active-n3", type=int, default=1)
    parser.add_argument("--multiblock-nx1", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    vc = root / "vertex"
    vc.mkdir()
    run([args.athena, "-i", args.input, "-d", str(vc)], root, True)

    table = (vc / "tab/z4c_vc_output.z4c_chi.00000.tab").read_text()
    require("# grid_sampling=vertex centering_schema=1 "
            "vertex_prolongation_order=8" in table,
            "formatted VC output omitted nodal sampling metadata")
    rows = [line.split() for line in table.splitlines() if not line.startswith("#")]
    expected_table_points = args.active_n1
    require(len(rows) == expected_table_points,
            f"VC table has {len(rows)} points, expected {expected_table_points}")
    x1_values = [float(row[2]) for row in rows]
    expected_endpoints = (0.0, 2.0) if args.dimensions == 2 else (-1.0, 1.0)
    require(min(x1_values) == expected_endpoints[0] and
            max(x1_values) == expected_endpoints[1],
            "VC formatted coordinates do not include exact x1 endpoints")

    binary = (vc / "bin/z4c_vc_output.z4c_chi.00000.bin").read_bytes()
    binary_header = binary.decode("latin-1", errors="ignore")
    require(re.search(r"(?m)^grid_sampling\s*=\s*vertex\b", binary_header) and
            re.search(r"(?m)^vertex_prolongation_order\s*=\s*8\b",
                      binary_header) and
            re.search(rf"(?m)^active_n1\s*=\s*{args.active_n1}\b", binary_header) and
            re.search(rf"(?m)^active_n2\s*=\s*{args.active_n2}\b", binary_header) and
            re.search(rf"(?m)^active_n3\s*=\s*{args.active_n3}\b", binary_header),
            "standard binary header omitted native VC sampling/extents")
    for file_id in ("adm", "con", "diag"):
        native_file = vc / f"bin/z4c_vc_output.{file_id}.00000.bin"
        require(native_file.is_file(),
                f"native VC {file_id} output was not produced")
        native_header = native_file.read_bytes().decode("latin-1", errors="ignore")
        require(re.search(r"(?m)^grid_sampling\s*=\s*vertex\b", native_header) and
                re.search(r"(?m)^vertex_prolongation_order\s*=\s*8\b",
                          native_header) and
                re.search(rf"(?m)^active_n1\s*=\s*{args.active_n1}\b", native_header) and
                re.search(rf"(?m)^active_n3\s*=\s*{args.active_n3}\b", native_header),
                f"native VC {file_id} output lost point-sampling metadata")

    vtk = (vc / "vtk/z4c_vc_output.z4c_chi.00000.vtk").read_bytes()
    dimensions_record = (f"DIMENSIONS {args.active_n1} {args.active_n2} "
                         f"{args.active_n3}").encode()
    point_count = args.active_n1 * args.active_n2 * args.active_n3
    require(b"grid_sampling=vertex" in vtk and
            b"vertex_prolongation_order=8" in vtk and
            dimensions_record in vtk and
            f"POINT_DATA {point_count}".encode() in vtk and b"CELL_DATA" not in vtk,
            "legacy VTK did not encode the native VC field as POINT_DATA")

    if args.dimensions == 3:
        history_rows = [[float(value) for value in line.split()]
                        for line in (vc / "z4c_vc_output.z4c.user.hst").read_text(
                            encoding="utf-8").splitlines()
                        if line.strip() and not line.startswith("#")]
        require(history_rows and
                all(math.isfinite(value) for row in history_rows for value in row) and
                all(row[10] == 8.0 for row in history_rows),
                "3D native VC history does not integrate the exact Cartesian volume")

    if args.dimensions == 2:
        cc = root / "cell"
        cc.mkdir()
        run([args.athena, "-i", args.input, "-d", str(cc),
             "z4c/grid_centering=cell"], root, True)
        for relative, expected in CC_SHA256.items():
            observed = digest(cc / relative)
            require(observed == expected,
                    f"CC output regression for {relative}: {observed} != {expected}")
        cc_binary = cc / "bin/z4c_vc_output.z4c_chi.00000.bin"
        require(binary_payload_digest(cc_binary) == CC_BINARY_PAYLOAD_SHA256,
                "CC binary MeshBlock/numerical payload changed from the reviewed "
                "pre-schedule checkpoint")
        require(CC_SHA256["bin/z4c_vc_output.z4c_chi.00000.bin"] !=
                CC_HISTORICAL_BINARY_FILE_SHA256,
                "metadata-only CC binary transition was accidentally hidden")

    rejected = root / "rejected_multiblock_vtk"
    rejected.mkdir()
    output = run([args.athena, "-i", args.input, "-d", str(rejected),
                  f"mesh/nx1={args.multiblock_nx1}"], root, False)
    require("cannot de-duplicate native VC data across multiple MeshBlocks" in output,
            "unsupported multi-MeshBlock native VC VTK did not fail clearly")

    multiblock = root / "multiblock_table"
    multiblock.mkdir()
    run([args.athena, "-i", args.input, "-d", str(multiblock),
         f"mesh/nx1={args.multiblock_nx1}", "output3/file_type=bin"],
        root, True)
    multiblock_table = (
        multiblock / "tab/z4c_vc_output.z4c_chi.00000.tab").read_text()
    multiblock_rows = [line.split() for line in multiblock_table.splitlines()
                       if not line.startswith("#")]
    multiblock_coordinates = [(float(row[2]),) for row in multiblock_rows]
    expected_multiblock_points = 2 * (args.active_n1 - 1) + 1
    require(len(multiblock_coordinates) == expected_multiblock_points and
            len(set(multiblock_coordinates)) == expected_multiblock_points,
            "native VC table did not de-duplicate a shared MeshBlock endpoint")
    require(min(point[0] for point in multiblock_coordinates) == expected_endpoints[0] and
            max(point[0] for point in multiblock_coordinates) == expected_endpoints[1],
            "de-duplicated VC table omitted a physical endpoint")

    print(f"PASS: native VC {args.dimensions}D binary/table/VTK sampling, "
          "table de-duplication, "
          "and frozen CC output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
