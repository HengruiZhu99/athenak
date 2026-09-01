#!/usr/bin/env python3
"""Exact implicit/default versus explicit cell-centered Z4c regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


HISTORY_SHA256 = "e2bc8e6aa86cbf554a7a108e5a7e64320bf716540bd8539b66bb6bc6014ca6cf"
TIMESTEP_SHA256 = "dad954f5938eea76aca74493ec5bd1ac8c66cdc67ac7ad24225988c19e5e3037"
RESTART_MARKER = b"<par_end>\n"
RESTART_OUTPUT = "\n<output4>\nfile_type = rst\ndcycle = 1\nfile_number = 0\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def history_table(data: bytes) -> tuple[list[str], list[list[str]]]:
    text = data.decode("utf-8")
    header = next(line for line in text.splitlines() if line.startswith("#  [1]="))
    labels = re.findall(r"\[\d+\]=([^ ]+)", header)
    require(len(labels) == len(set(labels)), "history labels must be unique")
    rows = [line.split() for line in text.splitlines()
            if line.strip() and not line.startswith("#")]
    require(rows and all(len(row) == len(labels) for row in rows),
            "history rows do not match the indexed header")
    return labels, rows


def run_case(athena: Path, input_text: str, case: Path) -> None:
    case.mkdir(parents=True)
    input_path = case / "input.athinput"
    input_path.write_text(input_text, encoding="utf-8")
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(
        [str(athena), "-i", str(input_path)], cwd=case, env=environment,
        text=True, capture_output=True, check=False, timeout=60)
    output = result.stdout + "\n" + result.stderr
    require(result.returncode == 0,
            f"CC selector case failed ({result.returncode}):\n{output}")
    require("cycle=1" in output, "CC selector case did not complete one RK cycle")


def compare_binary(left: dict, right: dict, label: str) -> str:
    import numpy as np  # pylint: disable=import-outside-toplevel

    scalar_keys = (
        "time", "cycle", "var_names", "n_mbs", "nx1_mb", "nx2_mb", "nx3_mb",
        "nx1_out_mb", "nx2_out_mb", "nx3_out_mb", "Nx1", "Nx2", "Nx3",
        "x1min", "x1max", "x2min", "x2max", "x3min", "x3max", "nvars",
    )
    for key in scalar_keys:
        require(left[key] == right[key], f"{label}: binary metadata {key} changed")
    for key in ("mb_index", "mb_logical", "mb_geometry"):
        require(np.array_equal(left[key], right[key]),
                f"{label}: binary MeshBlock metadata {key} changed")
    for name in left["var_names"]:
        require(name in right["mb_data"], f"{label}: missing variable {name}")
        require(np.array_equal(left["mb_data"][name], right["mb_data"][name]),
                f"{label}: numerical array {name} changed")

    # Metadata-independent canonical numerical payload.  The binary file header
    # legitimately differs because the explicit input contains one extra line.
    digest = hashlib.sha256()
    digest.update(struct.pack("<dq", left["time"], left["cycle"]))
    for key in ("mb_index", "mb_logical", "mb_geometry"):
        digest.update(np.asarray(left[key]).tobytes(order="C"))
    for name in left["var_names"]:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(np.asarray(left["mb_data"][name]).tobytes(order="C"))
    return digest.hexdigest()


def restart_payload(path: Path) -> bytes:
    data = path.read_bytes()
    require(data.count(RESTART_MARKER) == 1, f"invalid restart marker in {path}")
    return data.split(RESTART_MARKER, 1)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_dir / "vis/python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel

    base = args.input.read_text(encoding="utf-8")
    require("grid_centering" not in base,
            "authoritative implicit-CC input unexpectedly contains grid_centering")
    require(base.count("<z4c>\n") == 1, "expected exactly one <z4c> block")
    implicit_text = base.rstrip() + "\n" + RESTART_OUTPUT
    explicit_text = implicit_text.replace(
        "<z4c>\n", "<z4c>\ngrid_centering = cell\n", 1)

    root = args.work_dir
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    implicit = root / "implicit"
    explicit = root / "explicit"
    run_case(args.athena, implicit_text, implicit)
    run_case(args.athena, explicit_text, explicit)

    basename = "z4c_kerr_half_plane_init"
    history_name = f"{basename}.z4c.user.hst"
    timestep_name = "z4c_timestep_contract.csv"
    evidence = {
        "schema": 1,
        "disposition": "pass",
        "history_sha256": sha256_file(implicit / history_name),
        "timestep_contract_sha256": sha256_file(implicit / timestep_name),
        "binary_payload_sha256": {},
        "restart_payload_sha256": {},
    }
    for name, expected in ((history_name, HISTORY_SHA256),
                           (timestep_name, TIMESTEP_SHA256)):
        left = (implicit / name).read_bytes()
        right = (explicit / name).read_bytes()
        require(left == right, f"implicit/explicit {name} bytes changed")
        require(sha256_bytes(left) == expected,
                f"authoritative {name} SHA-256 changed")

    for variable in ("z4c", "con"):
        for output in (0, 1, 2):
            relative = Path("bin") / f"{basename}.{variable}.{output:05d}.bin"
            require((implicit / relative).is_file() and (explicit / relative).is_file(),
                    f"missing paired binary output {relative}")
            digest = compare_binary(
                bin_convert.read_binary(str(implicit / relative)),
                bin_convert.read_binary(str(explicit / relative)), str(relative))
            evidence["binary_payload_sha256"][str(relative)] = digest

    implicit_restarts = sorted((implicit / "rst").glob(f"{basename}.*.rst"))
    explicit_restarts = sorted((explicit / "rst").glob(f"{basename}.*.rst"))
    require(implicit_restarts and
            [path.name for path in implicit_restarts] ==
            [path.name for path in explicit_restarts],
            "implicit/explicit restart inventories differ")
    for left_path, right_path in zip(implicit_restarts, explicit_restarts):
        left_payload = restart_payload(left_path)
        right_payload = restart_payload(right_path)
        require(left_payload == right_payload,
                f"restart numerical payload changed for {left_path.name}")
        evidence["restart_payload_sha256"][left_path.name] = sha256_bytes(left_payload)

    (root / "cc_selector_equivalence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS: implicit/default and explicit cell-centered Z4c are exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
