#!/usr/bin/env python3
"""Exact half-plane Cartoon fresh/restart continuation regression."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


MARKER = b"<par_end>\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            check=False, timeout=30)
    output = result.stdout + "\n" + result.stderr
    require(result.returncode == 0,
            f"command failed ({result.returncode}): {' '.join(command)}\n{output}")
    require("Initialized arXiv:1001.4077 Kerr puncture" in output or
            "Setup complete, executing task list(s)" in output,
            f"run did not reach the Kerr/continuation contract\n{output}")
    return output


def restart_payload(path: Path) -> bytes:
    data = path.read_bytes()
    require(data.count(MARKER) == 1, f"{path} has an invalid restart header marker")
    header, payload = data.split(MARKER, 1)
    for key, value in (
        (b"symmetry", b"cartoon_so2"),
        (b"coordinate_map", b"half_rho_z_suppressed_y_v2"),
        (b"symmetry_schema", b"2"),
    ):
        field = re.compile(rb"^" + key + rb"\s*=\s*" + value + rb"(?:\s|$)",
                           re.MULTILINE)
        require(field.search(header) is not None,
                f"{path} omitted immutable metadata {key!r}={value!r}")
    return payload


def data_rows(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    require(rows, f"{path} contains no history rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--history-validator", required=True, type=Path)
    args = parser.parse_args()

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    fresh = args.work_dir / "fresh"
    resumed = args.work_dir / "resumed"
    fresh.mkdir(parents=True)
    resumed.mkdir()

    run([str(args.athena), "-i", str(args.input)], fresh)
    checkpoint = fresh / "rst" / "z4c_kerr_half_plane_restart.00001.rst"
    fresh_final = fresh / "rst" / "z4c_kerr_half_plane_restart.00002.rst"
    require(checkpoint.is_file() and fresh_final.is_file(),
            "fresh run omitted the cycle-one or cycle-two restart")

    run([str(args.athena), "-r", str(checkpoint)], resumed)
    resumed_final = resumed / "rst" / "z4c_kerr_half_plane_restart.00002.rst"
    require(resumed_final.is_file(), "resumed run omitted its cycle-two restart")

    # Parameter dumps legitimately record different output-directory history.
    # The complete binary payload (mesh, active/ghost state, and carrier) must be
    # byte-identical at the same accepted cycle.
    require(restart_payload(fresh_final) == restart_payload(resumed_final),
            "fresh and resumed cycle-two restart payloads differ")

    fresh_history = fresh / "z4c_kerr_half_plane_restart.z4c.user.hst"
    resumed_history = resumed / "z4c_kerr_half_plane_restart.z4c.user.hst"
    for history in (fresh_history, resumed_history):
        validation = subprocess.run(
            [str(args.python), str(args.history_validator),
             "--history", str(history)], cwd=args.work_dir, text=True,
            capture_output=True, check=False, timeout=15)
        require(validation.returncode == 0,
                f"history validation failed for {history}:\n"
                f"{validation.stdout}\n{validation.stderr}")
    require(data_rows(fresh_history)[-1] == data_rows(resumed_history)[-1],
            "fresh and resumed final diagnostic rows differ")
    print("half-plane Cartoon restart continuation is byte-identical")


if __name__ == "__main__":
    main()
