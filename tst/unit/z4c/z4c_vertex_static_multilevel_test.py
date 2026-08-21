#!/usr/bin/env python3
"""Bounded production-path VC wave/coarse-fine interface gate."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric_rows(path: Path) -> list[list[float]]:
    return [[float(value) for value in line.split()]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def active_coordinates(binary: dict[str, object], block: int,
                       i: int, j: int, k: int) -> tuple[float, float, float]:
    geometry = binary["mb_geometry"][block]
    nx1 = int(binary["nx1_out_mb"])
    nx2 = int(binary["nx2_out_mb"])
    nx3 = int(binary["nx3_out_mb"])
    x1 = float(geometry[0]) + i * (float(geometry[1]) - float(geometry[0])) / (nx1 - 1)
    x2 = float(geometry[2]) + j * (float(geometry[3]) - float(geometry[2])) / (nx2 - 1)
    x3 = float(geometry[4]) + k * (float(geometry[5]) - float(geometry[4])) / (nx3 - 1)
    return x1, x2, x3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(
        [str(args.athena.resolve()), "-i", str(args.input.resolve())],
        cwd=work, env=environment, text=True, capture_output=True, check=False)
    require(result.returncode == 0,
            f"native-VC static multilevel wave failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    require("Total number of MeshBlocks = 9" in result.stdout and
            "Number of physical levels of refinement = 1" in result.stdout,
            "static multilevel hierarchy inventory changed")
    require(not list(work.rglob("z4c_state_failure.json")),
            "static multilevel wave emitted a Z4c failure artifact")

    history = numeric_rows(work / "z4c_vc_static_multilevel_wave.z4c.user.hst")
    evaluated_history = [row for row in history if int(round(row[17])) > 0]
    require(len(evaluated_history) >= 4 and
            all(math.isfinite(value) for row in evaluated_history for value in row),
            "evaluated static multilevel history is incomplete or nonfinite")
    require(all(int(round(row[12])) == 9 and int(round(row[13])) == 1
                for row in evaluated_history),
            "static hierarchy changed during the bounded wave run")
    require(max(max(row[2], row[3], row[4], row[5])
                for row in evaluated_history) < 1.0e-4,
            "static multilevel constraint norm exceeds its bounded-wave guard")

    sys.path.insert(0, str(args.source_dir.resolve() / "vis/python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    state_paths = sorted((work / "bin").glob("*.state.*.bin"))
    constraint_paths = sorted((work / "bin").glob("*.constraints.*.bin"))
    require(len(state_paths) >= 2 and len(constraint_paths) >= 2,
            "static multilevel initial/final binary inventory is incomplete")
    state = bin_convert.read_binary(str(state_paths[-1]))
    constraints = bin_convert.read_binary(str(constraint_paths[-1]))
    require(state["n_mbs"] == 9 and constraints["n_mbs"] == 9,
            "binary hierarchy does not contain the expected leaves")

    # Shared physical vertices must be exactly synchronized after the RK stage.
    contributors: dict[tuple[float, float, float], list[tuple[float, ...]]] = defaultdict(list)
    for block in range(int(state["n_mbs"])):
        for k in range(int(state["nx3_out_mb"])):
            for j in range(int(state["nx2_out_mb"])):
                for i in range(int(state["nx1_out_mb"])):
                    coordinate = tuple(round(value, 14) for value in
                                       active_coordinates(state, block, i, j, k))
                    value = tuple(float(state["mb_data"][name][block][k, j, i])
                                  for name in state["var_names"])
                    contributors[coordinate].append(value)
    duplicate_groups = [values for values in contributors.values() if len(values) > 1]
    require(duplicate_groups, "static hierarchy exposed no shared VC nodes")
    maximum_shared_mismatch = max(
        max(abs(value[component] - values[0][component])
            for value in values[1:] for component in range(len(value)))
        for values in duplicate_groups)
    require(maximum_shared_mismatch == 0.0,
            f"shared VC nodes differ after synchronization: {maximum_shared_mismatch}")

    # Separate the fixed coarse-fine layer from block interiors without assigning
    # causality to either norm.  Both inventories must remain finite and bounded.
    interface_squares: list[float] = []
    interior_squares: list[float] = []
    for block in range(int(constraints["n_mbs"])):
        geometry = constraints["mb_geometry"][block]
        dx = ((float(geometry[1]) - float(geometry[0])) /
              (int(constraints["nx1_out_mb"]) - 1))
        for k in range(int(constraints["nx3_out_mb"])):
            for j in range(int(constraints["nx2_out_mb"])):
                for i in range(int(constraints["nx1_out_mb"])):
                    x1, _, _ = active_coordinates(constraints, block, i, j, k)
                    target = interface_squares if abs(x1) <= 2.0 * dx else interior_squares
                    target.extend(float(constraints["mb_data"][name][block][k, j, i]) ** 2
                                  for name in constraints["var_names"])
    require(interface_squares and interior_squares and
            all(math.isfinite(value) for value in interface_squares + interior_squares),
            "coarse-fine/interior constraint inventories are invalid")
    interface_rms = math.sqrt(sum(interface_squares) / len(interface_squares))
    interior_rms = math.sqrt(sum(interior_squares) / len(interior_squares))
    require(interface_rms < 1.0e-3 and interior_rms < 1.0e-3,
            "static multilevel local constraint carrier exceeded its bound")
    print("PASS: native-VC static multilevel wave "
          f"shared_mismatch={maximum_shared_mismatch} "
          f"interface_constraint_rms={interface_rms} "
          f"interior_constraint_rms={interior_rms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
