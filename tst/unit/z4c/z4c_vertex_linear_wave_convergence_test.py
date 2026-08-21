#!/usr/bin/env python3
"""Run the bounded native-VC Cartesian O4 N64/N128/N256 gate."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


RESOLUTIONS = (64, 128, 256)
WAVE_COLUMNS = {
    "z4c_gyy": 4,
    "z4c_gzz": 6,
    "z4c_Ayy": 11,
    "z4c_Azz": 13,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_case(athena: Path, input_path: Path, root: Path,
             resolution: int) -> tuple[float, float, list[list[float]]]:
    case = root / f"n{resolution}"
    case.mkdir(parents=True)
    basename = f"z4c_vc_linear_wave_o4_n{resolution}"
    command = [
        str(athena), "-i", str(input_path), "-d", str(case),
        f"mesh/nx1={resolution}", f"meshblock/nx1={resolution}",
        f"job/basename={basename}",
    ]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(command, cwd=case, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    require(result.returncode == 0,
            f"native-VC N{resolution} linear wave failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    require("Terminating on time limit" in result.stdout,
            f"N{resolution} did not complete one wave period")
    require(not list(case.rglob("z4c_state_failure.json")),
            f"N{resolution} emitted a Z4c state failure")
    error_file = case / f"{basename}-errs.dat"
    require(error_file.is_file(), f"N{resolution} omitted its error norm")
    rows = [line.split() for line in error_file.read_text(
        encoding="utf-8").splitlines() if line and not line.startswith("#")]
    require(len(rows) == 1 and len(rows[0]) == 12,
            f"N{resolution} linear-wave error schema changed")
    require(int(rows[0][0]) == resolution,
            f"N{resolution} error row reports a different grid")
    rms_l1 = float(rows[0][4])
    linf = float(rows[0][5])
    require(math.isfinite(rms_l1) and math.isfinite(linf) and
            rms_l1 > 0.0 and linf > 0.0,
            f"N{resolution} error norm is invalid")
    final_table = case / f"tab/{basename}.z4c.00001.tab"
    require(final_table.is_file(), f"N{resolution} omitted its final VC state")
    state_rows = [line.split() for line in final_table.read_text(
        encoding="utf-8").splitlines() if line and not line.startswith("#")]
    require(len(state_rows) == resolution,
            f"N{resolution} final state has {len(state_rows)} vertices")
    coordinates = [float(row[2]) for row in state_rows]
    require(coordinates[0] == 0.0 and coordinates[-1] < 1.0 and
            all(coordinates[index] < coordinates[index + 1]
                for index in range(resolution - 1)),
            f"N{resolution} final VC coordinates are not ordered periodic owners")
    state = [[float(value) for value in row[3:]] for row in state_rows]
    require(all(len(row) == 25 for row in state) and
            all(math.isfinite(value) for row in state for value in row),
            f"N{resolution} final VC state schema is invalid")
    return rms_l1, linf, state


def coincident_rms(coarse: list[list[float]],
                   fine: list[list[float]], columns: tuple[int, ...]) -> float:
    require(len(fine) == 2 * len(coarse),
            "self-convergence states are not nested 2:1 vertex grids")
    squared = 0.0
    count = 0
    for index, coarse_row in enumerate(coarse):
        fine_row = fine[2 * index]
        for column in columns:
            squared += (coarse_row[column] - fine_row[column]) ** 2
            count += 1
    return math.sqrt(squared / count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    athena = args.athena.resolve()
    input_path = args.input.resolve()
    require(athena.is_file() and input_path.is_file(),
            "linear-wave executable or input is missing")
    text = input_path.read_text(encoding="utf-8")
    for pattern, label in (
            (r"(?m)^grid_centering\s*=\s*vertex\s*$", "native VC"),
            (r"(?m)^spatial_order\s*=\s*4\s*$", "O4"),
            (r"(?m)^integrator\s*=\s*rk4\s*$", "RK4"),
            (r"(?m)^refinement\s*=\s*none\s*$", "uniform grid")):
        require(re.search(pattern, text) is not None,
                f"linear-wave input lost its {label} contract")

    work_dir = args.work_dir.resolve()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    results = [run_case(athena, input_path, work_dir, resolution)
             for resolution in RESOLUTIONS]
    orders: dict[str, float] = {}
    differences: dict[str, list[float]] = {}
    for name, column in WAVE_COLUMNS.items():
        coarse_medium = coincident_rms(
            results[0][2], results[1][2], (column,))
        medium_fine = coincident_rms(
            results[1][2], results[2][2], (column,))
        differences[name] = [coarse_medium, medium_fine]
        orders[name] = math.log(coarse_medium / medium_fine, 2.0)
    require(min(orders.values()) >= 3.5,
            "native-VC O4 excited-field self-convergence is too low: "
            f"orders={orders} differences={differences}")
    all_coarse_medium = coincident_rms(
        results[0][2], results[1][2], tuple(range(25)))
    all_medium_fine = coincident_rms(
        results[1][2], results[2][2], tuple(range(25)))
    require(max(all_coarse_medium, all_medium_fine) < 1.0e-8,
            "native-VC ancillary state is not bounded in the linear-wave gate")
    print("native-VC Cartesian O4 convergence passed: "
          f"wave_orders={orders} wave_differences={differences} "
          f"all_state_differences={[all_coarse_medium, all_medium_fine]} "
          f"analytic_norms="
          f"{[(value[0], value[1]) for value in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
