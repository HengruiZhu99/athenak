#!/usr/bin/env python3
"""Run a bounded analytic harmonic gauge-wave convergence pair."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_case(athena: Path, input_path: Path, root: Path, centering: str,
             order: int, resolution: int) -> tuple[float, float]:
    work = root / f"n{resolution}"
    work.mkdir(parents=True)
    basename = f"z4c_gauge_{centering}_o{order}_n{resolution}"
    command = [
        str(athena), "-i", str(input_path), "-d", str(work),
        f"job/basename={basename}", f"mesh/nx1={resolution}",
        f"meshblock/nx1={resolution // 2}",
        f"mesh/nghost={order // 2 + 1}", f"z4c/spatial_order={order}",
        f"z4c/grid_centering={centering}",
    ]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(command, cwd=work, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    require(result.returncode == 0,
            f"N{resolution} failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    require(not list(work.rglob("z4c_state_failure.json")),
            f"N{resolution} emitted a state-admissibility failure")

    sys.path.insert(0, str(input_path.parents[2] / "vis" / "python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    outputs = sorted((work / "bin").glob(f"{basename}.adm.*.bin"))
    require(len(outputs) == 2, "gauge wave did not write initial/final ADM states")
    data = bin_convert.read_binary(str(outputs[-1]))
    require(abs(float(data["time"]) - 0.05) <= 8.0 * np.finfo(float).eps,
            f"gauge wave stopped at unexpected time {data['time']}")
    tables = sorted((work / "tab").glob(f"{basename}.line.*.tab"))
    require(len(tables) == 2, "gauge wave did not write initial/final ADM slices")
    lines = tables[-1].read_text(encoding="utf-8").splitlines()
    rows = [line.split() for line in lines if line and not line.startswith("#")]
    require(rows, "final gauge-wave ADM slice is empty")
    amplitude = 1.0e-3
    wave_number = 2.0 * math.pi
    errors: list[float] = []
    maximum = 0.0
    duplicate_values: dict[tuple[float, int], float] = {}
    names = list(data["var_names"])
    selected = {"adm_gxx", "adm_Kxx", "adm_alpha"}.intersection(names)
    require(selected.issuperset({"adm_gxx", "adm_Kxx"}),
            f"ADM output omitted gauge-wave fields: {names}")
    for row in rows:
        x = float(row[2])
        values = [float(value) for value in row[-len(names):]]
        phase = wave_number * (x - float(data["time"]))
        h = 1.0 - amplitude * math.sin(phase)
        expected = {
            "adm_gxx": h,
            "adm_Kxx": -0.5 * amplitude * wave_number * math.cos(phase) /
                       math.sqrt(h),
            "adm_alpha": math.sqrt(h),
        }
        for name in selected:
            variable = names.index(name)
            difference = values[variable] - expected[name]
            errors.append(difference)
            maximum = max(maximum, abs(difference))
            if centering == "vertex":
                key = (x, variable)
                if key in duplicate_values:
                    require(values[variable] == duplicate_values[key],
                            f"shared VC ADM node differs for {key}")
                else:
                    duplicate_values[key] = values[variable]
    array = np.asarray(errors)
    return float(np.sqrt(np.mean(array * array))), maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--centering", required=True, choices=("cell", "vertex"))
    parser.add_argument("--order", required=True, type=int, choices=(2, 4, 6))
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    for needle in ("pgen_name = z4c_gauge_wave", "lapse_harmonic = 1.0",
                   "shift_mode = prescribed_zero", "damp_kappa1 = 0.0",
                   "damp_kappa2 = 0.0"):
        require(needle in text, f"gauge-wave input lost '{needle}'")
    root = args.work_dir.resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    errors = [run_case(args.athena.resolve(), args.input.resolve(), root,
                       args.centering, args.order, resolution)
              for resolution in (16, 32)]
    require(errors[0][0] > errors[1][0] > 0.0,
            f"gauge-wave RMS error did not decrease: {errors}")
    observed_order = math.log(errors[0][0] / errors[1][0], 2.0)
    minimum_order = {2: 1.5, 4: 3.0, 6: 3.5}[args.order]
    require(observed_order >= minimum_order,
            f"O{args.order} gauge-wave convergence is too low: "
            f"errors={errors} order={observed_order}")
    print(f"PASS: {args.centering} O{args.order} analytic gauge wave "
          f"errors={errors} observed_order={observed_order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
