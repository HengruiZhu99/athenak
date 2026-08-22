#!/usr/bin/env python3
"""Compare a nonconstant native-VC refine/derefine run with a uniform reference."""

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


def run_case(athena: Path, input_path: Path, root: Path, dimensions: int,
             resolution: int, adaptive: bool, order: int, integrator: str,
             meshblocks_per_dim: int) -> tuple[dict, dict[float, list[float]], str]:
    label = (f"{'amr' if adaptive else 'uniform'}_o{order}_{integrator}_"
             f"n{resolution}")
    work = root / label
    work.mkdir(parents=True)
    basename = f"z4c_vc_dynamic_linear_wave_{label}"
    meshblock_resolution = resolution // meshblocks_per_dim
    command = [
        str(athena), "-i", str(input_path), "-d", str(work),
        f"job/basename={basename}",
        f"mesh/nx1={resolution}", f"meshblock/nx1={meshblock_resolution}",
        f"mesh/nx2={resolution}", f"meshblock/nx2={meshblock_resolution}",
        f"z4c/spatial_order={order}", f"time/integrator={integrator}",
    ]
    if dimensions == 3:
        command.extend((f"mesh/nx3={resolution}",
                        f"meshblock/nx3={meshblock_resolution}"))
    if not adaptive:
        command.extend(("mesh_refinement/refinement=none",
                        "problem/exercise_deterministic_amr=false"))
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(command, cwd=work, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    require(result.returncode == 0,
            f"{label} failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    require(not list(work.rglob("z4c_state_failure.json")),
            f"{label} emitted a state-admissibility failure")
    if adaptive:
        created = (1 << dimensions) - 1
        require(f"{created} MeshBlocks created, {created} deleted by AMR" in
                result.stdout,
                f"{label} did not execute the deterministic refine/derefine pair")
        root_leaves = meshblocks_per_dim ** dimensions
        require(f"Current number of MeshBlocks = {root_leaves}" in result.stdout,
                f"{label} did not return to the root hierarchy")

    sys.path.insert(0, str(input_path.parents[2] / "vis" / "python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    outputs = sorted((work / "bin").glob(f"{basename}.state.*.bin"))
    require(outputs, f"{label} omitted native-VC binary output")
    tables = sorted((work / "tab").glob(f"{basename}.line.*.tab"))
    require(tables, f"{label} omitted its double-precision interface slice")
    rows = [line.split() for line in tables[-1].read_text(
        encoding="utf-8").splitlines() if line and not line.startswith("#")]
    line_state = {float(row[2]): [float(value) for value in row[3:]]
                  for row in rows}
    require(len(line_state) == resolution and
            all(len(values) == 25 for values in line_state.values()),
            f"{label} interface slice is not a canonical 25-variable VC line")
    return bin_convert.read_binary(str(outputs[-1])), line_state, result.stdout


def payload_difference(left: dict, right: dict,
                       left_line: dict[float, list[float]],
                       right_line: dict[float, list[float]]) -> tuple[float, float]:
    require(left["time"] == right["time"] and left["cycle"] >= 3 and
            right["cycle"] >= 3,
            "uniform/AMR comparison did not reach the same accepted time")
    require(left["n_mbs"] == right["n_mbs"] and
            np.array_equal(left["mb_logical"], right["mb_logical"]) and
            np.array_equal(left["mb_geometry"], right["mb_geometry"]),
            "uniform/AMR final hierarchies differ")
    require(left["var_names"] == right["var_names"] and
            len(left["var_names"]) == 25,
            "uniform/AMR native state inventories differ")
    require(left_line.keys() == right_line.keys(),
            "uniform/AMR canonical interface coordinates differ")
    differences = [left_value - right_value
                   for coordinate in left_line
                   for left_value, right_value in
                   zip(left_line[coordinate], right_line[coordinate])]
    values = np.asarray(differences)
    return float(np.sqrt(np.mean(values * values))), float(np.max(np.abs(values)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--dimensions", required=True, type=int, choices=(2, 3))
    parser.add_argument("--order", type=int, choices=(2, 4, 6), default=4)
    parser.add_argument("--integrator", choices=("rk1", "rk2", "rk3", "rk4"),
                        default="rk4")
    parser.add_argument("--meshblocks-per-dim", type=int, default=2)
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=(16, 32),
                        help="root resolutions used for the convergence check")
    args = parser.parse_args()
    require(len(args.resolutions) >= 2 and
            all(resolution > 0 and resolution % 2 == 0
                for resolution in args.resolutions),
            "need at least two positive even resolutions")
    require(args.meshblocks_per_dim > 0 and
            all(resolution % args.meshblocks_per_dim == 0
                for resolution in args.resolutions),
            "every resolution must be divisible by --meshblocks-per-dim")
    fine_nghost = 4
    transfer_order = {2: 4, 4: 6, 6: 8}[args.order]
    coarse_nghost = max(
        fine_nghost, (fine_nghost - 1) // 2 + transfer_order // 2)
    require(all((resolution // args.meshblocks_per_dim) // 2 >= coarse_nghost
                for resolution in args.resolutions),
            "selected MeshBlock size cannot supply the centered coarse halo; "
            "reduce --meshblocks-per-dim or increase the resolutions")

    text = args.input.read_text(encoding="utf-8")
    for needle in ("grid_centering = vertex", "spatial_order = 4",
                   "integrator = rk4",
                   "exercise_deterministic_amr = true",
                   "amr_refine_time = 0.01",
                   "amr_derefine_time = 0.024",
                   "write_period_error = false"):
        require(needle in text, f"nonconstant AMR input lost '{needle}'")
    root = args.work_dir.resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    errors = []
    for resolution in args.resolutions:
        uniform, uniform_line, _ = run_case(
            args.athena.resolve(), args.input.resolve(), root,
            args.dimensions, resolution, False, args.order, args.integrator,
            args.meshblocks_per_dim)
        adaptive, adaptive_line, _ = run_case(
            args.athena.resolve(), args.input.resolve(), root,
            args.dimensions, resolution, True, args.order, args.integrator,
            args.meshblocks_per_dim)
        errors.append(payload_difference(
            uniform, adaptive, uniform_line, adaptive_line))
    require(all(error[0] > 0.0 for error in errors),
            "nonconstant transfer comparison unexpectedly has zero error")
    orders = [math.log(coarse[0] / fine[0],
                       fine_resolution / coarse_resolution)
              for coarse, fine, coarse_resolution, fine_resolution in
              zip(errors, errors[1:], args.resolutions, args.resolutions[1:])]
    # The coarse member may precede the asymptotic regime, so require at least
    # p-1 on every interval while retaining strict monotonic decrease.  The
    # measured orders are always reported for the qualification matrix.
    minimum_order = max(1.5, args.order - 1.0)
    require(all(fine[0] < coarse[0] for coarse, fine in
                zip(errors, errors[1:])) and min(orders) >= minimum_order,
            f"nonconstant AMR mismatch does not decrease robustly: "
            f"errors={errors} orders={orders}")
    require(errors[-1][1] < 1.0e-6,
            f"nonconstant AMR Linf mismatch is unexpectedly large: {errors}")
    print(f"PASS: native-VC {args.dimensions}D O{args.order} {args.integrator} "
          "nonconstant dynamic AMR "
          f"errors={errors} observed_orders={orders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
