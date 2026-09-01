#!/usr/bin/env python3
"""Bounded native-VC refine/derefine integration regression."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import shutil
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mpiexec")
    parser.add_argument("--np-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=2)
    parser.add_argument("--geometry", choices=("cartesian", "cartoon"),
                        default="cartoon")
    parser.add_argument("--centering", choices=("cell", "vertex"),
                        default="vertex")
    parser.add_argument("--order", type=int, choices=(2, 4, 6), default=6)
    parser.add_argument("--vertex-prolongation-order",
                        choices=("auto", "4", "6", "8"), default="auto")
    parser.add_argument("--integrator", choices=("rk1", "rk2", "rk3", "rk4"),
                        default="rk4")
    parser.add_argument("--root-leaves", type=int, default=4)
    parser.add_argument("--meshblocks-per-dim", type=int, default=1)
    parser.add_argument("--expected-volume", type=float, default=64.0 * math.pi)
    parser.add_argument("--invalid-target-test", action="store_true")
    parser.add_argument("--invalid-halo-test", action="store_true")
    args = parser.parse_args()

    work = pathlib.Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    run_input = pathlib.Path(args.input)
    if args.vertex_prolongation_order != "auto":
        require(args.centering == "vertex",
                "explicit vertex prolongation is invalid for cell centering")
        input_text = run_input.read_text(encoding="utf-8")
        marker = "grid_centering = vertex"
        require(input_text.count(marker) == 1,
                "dynamic AMR fixture lacks a unique VC selector")
        run_input = work / pathlib.Path(args.input).name
        run_input.write_text(
            input_text.replace(
                marker, marker + "\nvertex_prolongation_order = " +
                args.vertex_prolongation_order),
            encoding="utf-8")
    command = [args.athena, "-i", str(run_input),
               f"z4c/spatial_order={args.order}",
               f"z4c/grid_centering={args.centering}",
               f"time/integrator={args.integrator}"]
    if args.meshblocks_per_dim > 1:
        input_text = pathlib.Path(args.input).read_text(encoding="utf-8")
        meshblock_text = input_text.split("<meshblock>", 1)[1].split("<", 1)[0]
        for direction in range(1, args.dimensions + 1):
            match = re.search(
                rf"^nx{direction}\s*=\s*(\d+)\s*$", meshblock_text,
                flags=re.MULTILINE)
            require(match is not None,
                    f"fixture is missing <meshblock>/nx{direction}")
            root_cells = args.meshblocks_per_dim * int(match.group(1))
            command.append(f"mesh/nx{direction}={root_cells}")
    if args.invalid_target_test:
        command.append("problem/amr_target_lx1=99")
    if args.invalid_halo_test:
        command.extend(("meshblock/nx1=8", "meshblock/nx2=8",
                        "meshblock/nx3=8"))
    if args.mpiexec:
        command = [args.mpiexec, args.np_flag, str(args.ranks), *command]
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("OMP_PROC_BIND", "false")
    completed = subprocess.run(
        command,
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if args.invalid_target_test:
        require(completed.returncode != 0,
                "invalid deterministic AMR target was unexpectedly accepted")
        require("lies outside the root MeshBlock lattice" in
                completed.stdout + completed.stderr,
                "invalid deterministic AMR target did not fail with the contract message")
        print(f"PASS: invalid {args.centering}-centered deterministic AMR target "
              "fails closed")
        return 0
    if args.invalid_halo_test:
        require(completed.returncode != 0,
                "undersized VC MeshBlock was unexpectedly accepted")
        require("increase the MeshBlock size" in completed.stdout + completed.stderr,
                "undersized VC MeshBlock did not fail with the halo contract message")
        print("PASS: undersized native-VC coarse halo fails closed")
        return 0
    require(completed.returncode == 0,
            f"{args.centering}-centered dynamic AMR run failed:\n"
            f"{completed.stdout}\n{completed.stderr}")
    child_count = 1 << args.dimensions
    created_leaves = child_count - 1
    if args.geometry == "cartoon":
        require("cycle=1 requested_refine=1" in completed.stdout and
                f"created_leaves={created_leaves}" in completed.stdout,
                "the deterministic refinement transaction was not observed")
        require(
            f"cycle=2 requested_refine=0 requested_derefine_leaves={child_count}" in
            completed.stdout and f"deleted_leaves={created_leaves}" in completed.stdout,
            "the deterministic derefinement transaction was not observed")
    else:
        require(
            f"{created_leaves} MeshBlocks created, {created_leaves} deleted by AMR" in
            completed.stdout,
            "the Cartesian refine/derefine transaction totals were not observed")
    require(f"Current number of MeshBlocks = {args.root_leaves}" in completed.stdout,
            "the final hierarchy did not return to the root leaf count")

    history = work / "z4c_vc_minkowski_dynamic.z4c.user.hst"
    history_text = history.read_text(encoding="utf-8")
    header = next(
        line for line in history_text.splitlines() if line.startswith("#  [1]=")
    )
    labels = re.findall(r"\[\d+\]=([^ ]+)", header)
    columns = {label: index for index, label in enumerate(labels)}
    required_columns = {
        "time", "C-norm2", "H-norm2", "Volume", "nmb_total", "minLapse",
        "maxRefLev", "cycle",
    }
    if args.geometry == "cartoon":
        required_columns.update(("axisLapse", "axisTau", "axisKret"))
    require(required_columns <= columns.keys(),
            f"history is missing columns: {sorted(required_columns - columns.keys())}")
    rows = [
        [float(value) for value in line.split()]
        for line in history_text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    require(len(rows) >= 4, "history is missing accepted states")
    refined_leaves = args.root_leaves + created_leaves
    require([int(round(row[columns["nmb_total"]])) for row in rows[:4]] ==
            [args.root_leaves, refined_leaves, args.root_leaves, args.root_leaves],
            "history leaf counts do not bind refine/derefine lifecycle")
    require([int(round(row[columns["maxRefLev"]])) for row in rows[:4]] ==
            [0, 1, 0, 0],
            "history maximum levels do not bind refine/derefine lifecycle")
    for row in rows:
        require(all(math.isfinite(value) for value in row),
                "history contains a nonfinite value")
        require(row[columns["C-norm2"]] < 1.0e-18 and
                row[columns["H-norm2"]] < 1.0e-18,
                "Minkowski constraints exceed the dynamic AMR regression bound")
        require(math.isclose(row[columns["Volume"]], args.expected_volume,
                             rel_tol=5.0e-6),
                f"{args.centering}-centered history does not integrate the exact "
                "physical volume")
        require(math.isclose(row[columns["minLapse"]], 1.0,
                             rel_tol=0.0, abs_tol=2.0e-14),
                f"{args.centering}-centered slice minimum lapse is not unity")
        if args.geometry == "cartoon":
            require(math.isclose(row[columns["axisLapse"]], 1.0,
                                 rel_tol=0.0, abs_tol=2.0e-14),
                    f"{args.centering}-centered central observer did not sample unit "
                    "lapse at the origin")
            require(math.isclose(row[columns["axisTau"]], row[columns["time"]],
                                 rel_tol=2.0e-6,
                                 abs_tol=2.0e-12),
                    f"{args.centering}-centered central proper time does not track "
                    "coordinate time")
            require(abs(row[columns["axisKret"]]) < 1.0e-18,
                    f"{args.centering}-centered origin curvature is nonzero in "
                    "Minkowski spacetime")
    require(max(row[columns["Volume"]] for row in rows) ==
            min(row[columns["Volume"]] for row in rows),
            f"{args.centering}-centered history volume changed across "
            "refine/derefine events")

    restarts = sorted((work / "rst").glob("*.rst"))
    require(len(restarts) >= 4 and all(path.stat().st_size > 0 for path in restarts),
            "dynamic AMR fixture did not produce valid restart files")
    print(f"PASS: {args.centering}-centered {args.dimensions}D {args.geometry} "
          f"O{args.order} q{args.vertex_prolongation_order} {args.integrator} "
          "dynamic refine/derefine lifecycle "
          "and quadrature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
