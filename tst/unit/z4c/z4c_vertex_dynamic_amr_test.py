#!/usr/bin/env python3
"""Bounded native-VC refine/derefine integration regression."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
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
    parser.add_argument("--order", type=int, choices=(2, 4, 6), default=6)
    parser.add_argument("--root-leaves", type=int, default=4)
    parser.add_argument("--expected-volume", type=float, default=64.0 * math.pi)
    parser.add_argument("--invalid-target-test", action="store_true")
    args = parser.parse_args()

    work = pathlib.Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    command = [args.athena, "-i", args.input,
               f"z4c/spatial_order={args.order}"]
    if args.invalid_target_test:
        command.append("problem/amr_target_lx1=99")
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
        print("PASS: invalid native VC deterministic AMR target fails closed")
        return 0
    require(completed.returncode == 0,
            f"native VC dynamic AMR run failed:\n{completed.stdout}\n{completed.stderr}")
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
    rows = [
        [float(value) for value in line.split()]
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    require(len(rows) >= 4, "history is missing accepted states")
    refined_leaves = args.root_leaves + created_leaves
    require([int(round(row[12])) for row in rows[:4]] ==
            [args.root_leaves, refined_leaves, args.root_leaves, args.root_leaves],
            "history leaf counts do not bind refine/derefine lifecycle")
    require([int(round(row[13])) for row in rows[:4]] == [0, 1, 0, 0],
            "history maximum levels do not bind refine/derefine lifecycle")
    for row in rows:
        require(all(math.isfinite(value) for value in row),
                "history contains a nonfinite value")
        require(row[2] < 1.0e-18 and row[3] < 1.0e-18,
                "Minkowski constraints exceed the dynamic AMR regression bound")
        require(math.isclose(row[10], args.expected_volume, rel_tol=5.0e-6),
                "native VC history does not integrate the exact physical volume")
        if args.geometry == "cartoon":
            require(math.isclose(row[18], 1.0, rel_tol=0.0, abs_tol=2.0e-14),
                    "native VC central observer did not sample unit lapse at the origin")
            require(math.isclose(row[19], row[0], rel_tol=2.0e-6,
                                 abs_tol=2.0e-12),
                    "native VC central proper time does not track coordinate time")
            require(abs(row[20]) < 1.0e-18,
                    "native VC origin curvature is nonzero in Minkowski spacetime")
    require(max(row[10] for row in rows) == min(row[10] for row in rows),
            "native VC history volume changed across refine/derefine events")

    restarts = sorted((work / "rst").glob("*.rst"))
    require(len(restarts) >= 4 and all(path.stat().st_size > 0 for path in restarts),
            "dynamic AMR fixture did not produce valid restart files")
    print(f"PASS: native VC {args.dimensions}D {args.geometry} O{args.order} "
          "dynamic refine/derefine lifecycle and quadrature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
