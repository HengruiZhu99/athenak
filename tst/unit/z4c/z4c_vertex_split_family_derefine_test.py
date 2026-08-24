#!/usr/bin/env python3
"""Rank-invariance regression for split native-VC derefinement families."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_case(args: argparse.Namespace, ranks: int, work: Path) -> dict:
    work.mkdir(parents=True)
    command = [args.mpiexec, args.np_flag, str(ranks),
               str(args.athena.resolve()), "-i", str(args.input.resolve()),
               "-d", str(work.resolve()),
               "z4c/spatial_order=2",
               "z4c/vertex_prolongation_order=4",
               "problem/amp=1e-4",
               # Root Z-order GID 15 becomes old lower-child GID 18 after
               # the first family is refined.  With 2 or 4 equal-cost ranks,
               # the family straddles a partition while parent GID 15 stays
               # with its lower child.
               "problem/amr_target1_lx1=3",
               "problem/amr_target1_lx2=3"]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment.setdefault("OMP_PROC_BIND", "false")
    completed = subprocess.run(command, cwd=work, env=environment,
                               text=True, capture_output=True, check=False)
    require(completed.returncode == 0,
            f"{ranks}-rank split-family fixture failed "
            f"({completed.returncode})\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}")
    require("Current number of MeshBlocks = 32" in completed.stdout and
            "9 MeshBlocks created, 9 deleted by AMR" in completed.stdout,
            f"{ranks}-rank fixture lost its three-family hierarchy")
    outputs = sorted((work / "bin").glob("*.bin"))
    require(outputs, f"{ranks}-rank fixture omitted native-VC output")
    sys.path.insert(0, str(args.input.parents[2] / "vis" / "python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    accepted = [bin_convert.read_binary(str(path)) for path in outputs]
    final = next((state for state in accepted
                  if state["cycle"] == 2 and state["n_mbs"] == 32), None)
    require(final is not None, f"{ranks}-rank output omitted the post-derefine state")
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--np-flag", default="-n")
    parser.add_argument("--ranks", choices=(2, 4), type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    one = run_case(args, 1, args.work_dir / "one_rank")
    split = run_case(args, args.ranks, args.work_dir / f"split_{args.ranks}_ranks")
    require(one["time"] == split["time"] and one["cycle"] == split["cycle"],
            "rank comparison did not reach the same accepted state")
    require(np.array_equal(one["mb_logical"], split["mb_logical"]) and
            np.array_equal(one["mb_geometry"], split["mb_geometry"]) and
            one["var_names"] == split["var_names"],
            "rank comparison changed logical or physical hierarchy metadata")

    mismatches = []
    for variable in one["var_names"]:
        reference = np.stack(one["mb_data"][variable])
        candidate = np.stack(split["mb_data"][variable])
        indices = np.argwhere(reference != candidate)
        if len(indices):
            first = tuple(int(value) for value in indices[0])
            difference = np.abs(candidate.astype(np.float64) -
                                reference.astype(np.float64))
            mismatches.append((variable, first, int(np.count_nonzero(difference)),
                               float(np.max(difference))))
    require(not mismatches,
            "split-family native-VC derefinement is rank dependent; expected "
            "the lower-child/parent-local family at parent GID 15 to expose "
            f"missing local quadrants, first mismatches={mismatches[:5]}")
    print(f"PASS: native-VC split-family derefinement is bitwise invariant "
          f"between one and {args.ranks} ranks for all 25 variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
