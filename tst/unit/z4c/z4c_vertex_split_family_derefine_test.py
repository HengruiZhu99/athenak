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
               "problem/amp=1e-4"]
    target_count = 3
    if args.ownership_case == "local_lower":
        # Root Z-order GID 15 becomes old lower-child GID 18 after the first
        # family is refined.  With 2 or 4 equal-cost ranks, the family spans a
        # partition while parent GID 15 stays with its lower child (layout B).
        command.extend(("problem/amr_target1_lx1=3",
                        "problem/amr_target1_lx2=3"))
    elif args.ownership_case == "remote_lower":
        # Preserve the input's root GID 16 target.  It becomes old lower-child
        # GID 19.  That lower child stays on the lower rank while the new parent
        # moves right to the rank owning GID 16; upper siblings are already on
        # the target rank (layout C and the move-right discriminator).
        pass
    elif args.ownership_case == "dual_split":
        # Four roots with Z-order GIDs 8,9,10,12 become old family bases
        # 8,12,16,21.  At four ranks, families 8:11 and 21:24 straddle two old
        # partitions and both new parents 8 and 12 target rank 1 (layout E).
        target_count = 4
        command.extend(("problem/amr_target_count=4",
                        "problem/amr_target_lx1=0",
                        "problem/amr_target_lx2=2",
                        "problem/amr_target1_lx1=1",
                        "problem/amr_target1_lx2=2",
                        "problem/amr_target2_lx1=0",
                        "problem/amr_target2_lx2=3",
                        "problem/amr_target3_lx1=2",
                        "problem/amr_target3_lx2=2"))
    else:
        raise AssertionError(f"unsupported ownership case {args.ownership_case}")
    if args.mixed_refine:
        # Root Z-order GID 16 follows the split parent at root GID 15.  Its
        # new child slots overlap old local child slots 18:20, exercising
        # preservation of split-family sources across A7 refinement copying.
        command.extend(("problem/exercise_mixed_amr=true",
                        "problem/amr_mixed_refine_lx1=0",
                        "problem/amr_mixed_refine_lx2=4"))
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment.setdefault("OMP_PROC_BIND", "false")
    completed = subprocess.run(command, cwd=work, env=environment,
                               text=True, capture_output=True, check=False)
    require(completed.returncode == 0,
            f"{ranks}-rank split-family fixture failed "
            f"({completed.returncode})\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}")
    terminal_blocks = 35 if args.mixed_refine else 32
    created_blocks = 3 * target_count + (3 if args.mixed_refine else 0)
    deleted_blocks = 3 * target_count
    require(f"Current number of MeshBlocks = {terminal_blocks}" in completed.stdout and
            f"{created_blocks} MeshBlocks created, {deleted_blocks} deleted by AMR"
            in completed.stdout,
            f"{ranks}-rank fixture lost its {target_count}-family hierarchy")
    outputs = sorted((work / "bin").glob("*.bin"))
    require(outputs, f"{ranks}-rank fixture omitted native-VC output")
    sys.path.insert(0, str(args.input.parents[2] / "vis" / "python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    accepted = [bin_convert.read_binary(str(path)) for path in outputs]
    final = next((state for state in accepted
                  if state["cycle"] == 2 and state["n_mbs"] == terminal_blocks), None)
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
    parser.add_argument("--mixed-refine", action="store_true")
    parser.add_argument("--ownership-case",
                        choices=("local_lower", "remote_lower", "dual_split"),
                        default="local_lower")
    args = parser.parse_args()

    if args.ownership_case == "dual_split":
        require(args.ranks == 4,
                "dual_split is defined for the four-rank partition")
        require(not args.mixed_refine,
                "dual_split isolates simultaneous split derefinement")

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
            "split-family native-VC derefinement is rank dependent; "
            f"ownership_case={args.ownership_case}; "
            f"missing local quadrants, first mismatches={mismatches[:5]}")
    print(f"PASS: native-VC split-family derefinement is bitwise invariant "
          f"between one and {args.ranks} ranks for all 25 variables "
          f"(ownership_case={args.ownership_case})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
