#!/usr/bin/env python3
"""Bounded native-VC refine/derefine integration regression."""

from __future__ import annotations

import argparse
import math
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
    args = parser.parse_args()

    work = pathlib.Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    completed = subprocess.run(
        [args.athena, "-i", args.input],
        cwd=work,
        text=True,
        capture_output=True,
        check=False,
    )
    require(completed.returncode == 0,
            f"native VC dynamic AMR run failed:\n{completed.stdout}\n{completed.stderr}")
    require("cycle=1 requested_refine=1" in completed.stdout and
            "created_leaves=3" in completed.stdout,
            "the deterministic refinement transaction was not observed")
    require("cycle=2 requested_refine=0 requested_derefine_leaves=4" in
            completed.stdout and "deleted_leaves=3" in completed.stdout,
            "the deterministic derefinement transaction was not observed")
    require("Current number of MeshBlocks = 4" in completed.stdout,
            "the final hierarchy did not return to four root leaves")

    history = work / "z4c_vc_minkowski_dynamic.z4c.user.hst"
    rows = [
        [float(value) for value in line.split()]
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    require(len(rows) >= 4, "history is missing accepted states")
    require([int(round(row[12])) for row in rows[:4]] == [4, 7, 4, 4],
            "history leaf counts do not bind refine/derefine lifecycle")
    require([int(round(row[13])) for row in rows[:4]] == [0, 1, 0, 0],
            "history maximum levels do not bind refine/derefine lifecycle")
    for row in rows:
        require(all(math.isfinite(value) for value in row),
                "history contains a nonfinite value")
        require(row[2] < 1.0e-18 and row[3] < 1.0e-18,
                "Minkowski constraints exceed the dynamic AMR regression bound")

    print("PASS: native VC dynamic refine/derefine lifecycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
