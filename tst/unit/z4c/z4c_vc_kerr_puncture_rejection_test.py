#!/usr/bin/env python3
"""Require a true-vertex Kerr puncture at r=0 to fail closed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("OMP_PROC_BIND", "false")
    result = subprocess.run(
        [str(args.athena.resolve()), "-i", str(args.input.resolve())],
        cwd=work, env=environment, text=True, capture_output=True, check=False)
    output = result.stdout + "\n" + result.stderr
    require(result.returncode != 0,
            "a singular Kerr puncture on an evolved true vertex was accepted")
    require("sampling topology that does not contain r=0" in output and
            "no epsilon clipping is permitted" in output,
            "true-vertex Kerr puncture did not fail with the strict contract")
    require("Initialized arXiv:1001.4077 Kerr puncture" not in output,
            "singular data reached the initialized-state claim")
    print("PASS: singular true-vertex Kerr puncture fails closed without clipping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
