#!/usr/bin/env python3
"""Measure the finest Cartesian cell spacing stored in AthenaK binary output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vis" / "python"))
import bin_convert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    report = {}
    for path in args.files:
        data = bin_convert.read_binary(path)
        geometry = np.asarray(data["mb_geometry"], dtype=float)
        cell_counts = np.asarray(
            [data["nx1_out_mb"], data["nx2_out_mb"], data["nx3_out_mb"]],
            dtype=float,
        )
        widths = geometry[:, [1, 3, 5]] - geometry[:, [0, 2, 4]]
        active = cell_counts > 1
        spacings = widths[:, active]/cell_counts[active]
        report[str(path)] = {
            "time": float(data["time"]),
            "meshblocks_in_output": int(data["n_mbs"]),
            "finest_spacing": float(np.min(spacings)),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
