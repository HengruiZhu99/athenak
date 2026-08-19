#!/usr/bin/env python3
"""Summarize frozen Ref-GH stationary-trumpet history files.

The script deliberately uses the column order defined by
``HistoryOutput::LoadRefGhHistoryData`` rather than the abbreviated labels in
Athena history headers.  It produces matched-time rows for one or more cases,
preserving raw L2-squared quantities and adding explicitly named square-root
integrated and volume-normalized RMS values.  The standard Ref-GH history does
not contain regular-field L1 or Linf error norms; this tool reports that gap as
``not-recorded`` instead of manufacturing a norm from L2 data.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


REF_GH_COLUMNS = (
    "time", "dt", "GH-L2sq", "Reduction-L2sq", "Curl-L2sq",
    "PsiError-L2sq", "Pi-L2sq", "Phi-L2sq", "GHnear-L2sq",
    "ReductionNear-L2sq", "CurlNear-L2sq", "Volume", "alpha-max",
    "minus-alpha-min", "regular-max", "G-condition-max",
    "coordinate-g-max", "char-speed-max", "effective-CFL",
    "minus-detg-margin", "NearVolume", "bad-state", "Q-Linf",
    "Delta-Linf", "frame-Ricci-Linf", "coordinate-Ricci-Linf",
    "source-curvature-Linf", "source-QQ-Linf", "source-DeltaDelta-Linf",
    "source-damping-Linf", "source-frame-correction-Linf",
)

DEFAULT_TIMES = (0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0)


def read_history(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            if len(values) != len(REF_GH_COLUMNS):
                raise ValueError(
                    f"{path}:{number}: expected {len(REF_GH_COLUMNS)} columns, "
                    f"found {len(values)}"
                )
            rows.append(dict(zip(REF_GH_COLUMNS, map(float, values), strict=True)))
    if not rows:
        raise ValueError(f"{path}: contains no numeric Ref-GH history rows")
    return rows


def closest(rows: Iterable[dict[str, float]], target: float) -> dict[str, float]:
    return min(rows, key=lambda row: abs(row["time"] - target))


def root(value: float) -> float:
    # Tiny negative values can arise only from a corrupt sum, not roundoff,
    # because the source values are nonnegative.  Preserve that evidence.
    return math.sqrt(value) if value >= 0.0 else math.nan


def enriched(case: str, requested: float, row: dict[str, float]) -> dict[str, str]:
    volume = row["Volume"]
    near_volume = row["NearVolume"]
    result: dict[str, str] = {
        "case": case,
        "requested_time": f"{requested:.16e}",
        "actual_time": f"{row['time']:.16e}",
        "time_offset": f"{row['time'] - requested:.16e}",
        "regular_error_L1": "not-recorded",
        "regular_error_Linf": "not-recorded",
        "lapse_min": f"{-row['minus-alpha-min']:.16e}",
    }
    for column in REF_GH_COLUMNS[1:]:
        result[column] = f"{row[column]:.16e}"
    for label in ("GH", "Reduction", "Curl", "PsiError", "Pi", "Phi"):
        l2sq = row[f"{label}-L2sq"]
        result[f"{label}-sqrt_integrated_L2"] = f"{root(l2sq):.16e}"
        result[f"{label}-RMS_L2"] = f"{root(l2sq / volume):.16e}"
    for label in ("GHnear", "ReductionNear", "CurlNear"):
        l2sq = row[f"{label}-L2sq"]
        result[f"{label}-sqrt_integrated_L2"] = f"{root(l2sq):.16e}"
        result[f"{label}-RMS_L2"] = (
            f"{root(l2sq / near_volume):.16e}" if near_volume > 0.0 else "nan"
        )
    return result


def parse_case(text: str) -> tuple[str, Path]:
    label, separator, raw_path = text.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--case must be LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", type=parse_case, required=True,
        help="case label and Ref-GH history file, as LABEL=PATH (repeatable)",
    )
    parser.add_argument(
        "--time", action="append", type=float,
        help="requested matched output time; defaults to the campaign table times",
    )
    parser.add_argument("--output", type=Path, required=True, help="TSV output path")
    args = parser.parse_args()

    requested_times = tuple(args.time) if args.time else DEFAULT_TIMES
    output_rows: list[dict[str, str]] = []
    for label, path in args.case:
        rows = read_history(path)
        for requested in requested_times:
            output_rows.append(enriched(label, requested, closest(rows, requested)))

    fieldnames = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
