#!/usr/bin/env python3
"""Compare compact analytic and generic-oracle evolved Ref-GH histories."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def load(path: Path) -> tuple[list[str], list[list[float]]]:
    names: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#  ["):
            names = re.findall(r"\[\d+\]=([^ ]+)", line)
        elif line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])
    if not rows:
        raise RuntimeError(f"no history rows in {path}")
    if not names:
        names = [f"column_{index + 1}" for index in range(len(rows[0]))]
    return names, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analytic", type=Path)
    parser.add_argument("generic", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=5.0e-12)
    args = parser.parse_args()

    maxima: list[dict[str, object]] = []
    excluded = {"frame-Ricc", "coordinate"}
    for analytic_path in sorted(args.analytic.glob("*.hst")):
        generic_path = args.generic / analytic_path.name
        names, analytic_rows = load(analytic_path)
        generic_names, generic_rows = load(generic_path)
        if names != generic_names or len(analytic_rows) != len(generic_rows):
            raise RuntimeError(f"history shape mismatch for {analytic_path.name}")
        maximum = 0.0
        location: dict[str, object] = {}
        for row_index, (analytic_row, generic_row) in enumerate(
                zip(analytic_rows, generic_rows)):
            if analytic_row[0] <= 0.0:
                continue
            for column, (analytic, generic) in enumerate(
                    zip(analytic_row, generic_row)):
                name = names[column] if column < len(names) else f"column_{column + 1}"
                if analytic_path.name.endswith(".ref_gh.hst") and name in excluded:
                    continue
                if not math.isfinite(analytic) or not math.isfinite(generic):
                    raise RuntimeError(
                        f"nonfinite value in {analytic_path.name} row {row_index}")
                error = abs(analytic - generic)/max(
                    1.0, abs(analytic), abs(generic))
                if error > maximum:
                    maximum = error
                    location = {
                        "row": row_index,
                        "column": column + 1,
                        "name": name,
                        "analytic": analytic,
                        "generic": generic,
                    }
        maxima.append({
            "file": analytic_path.name,
            "conditioned_linf": maximum,
            "location": location,
        })

    overall = max((entry["conditioned_linf"] for entry in maxima), default=0.0)
    result = {
        "analytic": str(args.analytic),
        "generic": str(args.generic),
        "method": "positive-time rows; abs(a-b)/max(1,abs(a),abs(b))",
        "excluded_ref_gh_columns": sorted(excluded),
        "tolerance": args.tolerance,
        "overall_conditioned_linf": overall,
        "pass": overall <= args.tolerance,
        "files": maxima,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
