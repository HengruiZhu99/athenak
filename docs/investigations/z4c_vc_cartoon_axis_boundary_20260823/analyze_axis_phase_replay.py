#!/usr/bin/env python3
"""Compare central-axis RHS phases and named terms across N128/N256/N512."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


RESOLUTIONS = (128, 256, 512)
TAGS = ("0000", "0125", "0250", "0500")
PHASE_KEYS = (
    "raw_pre_axis",
    "post_axis_pre_ko",
    "axis_correction",
    "ko_contribution",
    "post_ko",
)


def fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        result[key] = value
    return result


def load(path: Path) -> tuple[dict[tuple[int, str, str], float],
                              dict[tuple[int, str], float]]:
    phase: dict[tuple[int, str, str], float] = {}
    terms: dict[tuple[int, str], float] = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("Z4C_AXIS_RHS_PHASE_DIAGNOSTIC "):
                row = fields(line)
                stage = int(row["stage"])
                variable = row["variable"]
                for key in PHASE_KEYS:
                    phase[(stage, variable, key)] = float(row[key])
            elif line.startswith("Z4C_AXIS_TERM_POINT_DIAGNOSTIC "):
                row = fields(line)
                terms[(int(row["stage"]), row["term"])] = float(row["value"])
    return phase, terms


def convergence(values: list[float]) -> tuple[float, float, float, float]:
    d128_256 = abs(values[0] - values[1])
    d256_512 = abs(values[1] - values[2])
    order = (math.log2(d128_256 / d256_512)
             if d128_256 > 0.0 and d256_512 > 0.0 else math.nan)
    relative = d256_512 / max(max(abs(value) for value in values), 1.0e-300)
    return d128_256, d256_512, order, relative


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tau0-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for resolution in RESOLUTIONS:
        for tag in TAGS:
            root = (args.tau0_root
                    if tag == "0000" and args.tau0_root is not None
                    else args.root)
            directory = root / f"N{resolution}" / f"tau{tag}"
            path = directory / "z4c_rhs_stage_rank0.log.gz"
            if not path.exists():
                path = directory / "diagnostic" / "z4c_rhs_stage_rank0.log.gz"
            loaded[(resolution, tag)] = load(path)

    phase_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    for tag in TAGS:
        phase_keys = set(loaded[(128, tag)][0])
        term_keys = set(loaded[(128, tag)][1])
        for resolution in RESOLUTIONS[1:]:
            assert set(loaded[(resolution, tag)][0]) == phase_keys
            assert set(loaded[(resolution, tag)][1]) == term_keys
        for stage, variable, phase in sorted(phase_keys):
            values = [loaded[(resolution, tag)][0][(stage, variable, phase)]
                      for resolution in RESOLUTIONS]
            d12, d23, order, relative = convergence(values)
            phase_rows.append({
                "axis_tau": int(tag) / 1000.0,
                "stage": stage,
                "variable": variable,
                "phase": phase,
                "N128": values[0], "N256": values[1], "N512": values[2],
                "abs_N128_N256": d12, "abs_N256_N512": d23,
                "observed_order": order, "relative_N256_N512": relative,
            })
        for stage, term in sorted(term_keys):
            values = [loaded[(resolution, tag)][1][(stage, term)]
                      for resolution in RESOLUTIONS]
            d12, d23, order, relative = convergence(values)
            term_rows.append({
                "axis_tau": int(tag) / 1000.0,
                "stage": stage,
                "term": term,
                "N128": values[0], "N256": values[1], "N512": values[2],
                "abs_N128_N256": d12, "abs_N256_N512": d23,
                "observed_order": order, "relative_N256_N512": relative,
            })

    write_csv(args.output / "phase_convergence.csv", phase_rows)
    write_csv(args.output / "term_convergence.csv", term_rows)

    def significant(row: dict[str, object]) -> bool:
        return float(row["abs_N256_N512"]) > 1.0e-12

    raw = [row for row in phase_rows
           if row["phase"] == "raw_pre_axis" and significant(row)]
    raw.sort(key=lambda row: (float(row["observed_order"])
                              if math.isfinite(float(row["observed_order"]))
                              else math.inf,
                              -float(row["abs_N256_N512"])))
    term_rank = [row for row in term_rows if significant(row)]
    term_rank.sort(key=lambda row: (float(row["observed_order"])
                                    if math.isfinite(float(row["observed_order"]))
                                    else math.inf,
                                    -float(row["abs_N256_N512"])))

    summary = {
        "interpretation_boundary": (
            "Pointwise three-resolution differences diagnose the first RHS phase "
            "and named term family that loses refinement compatibility; they do not "
            "by themselves identify a unique continuum-formula defect."),
        "raw_pre_axis_worst_significant": raw[:30],
        "term_worst_significant": term_rank[:30],
        "axis_projection_max_abs": max(
            abs(float(row[key]))
            for row in phase_rows for key in ("N128", "N256", "N512")
            if row["phase"] == "axis_correction"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Central-axis phase/term convergence summary",
        "",
        summary["interpretation_boundary"],
        "",
        f"Maximum recorded absolute axis projection correction: "
        f"{summary['axis_projection_max_abs']:.9e}.",
        "",
        "## Worst significant raw pre-axis RHS rows",
        "",
        "| tau | stage | variable | p | |N256-N512| |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in raw[:20]:
        lines.append(
            f"| {row['axis_tau']} | {row['stage']} | {row['variable']} | "
            f"{float(row['observed_order']):.6g} | "
            f"{float(row['abs_N256_N512']):.6e} |")
    lines.extend([
        "", "## Worst significant named term rows", "",
        "| tau | stage | term | p | |N256-N512| |",
        "|---:|---:|---|---:|---:|",
    ])
    for row in term_rank[:30]:
        lines.append(
            f"| {row['axis_tau']} | {row['stage']} | {row['term']} | "
            f"{float(row['observed_order']):.6g} | "
            f"{float(row['abs_N256_N512']):.6e} |")
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
