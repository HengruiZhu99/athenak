#!/usr/bin/env python3
"""Gate the fully subtracted 64/96/128 stationary Ref-GH ladder."""

import argparse
import csv
import json
import math
import pathlib
import re


RESOLUTIONS = (64, 96, 128)
EXACT_SECTORS = (
    ("actual", "Hhat"),
    ("actual", "theta"),
    ("actual", "Upsilon"),
    ("ordinary_gauge_increment", "Pi"),
    ("driver", "Hhat"),
    ("driver", "theta"),
    ("driver", "Upsilon"),
    ("KO", "Hhat"),
    ("KO", "theta"),
    ("KO", "Upsilon"),
)


def header_float(text, name):
    match = re.search(r"(?:^|\s){}=([^\s]+)".format(re.escape(name)), text)
    if match is None:
        raise ValueError("missing {}".format(name))
    return float(match.group(1))


def read_sectors(path):
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    table = {}
    for row in csv.DictReader(lines, delimiter="\t"):
        table[(row["sector"], row["family"])] = row
    return text, table


def match_values(text, pattern, names):
    match = re.search(pattern, text)
    if match is None:
        raise ValueError("missing log record: {}".format(pattern))
    return dict(zip(names, (float(value) for value in match.groups())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows = []
    for nx in RESOLUTIONS:
        directory = args.run_root / "fixed_point_n{}".format(nx)
        basename = "fixed_point_n{}".format(nx)
        log_text = (directory / "run.log").read_text(encoding="utf-8")
        sector_text, sectors = read_sectors(
            directory / "{}.ref_gh_rhs_sectors.tsv".format(basename)
        )
        legacy = match_values(
            log_text,
            r"stationary physical-target audit: "
            r"\|F-H_constraint\|_Linf=([^,]+), "
            r"\|tildeGamma\|_Linf=([^,]+), "
            r"\|stored_Hhat_A\|_Linf=([^,]+), "
            r"\|stored_theta_A\|_Linf=([^\s]+)",
            ("legacy_full_target", "legacy_conformal_gamma",
             "stored_Hhat", "stored_theta"),
        )
        residual = match_values(
            log_text,
            r"stationary residual-target audit: "
            r"\|deltaF_A\|_Linf=([^,]+), "
            r"\|delta_tildeGamma\|_Linf=([^,]+), "
            r"\|delta_beta\|_Linf=([^\s]+)",
            ("deltaF", "delta_conformal_gamma", "delta_beta"),
        )
        rhs = match_values(
            log_text,
            r"stationary initial RHS Linf = ([^,]+), component=([^,]+), "
            r"radius=([^,]+), coordinate reference Ricci Linf=([^,]+), "
            r"frame reference Ricci Linf=([^\s]+)",
            ("total_rhs", "total_rhs_component", "total_rhs_radius",
             "coordinate_reference_ricci", "frame_reference_ricci"),
        )
        h = 4.0 / float(nx)
        row = {
            "nx": nx,
            "h": h,
            "minimum_included_radius": math.sqrt(12.75) * h,
            "minimum_included_r_over_h": math.sqrt(12.75),
            "reproduction_conditioned_linf": header_float(
                sector_text, "reproduction_conditioned_linf"),
            "production_rerun_conditioned_linf": header_float(
                sector_text, "production_rerun_conditioned_linf"),
            "tolerance": header_float(sector_text, "tolerance"),
        }
        row.update(legacy)
        row.update(residual)
        row.update(rhs)
        for sector, family in EXACT_SECTORS:
            key = "{}_{}".format(sector, family)
            if (sector, family) not in sectors:
                raise ValueError("N={} missing sector {}".format(nx, key))
            row[key] = float(sectors[(sector, family)]["maximum"])
        rows.append(row)

    exact_keys = (
        "stored_Hhat", "stored_theta", "deltaF",
        "delta_conformal_gamma", "delta_beta",
    ) + tuple("{}_{}".format(sector, family)
              for sector, family in EXACT_SECTORS)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise SystemExit("N={} nonfinite {}".format(row["nx"], key))
        if row["reproduction_conditioned_linf"] > row["tolerance"]:
            raise SystemExit("N={} reproduction gate failed".format(row["nx"]))
        if row["production_rerun_conditioned_linf"] > row["tolerance"]:
            raise SystemExit("N={} production rerun gate failed".format(row["nx"]))
        for key in exact_keys:
            if row[key] != 0.0:
                raise SystemExit(
                    "N={} expected exact-zero {} but found {:.17e}".format(
                        row["nx"], key, row[key]
                    )
                )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "resolutions": list(RESOLUTIONS),
        "puncture_stencil_radius_cells": 3,
        "all_required_gauge_residuals_exact_zero": True,
        "moving_r_over_h_gauge_envelope_removed": True,
        "positive_time_cycles": 0,
        "claim_boundary": (
            "Cycle-zero fully subtracted fixed-point ladder only; no evolved "
            "stability or convergence claim."
        ),
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
