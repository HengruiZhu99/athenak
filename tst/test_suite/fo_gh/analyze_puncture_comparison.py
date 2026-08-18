#!/usr/bin/env python3
"""Analyze common-ADM FO-GH/Z4c puncture histories and initial-data parity."""

import argparse
import csv
import glob
import math
import os
import re
import sys


REGIONS = ("all", "lt1", "lt2", "r2to4", "r4to8", "gt8",
           "if64", "if32", "if16", "if8", "if4", "if2")
DX = (1.0 / 16.0, 1.0 / 24.0, 1.0 / 32.0)


def read_history(path):
    labels = None
    rows = {}
    label_re = re.compile(r"\[\d+\]=([^\s]+)")
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                found = label_re.findall(line)
                if found and found[0] == "time":
                    labels = found
                continue
            if not line.strip():
                continue
            if labels is None:
                raise RuntimeError(f"missing labeled header in {path}")
            values = [float(value) for value in line.split()]
            if len(values) != len(labels):
                raise RuntimeError(f"column mismatch in {path}: {len(values)} != {len(labels)}")
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"nonfinite history value in {path}")
            rows[round(values[0], 12)] = dict(zip(labels, values))
    return rows


def read_run(directory):
    merged = {}
    paths = sorted(glob.glob(os.path.join(directory, "*.adm_common[0-5].hst")))
    if len(paths) != 6:
        raise RuntimeError(f"expected six common-ADM histories in {directory}, found {len(paths)}")
    for path in paths:
        for time, row in read_history(path).items():
            merged.setdefault(time, {}).update(row)
    return merged


def normalized(row, region):
    volume = row[region + "V"]
    if volume <= 0.0:
        return None
    return {
        "H_L1": row[region + "H1"] / volume,
        "H_L2": math.sqrt(max(0.0, row[region + "H2"] / volume)),
        "H_Linf": row[region + "Hi"],
        "M_L1": row[region + "M1"] / volume,
        "M_L2": math.sqrt(max(0.0, row[region + "M2"] / volume)),
        "M_Linf": row[region + "Mi"],
    }


def order(a, b, ha, hb):
    if a <= 0.0 or b <= 0.0 or a == b:
        return float("nan")
    return math.log(a / b) / math.log(ha / hb)


def summarize(args):
    writer_stream = open(args.output, "w", newline="", encoding="utf-8") \
        if args.output else sys.stdout
    try:
        writer = csv.writer(writer_stream)
        writer.writerow(("formulation", "time", "region", "norm", "coarse", "medium",
                         "fine", "p_coarse_medium", "p_medium_fine",
                         "resolution_growing"))
        for formulation, directories in (("FO-GH", args.fo_gh), ("Z4c", args.z4c)):
            runs = [read_run(directory) for directory in directories]
            common_times = sorted(set(runs[0]).intersection(runs[1], runs[2]))
            if not common_times:
                raise RuntimeError(f"no common output times for {formulation}")
            for time in common_times:
                for region in REGIONS:
                    values = [normalized(run[time], region) for run in runs]
                    if any(value is None for value in values):
                        continue
                    for norm in ("H_L1", "H_L2", "H_Linf", "M_L1", "M_L2", "M_Linf"):
                        errors = [value[norm] for value in values]
                        writer.writerow((formulation, f"{time:.12g}", region, norm,
                                         *(f"{value:.17e}" for value in errors),
                                         f"{order(errors[0], errors[1], DX[0], DX[1]):.9g}",
                                         f"{order(errors[1], errors[2], DX[1], DX[2]):.9g}",
                                         str(errors[2] > errors[1] > errors[0]).lower()))
    finally:
        if args.output:
            writer_stream.close()


def read_adm_slice(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            values = [float(value) for value in line.split()]
            # gid, local index, x, then gamma_ij, K_ij, and psi4.  FO-GH has
            # four additional lapse/shift columns that are intentionally ignored here.
            rows.append((values[2], tuple(values[3:16])))
    return sorted(rows)


def read_gauge_slice(path, indices):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            values = [float(value) for value in line.split()]
            rows.append((values[2], tuple(values[index] for index in indices)))
    return sorted(rows)


def compare_initial(args):
    fo_rows = read_adm_slice(args.fo_gh_tab)
    z4c_rows = read_adm_slice(args.z4c_tab)
    if len(fo_rows) != len(z4c_rows):
        raise RuntimeError(f"slice length mismatch: {len(fo_rows)} != {len(z4c_rows)}")
    maximum = 0.0
    for fo_row, z4c_row in zip(fo_rows, z4c_rows):
        for fo_value, z4c_value in zip((fo_row[0],) + fo_row[1],
                                       (z4c_row[0],) + z4c_row[1]):
            maximum = max(maximum, abs(fo_value - z4c_value))
    print(f"common ADM gamma,K,psi4 max_abs={maximum:.17e}")
    fo_gauge = read_gauge_slice(args.fo_gh_tab, (16, 17, 18, 19))
    z4c_gauge = read_gauge_slice(args.z4c_state_tab, (21, 22, 23, 24))
    if len(fo_gauge) != len(z4c_gauge):
        raise RuntimeError(f"gauge slice length mismatch: {len(fo_gauge)} != {len(z4c_gauge)}")
    gauge_maximum = 0.0
    for fo_row, z4c_row in zip(fo_gauge, z4c_gauge):
        for fo_value, z4c_value in zip((fo_row[0],) + fo_row[1],
                                       (z4c_row[0],) + z4c_row[1]):
            gauge_maximum = max(gauge_maximum, abs(fo_value - z4c_value))
    print(f"common ADM alpha,beta max_abs={gauge_maximum:.17e}")
    maximum = max(maximum, gauge_maximum)
    if maximum > args.tolerance:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--fo-gh", nargs=3, required=True,
                         metavar=("COARSE", "MEDIUM", "FINE"))
    summary.add_argument("--z4c", nargs=3, required=True,
                         metavar=("COARSE", "MEDIUM", "FINE"))
    summary.add_argument("--output")
    summary.set_defaults(function=summarize)
    parity = subparsers.add_parser("compare-initial")
    parity.add_argument("fo_gh_tab")
    parity.add_argument("z4c_tab")
    parity.add_argument("z4c_state_tab")
    parity.add_argument("--tolerance", type=float, default=2.0e-14)
    parity.set_defaults(function=compare_initial)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
