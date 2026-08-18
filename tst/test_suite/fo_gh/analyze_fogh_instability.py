#!/usr/bin/env python3
"""Summarize an incomplete three-resolution FO-GH puncture campaign.

This deliberately accepts runs that stalled before the requested final time.  It
extracts the timestep-collapse evidence from Athena logs and normalizes the
fixed-coordinate common ADM histories without using the native lapse mask.
"""

import argparse
import csv
import glob
import math
import os
import re


RESOLUTIONS = ("coarse", "medium", "fine")
DX = {"coarse": 1.0 / 16.0, "medium": 1.0 / 24.0, "fine": 1.0 / 32.0}
REGIONS = ("all", "lt1", "lt2", "r2to4", "r4to8", "gt8",
           "if64", "if32", "if16", "if8", "if4", "if2")
TARGET_TIMES = (0.0, 1.0, 2.0)
LABEL_RE = re.compile(r"\[\d+\]=([^\s]+)")
STEP_RE = re.compile(r"cycle=(\d+)\s+time=([0-9.eE+-]+)\s+dt=([0-9.eE+-]+)")


def read_history(path):
    labels = None
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                found = LABEL_RE.findall(line)
                if found and found[0] == "time":
                    labels = found
                continue
            if not line.strip():
                continue
            if labels is None:
                raise RuntimeError("missing labeled header in {}".format(path))
            values = [float(value) for value in line.split()]
            if len(values) != len(labels):
                raise RuntimeError("column mismatch in {}".format(path))
            rows.append(dict(zip(labels, values)))
    if not rows:
        raise RuntimeError("empty history {}".format(path))
    return rows


def read_common(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.adm_common[0-5].hst")))
    if len(paths) != 6:
        raise RuntimeError("expected six common histories in {}".format(directory))
    series = [read_history(path) for path in paths]
    count = min(len(rows) for rows in series)
    merged = []
    for index in range(count):
        row = {}
        times = []
        for rows in series:
            row.update(rows[index])
            times.append(rows[index]["time"])
        if max(times) - min(times) > 1.0e-10:
            raise RuntimeError("common-history time mismatch in {}".format(directory))
        merged.append(row)
    return merged


def nearest(rows, target):
    return min(rows, key=lambda row: abs(row["time"] - target))


def norm(row, region, family):
    volume = row[region + "V"]
    if volume <= 0.0:
        return float("nan")
    return math.sqrt(max(0.0, row[region + family + "2"] / volume))


def observed_order(coarser, finer, dx_coarser, dx_finer):
    if coarser <= 0.0 or finer <= 0.0:
        return float("nan")
    return math.log(coarser / finer) / math.log(dx_coarser / dx_finer)


def read_steps(path):
    steps = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            match = STEP_RE.search(line)
            if match:
                steps.append((int(match.group(1)), float(match.group(2)),
                              float(match.group(3))))
    if not steps:
        raise RuntimeError("no timestep records in {}".format(path))
    return steps


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def analyze(args):
    os.makedirs(args.output_dir, exist_ok=True)
    common = {}
    native = {}
    status_rows = []
    for resolution in RESOLUTIONS:
        run_dir = os.path.join(args.root, "runs", "fogh_" + resolution)
        common[resolution] = read_common(run_dir)
        native_paths = glob.glob(os.path.join(run_dir, "*.fo_gh.hst"))
        if len(native_paths) != 1:
            raise RuntimeError("expected one native history in {}".format(run_dir))
        native[resolution] = read_history(native_paths[0])
        steps = read_steps(os.path.join(args.root, "logs", "fogh_" + resolution + ".log"))
        collapsed = next((step for step in steps if step[2] < args.collapse_dt), None)
        if collapsed is None:
            collapsed = (steps[-1][0], steps[-1][1], steps[-1][2])
        previous = max((step for step in steps if step[0] < collapsed[0]),
                       default=steps[0], key=lambda step: step[0])
        last_native = native[resolution][-1]
        status_rows.append((
            resolution, "{:.17e}".format(DX[resolution]),
            collapsed[0], "{:.17e}".format(collapsed[1]),
            "{:.17e}".format(collapsed[2]), previous[0],
            "{:.17e}".format(previous[1]), "{:.17e}".format(previous[2]),
            "{:.17e}".format(last_native["time"]),
            "{:.17e}".format(last_native["H-L2sq"]),
            "{:.17e}".format(last_native["M-L2sq"]),
            "{:.17e}".format(last_native["Cp-L2sq"]),
            "{:.17e}".format(last_native["c-L2sq"])))

    write_csv(os.path.join(args.output_dir, "failure_summary.csv"),
              ("resolution", "dx_min", "collapse_cycle", "collapse_time",
               "collapse_dt", "previous_sample_cycle", "previous_sample_time",
               "previous_sample_dt", "last_native_history_time", "H_L2sq",
               "M_L2sq", "Cperp_L2sq", "c_L2sq"), status_rows)

    spatial_rows = []
    for target in TARGET_TIMES:
        for resolution in RESOLUTIONS:
            row = nearest(common[resolution], target)
            for region in REGIONS:
                spatial_rows.append((
                    "{:.1f}".format(target), resolution,
                    "{:.17e}".format(row["time"]), region,
                    "{:.17e}".format(norm(row, region, "H")),
                    "{:.17e}".format(norm(row, region, "M"))))
    write_csv(os.path.join(args.output_dir, "fixed_region_l2.csv"),
              ("target_time", "resolution", "sample_time", "region",
               "H_L2", "M_L2"), spatial_rows)

    convergence_rows = []
    for target in TARGET_TIMES:
        selected = {resolution: nearest(common[resolution], target)
                    for resolution in RESOLUTIONS}
        for region in REGIONS:
            for family in ("H", "M"):
                values = {resolution: norm(selected[resolution], region, family)
                          for resolution in RESOLUTIONS}
                convergence_rows.append((
                    "{:.1f}".format(target), region, family + "_L2",
                    "{:.17e}".format(values["coarse"]),
                    "{:.17e}".format(values["medium"]),
                    "{:.17e}".format(values["fine"]),
                    "{:.9g}".format(observed_order(
                        values["coarse"], values["medium"],
                        DX["coarse"], DX["medium"])),
                    "{:.9g}".format(observed_order(
                        values["medium"], values["fine"],
                        DX["medium"], DX["fine"]))))
    write_csv(os.path.join(args.output_dir, "fixed_region_convergence.csv"),
              ("target_time", "region", "norm", "coarse", "medium", "fine",
               "p_coarse_medium", "p_medium_fine"), convergence_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="partial artifact directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collapse-dt", type=float, default=1.0e-100)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
