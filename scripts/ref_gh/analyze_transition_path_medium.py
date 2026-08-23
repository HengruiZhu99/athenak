#!/usr/bin/env python3
"""Create header-aware summaries for one or more medium Ref-GH path segments."""

import argparse
import csv
import json
import math
from pathlib import Path


TARGETS = (0.5, 1.0, 1.5, 1.6, 1.8, 2.0, 2.2, 3.0, 4.0)
MAXLOC_NAMES = (
    "GH_constraint", "reduction_constraint", "curl_constraint", "Psi", "Pi", "Phi",
    "reference_Ricci", "reference_Riemann", "spin_connection", "spin_derivative",
    "source_curvature", "source_QQ", "source_DeltaDelta", "source_damping",
    "source_frame_correction", "shift_lapse_ratio",
)


def data_rows(path):
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip() and not line.startswith("#"):
                rows.append([float(value) for value in line.split()])
    if not rows:
        raise RuntimeError(f"no data rows in {path}")
    return rows


def one_file(directory, suffix):
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *{suffix} in {directory}, got {matches}")
    return matches[0]


def merged_rows(directories, suffix):
    by_time = {}
    for directory in directories:
        for row in data_rows(one_file(directory, suffix)):
            by_time[row[0]] = row
    return [by_time[time] for time in sorted(by_time)]


def status(directory):
    path = directory / "segment_status.txt"
    result = {"path": str(directory), "exit_status": None, "target": None}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("exit_status="):
                result["exit_status"] = int(line.split("=", 1)[1])
            elif line.startswith("target="):
                result["target"] = float(line.split("=", 1)[1])
    return result


def nearest(rows, target, tolerance=0.03):
    row = min(rows, key=lambda value: abs(value[0] - target))
    return row if abs(row[0] - target) <= tolerance else None


def l2(total, volume):
    return math.sqrt(max(total, 0.0) / volume) if volume > 0.0 else math.nan


def extract(ref, user):
    volume = ref[11]
    return {
        "time": ref[0], "dt": ref[1],
        "GH_L2": l2(ref[2], volume),
        "reduction_L2": l2(ref[3], volume),
        "curl_L2": l2(ref[4], volume),
        "Psi_error_L2": l2(ref[5], volume),
        "Pi_L2": l2(ref[6], volume),
        "Phi_L2": l2(ref[7], volume),
        "bad_state": ref[21], "Q_Linf": ref[22], "Delta_Linf": ref[23],
        "reference_Ricci_Linf": ref[24], "source_curvature_Linf": ref[26],
        "source_QQ_Linf": ref[27], "source_DeltaDelta_Linf": ref[28],
        "source_damping_Linf": ref[29], "source_frame_Linf": ref[30],
        "physical_lapse_min": -ref[13], "regular_max": ref[14],
        "metric_condition_max": ref[15], "characteristic_speed_max": ref[17],
        "effective_CFL": ref[18], "detg_min": -ref[19],
        "delta_q": user[2], "delta_p": user[4], "e_G": user[6],
        "e_alpha": user[7], "G_lambda_min": user[9],
        "G_lambda_max": user[10], "G_condition_max": user[13],
        "relative_lapse_min": user[14], "relative_lapse_max": user[15],
        "relative_v2_max": user[16], "Psi_max": user[17],
        "inverse_Psi_max": user[18], "r_core": user[22],
        "transition_amplitude": user[23], "feedback": user[24],
        "shell_valid": user[25], "controller_generation": user[26],
    }


def maxloc_rows(directories):
    result = []
    for directory in directories:
        path = one_file(directory, ".ref_gh_maxloc.tsv")
        with path.open() as stream:
            result.extend(csv.DictReader(stream, delimiter="\t"))
    return result


def extract_maxloc(rows, final_time, tolerance=0.03):
    times = sorted({float(row["time"]) for row in rows})
    time = min(times, key=lambda value: abs(value - final_time))
    if abs(time - final_time) > tolerance:
        return {}
    result = {}
    for row in rows:
        if abs(float(row["time"]) - time) > 1.0e-12:
            continue
        name = row["diagnostic"]
        if name in MAXLOC_NAMES:
            result[name] = {
                key: (float(row[key]) if key not in ("level", "rank", "gid")
                      else int(row[key]))
                for key in ("maximum", "radius", "r_over_r_core", "level", "rank", "gid")
            }
    return result


def parse_case(specification):
    if "=" not in specification:
        raise ValueError(f"case must be LABEL=DIR[,DIR...], got {specification}")
    label, paths = specification.split("=", 1)
    directories = [Path(value) for value in paths.split(",")]
    return label, directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True,
                        help="LABEL=DIR[,DIR...] in chronological segment order")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", type=float, default=TARGETS)
    args = parser.parse_args()

    payload = {"schema": "ref-gh-transition-path-medium-summary-v1", "cases": {}}
    table = []
    for specification in args.case:
        label, directories = parse_case(specification)
        ref_rows = merged_rows(directories, ".ref_gh.hst")
        user_rows = merged_rows(directories, ".user.hst")
        records = []
        for target in args.targets:
            ref = nearest(ref_rows, target)
            user = nearest(user_rows, target)
            if ref is not None and user is not None:
                record = extract(ref, user)
                record["target"] = target
                records.append(record)
                table.append((label, record))
        final = extract(ref_rows[-1], user_rows[-1])
        payload["cases"][label] = {
            "directories": [str(path) for path in directories],
            "segments": [status(path) for path in directories],
            "records": records, "final": final,
            "final_max_locations": extract_maxloc(maxloc_rows(directories), final["time"]),
        }

    json_path = Path(str(args.output_prefix) + ".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    columns = ("target", "time", "dt", "GH_L2", "reduction_L2", "curl_L2",
               "Psi_error_L2", "Pi_L2", "Phi_L2", "bad_state",
               "physical_lapse_min", "G_lambda_min", "G_condition_max",
               "relative_lapse_min", "relative_v2_max", "r_core",
               "transition_amplitude", "feedback")
    with tsv_path.open("w") as stream:
        stream.write("case\t" + "\t".join(columns) + "\n")
        for label, record in table:
            stream.write(label + "\t" + "\t".join(str(record[key]) for key in columns)
                         + "\n")
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
