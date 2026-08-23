#!/usr/bin/env python3
"""Analyze the three-resolution Ref-GH transition survivor gate."""

import argparse
import json
import math
from pathlib import Path

from analyze_transition_path_medium import extract, merged_rows, nearest


SPACINGS = {"coarse": 1.0 / 16.0, "medium": 1.0 / 24.0, "fine": 1.0 / 32.0}
TARGETS = (0.5, 1.0, 2.0, 2.2, 3.0, 4.0)
ORDER_METRICS = (
    "GH_L2", "reduction_L2", "curl_L2", "Psi_error_L2",
    "all_H_L2", "all_M_L2", "lt1_H_L2", "lt1_M_L2",
    "lt2_H_L2", "lt2_M_L2", "r2to4_H_L2", "r2to4_M_L2",
)


def parse_directories(value):
    directories = [Path(item) for item in value.split(",")]
    if not all(path.is_dir() for path in directories):
        raise ValueError(f"missing case directory in {value}")
    return directories


def l2(total, volume):
    return math.sqrt(max(total, 0.0) / volume) if volume > 0.0 else math.nan


def adm_region(row, base, prefix, result):
    volume = row[base + 6]
    result[f"{prefix}_H_L1"] = row[base] / volume if volume > 0.0 else math.nan
    result[f"{prefix}_H_L2"] = l2(row[base + 1], volume)
    result[f"{prefix}_H_Linf"] = row[base + 2]
    result[f"{prefix}_M_L1"] = row[base + 3] / volume if volume > 0.0 else math.nan
    result[f"{prefix}_M_L2"] = l2(row[base + 4], volume)
    result[f"{prefix}_M_Linf"] = row[base + 5]


def load_case(directories):
    return {
        "ref": merged_rows(directories, ".ref_gh.hst"),
        "user": merged_rows(directories, ".user.hst"),
        "adm0": merged_rows(directories, ".adm_common0.hst"),
        "adm1": merged_rows(directories, ".adm_common1.hst"),
        "adm2": merged_rows(directories, ".adm_common2.hst"),
        "adm5": merged_rows(directories, ".adm_common5.hst"),
    }


def record_at(case, target):
    rows = {name: nearest(values, target) for name, values in case.items()}
    if any(value is None for value in rows.values()):
        return None
    result = extract(rows["ref"], rows["user"])
    result["target"] = target
    adm_region(rows["adm0"], 2, "all", result)
    adm_region(rows["adm0"], 9, "lt1", result)
    adm_region(rows["adm1"], 2, "lt2", result)
    adm_region(rows["adm1"], 9, "r2to4", result)
    adm_region(rows["adm2"], 2, "r4to8", result)
    adm_region(rows["adm5"], 9, "if2", result)
    return result


def order(coarse, fine, h_coarse, h_fine):
    if coarse > 0.0 and fine > 0.0:
        return math.log(coarse / fine) / math.log(h_coarse / h_fine)
    return math.nan


def main():
    parser = argparse.ArgumentParser()
    for label in ("coarse", "medium", "fine"):
        parser.add_argument(f"--{label}", required=True,
                            help="comma-separated chronological segment directories")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", type=float, default=TARGETS)
    args = parser.parse_args()

    cases = {label: load_case(parse_directories(getattr(args, label)))
             for label in ("coarse", "medium", "fine")}
    records = []
    for target in args.targets:
        values = {label: record_at(case, target) for label, case in cases.items()}
        if any(value is None for value in values.values()):
            continue
        orders = {}
        for metric in ORDER_METRICS:
            orders[metric] = {
                "coarse_medium": order(values["coarse"][metric],
                                         values["medium"][metric],
                                         SPACINGS["coarse"], SPACINGS["medium"]),
                "medium_fine": order(values["medium"][metric],
                                       values["fine"][metric],
                                       SPACINGS["medium"], SPACINGS["fine"]),
                "resolution_reversed": not (
                    values["coarse"][metric] >= values["medium"][metric]
                    >= values["fine"][metric]),
            }
        records.append({"target": target, "resolutions": values, "orders": orders})

    payload = {"schema": "ref-gh-transition-resolution-gate-v1",
               "candidate": "fixed_core_tau8_compatible",
               "spacings": SPACINGS, "records": records}
    json_path = Path(str(args.output_prefix) + ".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    with tsv_path.open("w") as stream:
        stream.write("target\tmetric\tcoarse\tmedium\tfine\torder_cm\torder_mf\t"
                     "resolution_reversed\n")
        for item in records:
            for metric in ORDER_METRICS:
                values = item["resolutions"]
                info = item["orders"][metric]
                stream.write("\t".join(str(value) for value in (
                    item["target"], metric, values["coarse"][metric],
                    values["medium"][metric], values["fine"][metric],
                    info["coarse_medium"], info["medium_fine"],
                    info["resolution_reversed"])) + "\n")
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
