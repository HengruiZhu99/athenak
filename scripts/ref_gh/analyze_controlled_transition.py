#!/usr/bin/env python3
"""Summarize matched Ref-GH controlled-transition histories."""

import argparse
import json
import math
from pathlib import Path


TARGETS = (0.5, 1.0, 2.0, 4.0)
SPACINGS = {"coarse": 1.0 / 16.0, "medium": 1.0 / 24.0,
            "fine": 1.0 / 32.0}


def rows(path):
    result = []
    with path.open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            result.append([float(token) for token in line.split()])
    if not result:
        raise RuntimeError("no data rows in {}".format(path))
    return result


def matched(data, target, tolerance=0.026):
    row = min(data, key=lambda item: abs(item[0] - target))
    if abs(row[0] - target) > tolerance:
        raise RuntimeError("no row near t={} (nearest {})".format(target, row[0]))
    return row


def safe_l2(total, volume):
    return math.sqrt(max(total, 0.0) / volume) if volume > 0.0 else float("nan")


def finite(value):
    return math.isfinite(value)


def pair_order(coarse, fine, hcoarse, hfine):
    if coarse > 0.0 and fine > 0.0:
        return math.log(coarse / fine) / math.log(hcoarse / hfine)
    return float("nan")


def load_case(directory):
    candidates = sorted(directory.glob("*.ref_gh.hst"))
    if len(candidates) != 1:
        raise RuntimeError("expected one ref_gh history in {}".format(directory))
    stem = candidates[0].name[:-len(".ref_gh.hst")]
    return {
        "ref": rows(candidates[0]),
        "user": rows(directory / (stem + ".user.hst")),
        "adm0": rows(directory / (stem + ".adm_common0.hst")),
        "adm1": rows(directory / (stem + ".adm_common1.hst")),
        "adm2": rows(directory / (stem + ".adm_common2.hst")),
        "adm5": rows(directory / (stem + ".adm_common5.hst")),
        "stem": stem,
    }


def extract(case, target):
    ref = matched(case["ref"], target)
    user = matched(case["user"], target)
    adm0 = matched(case["adm0"], target)
    adm1 = matched(case["adm1"], target)
    adm2 = matched(case["adm2"], target)
    adm5 = matched(case["adm5"], target)
    values = {
        "actual_time": ref[0],
        "gh_l2": safe_l2(ref[2], ref[11]),
        "reduction_l2": safe_l2(ref[3], ref[11]),
        "curl_l2": safe_l2(ref[4], ref[11]),
        "psi_error_l2": safe_l2(ref[5], ref[11]),
        "pi_l2": safe_l2(ref[6], ref[11]),
        "phi_l2": safe_l2(ref[7], ref[11]),
        "bad_state": ref[21],
        "physical_lapse_min": -ref[13],
        "regular_max": ref[14],
        "metric_condition_max": ref[15],
        "coordinate_condition_max": ref[16],
        "characteristic_speed_max": ref[17],
        "effective_cfl": ref[18],
        "detg_min": -ref[19],
        "delta_q": user[2],
        "delta_q_dot": user[3],
        "delta_p": user[4],
        "delta_p_dot": user[5],
        "e_G": user[6],
        "e_alpha": user[7],
        "fit_cells": user[8],
        "G_lambda_min": user[9],
        "G_lambda_max": user[10],
        "relative_lapse_min": user[14],
        "relative_lapse_max": user[15],
        "relative_v2_max": user[16],
        "relative_psi_max": user[17],
        "relative_inverse_psi_max": user[18],
        "transition": user[23],
        "feedback": user[24],
        "shell_valid": user[25],
        "controller_generation": user[26],
    }
    for prefix, row, base in (("all", adm0, 2), ("lt1", adm0, 9),
                              ("lt2", adm1, 2), ("r2to4", adm1, 9),
                              ("r4to8", adm2, 2), ("if2", adm5, 9)):
        volume = row[base + 6]
        values[prefix + "_H_l1"] = row[base] / volume if volume > 0 else float("nan")
        values[prefix + "_H_l2"] = safe_l2(row[base + 1], volume)
        values[prefix + "_H_linf"] = row[base + 2]
        values[prefix + "_M_l1"] = row[base + 3] / volume if volume > 0 else float("nan")
        values[prefix + "_M_l2"] = safe_l2(row[base + 4], volume)
        values[prefix + "_M_linf"] = row[base + 5]
    values["finite"] = all(finite(value) for value in values.values())
    return values


def main():
    parser = argparse.ArgumentParser()
    for label in ("coarse", "medium", "fine"):
        parser.add_argument("--" + label, required=True, type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    cases = {label: load_case(getattr(args, label))
             for label in ("coarse", "medium", "fine")}
    records = []
    for target in TARGETS:
        extracted = {label: extract(case, target) for label, case in cases.items()}
        record = {"target_time": target, "resolutions": extracted, "orders": {}}
        for metric in ("gh_l2", "reduction_l2", "curl_l2", "all_H_l2",
                       "all_M_l2", "lt1_H_l2", "lt1_M_l2", "lt2_H_l2",
                       "lt2_M_l2", "r2to4_H_l2", "r2to4_M_l2",
                       "if2_H_l2", "if2_M_l2"):
            record["orders"][metric] = {
                "coarse_medium": pair_order(extracted["coarse"][metric],
                                             extracted["medium"][metric],
                                             SPACINGS["coarse"], SPACINGS["medium"]),
                "medium_fine": pair_order(extracted["medium"][metric],
                                           extracted["fine"][metric],
                                           SPACINGS["medium"], SPACINGS["fine"]),
            }
        records.append(record)
    payload = {"schema": "ref-gh-controlled-transition-summary-v1",
               "mode": args.mode, "spacings": SPACINGS, "records": records}
    json_path = Path(str(args.output_prefix) + ".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    with tsv_path.open("w") as stream:
        stream.write("time\tresolution\tdx\tfinite\tbad_state\tgh_l2\t"
                     "reduction_l2\tall_H_l2\tall_M_l2\tlt1_H_l2\tlt1_M_l2\t"
                     "r2to4_H_l2\tif2_H_l2\tdelta_q\tdelta_p\te_G\te_alpha\t"
                     "feedback\tG_condition\trelative_lapse_min\tphysical_lapse_min\t"
                     "char_max\teffective_cfl\n")
        for record in records:
            for label in ("coarse", "medium", "fine"):
                value = record["resolutions"][label]
                columns = [record["target_time"], label, SPACINGS[label],
                           value["finite"], value["bad_state"], value["gh_l2"],
                           value["reduction_l2"], value["all_H_l2"],
                           value["all_M_l2"], value["lt1_H_l2"],
                           value["lt1_M_l2"], value["r2to4_H_l2"],
                           value["if2_H_l2"], value["delta_q"], value["delta_p"],
                           value["e_G"], value["e_alpha"], value["feedback"],
                           value["metric_condition_max"], value["relative_lapse_min"],
                           value["physical_lapse_min"], value["characteristic_speed_max"],
                           value["effective_cfl"]]
                stream.write("\t".join(str(item) for item in columns) + "\n")
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
