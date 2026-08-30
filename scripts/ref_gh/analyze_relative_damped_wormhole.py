#!/usr/bin/env python3
"""Analyze relative-damped Ref-GH wormhole-to-trumpet histories.

The controlled-transition user-history layout changed after the original
wave-map campaign.  This analyzer validates the current 34-column schemas,
allows exponent-fit NaNs only when the fitting shell is explicitly invalid,
and keeps endpoint completion, fast-mode recurrence, settling, causal reach,
and inter-resolution behavior as separate statements.
"""

import argparse
import csv
import json
import math
from pathlib import Path


REF_COLUMNS = {
    "time": 0, "dt": 1, "GH_L2sq": 2, "reduction_L2sq": 3,
    "curl_L2sq": 4, "Psi_error_L2sq": 5, "Pi_L2sq": 6,
    "Phi_L2sq": 7, "GH_near_L2sq": 8, "reduction_near_L2sq": 9,
    "curl_near_L2sq": 10, "volume": 11, "alpha_max": 12,
    "minus_alpha_min": 13, "regular_max": 14,
    "relative_metric_condition_max": 15, "coordinate_condition_max": 16,
    "characteristic_speed_max": 17, "effective_CFL": 18,
    "minus_detg_min": 19, "near_volume": 20, "bad_state": 21,
    "Q_Linf": 22, "Delta_Linf": 23, "reference_Ricci_Linf": 24,
    "coordinate_Ricci_Linf": 25, "source_curvature_Linf": 26,
    "source_QQ_Linf": 27, "source_DeltaDelta_Linf": 28,
    "source_damping_Linf": 29, "source_frame_Linf": 30,
    "relative_D_Linf": 31, "relative_WD_Linf": 32,
    "relative_source_Linf": 33,
}

USER_COLUMNS = {
    "time": 0, "dt": 1, "delta_q": 2, "delta_p": 3,
    "e_G": 4, "e_alpha": 5, "fit_cells": 6, "shell_valid": 7,
    "relative_metric_condition_max": 8, "relative_lapse_min": 9,
    "relative_lapse_max": 10, "relative_v2_max": 11,
    "relative_Psi_max": 12, "relative_inverse_Psi_max": 13,
    "minus_physical_lapse_min": 14, "physical_lapse_max": 15,
    "characteristic_speed_max": 16, "transition_amplitude": 17,
    "controller_generation": 18, "xi": 19, "xi_dot": 20,
    "xi_ddot": 21, "v_cmd": 22, "risk": 23,
    "risk_condition": 24, "risk_lapse_min": 25,
    "risk_lapse_max": 26, "risk_v2": 27, "GH_L2": 28,
    "reduction_L2": 29, "curl_L2": 30, "constraint_veto": 31,
    "controller_frozen": 32, "controller_completed": 33,
}

COMMON_LAYOUT = {
    "adm_common0": (("whole", 2), ("r_lt_1", 9)),
    "adm_common1": (("r_lt_2", 2), ("r_2_4", 9)),
    "adm_common2": (("r_4_8", 2), ("r_ge_8", 9)),
    "adm_common3": (("interface_64", 2), ("interface_32", 9)),
    "adm_common4": (("interface_16", 2), ("interface_8", 9)),
    "adm_common5": (("interface_4", 2), ("interface_2", 9)),
}

OLD_FAST_RATE = 26.654903904216415
OLD_FAST_FRACTION = 0.25
OLD_FAST_R2 = 0.95
DEFAULT_TARGETS = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def data_rows(path, expected_columns):
    """Read numeric history rows, deduplicated by time."""
    by_time = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) != expected_columns:
                raise ValueError(
                    "{}: expected {} columns, got {}".format(
                        path, expected_columns, len(values)))
            by_time[values[0]] = values
    if not by_time:
        raise ValueError("{}: no numeric rows".format(path))
    return [by_time[time] for time in sorted(by_time)]


def one_file(directory, suffix):
    matches = sorted(directory.glob("*" + suffix))
    if len(matches) != 1:
        raise ValueError(
            "{}: expected one *{}, got {}".format(directory, suffix, matches))
    return matches[0]


def merged_rows(directories, suffix, expected_columns):
    by_time = {}
    for directory in directories:
        for row in data_rows(one_file(directory, suffix), expected_columns):
            by_time[row[0]] = row
    return [by_time[time] for time in sorted(by_time)]


def nearest(rows, target, tolerance):
    row = min(rows, key=lambda value: abs(value[0] - target))
    return row if abs(row[0] - target) <= tolerance else None


def safe_rms(total, volume):
    if total < 0.0 or volume <= 0.0:
        return math.nan
    return math.sqrt(total/volume)


def common_region(row, base):
    volume = row[base + 6]
    return {
        "H_L1": row[base]/volume if volume > 0.0 else math.nan,
        "H_RMS": safe_rms(row[base + 1], volume),
        "H_Linf": row[base + 2],
        "M_L1": row[base + 3]/volume if volume > 0.0 else math.nan,
        "M_RMS": safe_rms(row[base + 4], volume),
        "M_Linf": row[base + 5],
        "volume": volume,
    }


def extract_record(ref_row, user_row, common_rows):
    volume = ref_row[REF_COLUMNS["volume"]]
    near_volume = ref_row[REF_COLUMNS["near_volume"]]
    result = {
        "time": ref_row[REF_COLUMNS["time"]],
        "dt": ref_row[REF_COLUMNS["dt"]],
        "GH_RMS": safe_rms(ref_row[REF_COLUMNS["GH_L2sq"]], volume),
        "reduction_RMS": safe_rms(
            ref_row[REF_COLUMNS["reduction_L2sq"]], volume),
        "curl_RMS": safe_rms(ref_row[REF_COLUMNS["curl_L2sq"]], volume),
        "Psi_error_RMS": safe_rms(
            ref_row[REF_COLUMNS["Psi_error_L2sq"]], volume),
        "Pi_RMS": safe_rms(ref_row[REF_COLUMNS["Pi_L2sq"]], volume),
        "Phi_RMS": safe_rms(ref_row[REF_COLUMNS["Phi_L2sq"]], volume),
        "GH_near_RMS": safe_rms(
            ref_row[REF_COLUMNS["GH_near_L2sq"]], near_volume),
        "reduction_near_RMS": safe_rms(
            ref_row[REF_COLUMNS["reduction_near_L2sq"]], near_volume),
        "curl_near_RMS": safe_rms(
            ref_row[REF_COLUMNS["curl_near_L2sq"]], near_volume),
        "physical_lapse_min": -user_row[
            USER_COLUMNS["minus_physical_lapse_min"]],
        "physical_lapse_max": user_row[USER_COLUMNS["physical_lapse_max"]],
        "relative_lapse_min": user_row[USER_COLUMNS["relative_lapse_min"]],
        "relative_lapse_max": user_row[USER_COLUMNS["relative_lapse_max"]],
        "relative_v2_max": user_row[USER_COLUMNS["relative_v2_max"]],
        "relative_metric_condition_max": user_row[
            USER_COLUMNS["relative_metric_condition_max"]],
        "relative_Psi_max": user_row[USER_COLUMNS["relative_Psi_max"]],
        "relative_inverse_Psi_max": user_row[
            USER_COLUMNS["relative_inverse_Psi_max"]],
        "e_G": user_row[USER_COLUMNS["e_G"]],
        "e_alpha": user_row[USER_COLUMNS["e_alpha"]],
        "fit_cells": user_row[USER_COLUMNS["fit_cells"]],
        "shell_valid": user_row[USER_COLUMNS["shell_valid"]] > 0.5,
        "transition_amplitude": user_row[
            USER_COLUMNS["transition_amplitude"]],
        "xi": user_row[USER_COLUMNS["xi"]],
        "xi_dot": user_row[USER_COLUMNS["xi_dot"]],
        "bad_state": ref_row[REF_COLUMNS["bad_state"]],
        "Q_Linf": ref_row[REF_COLUMNS["Q_Linf"]],
        "Delta_Linf": ref_row[REF_COLUMNS["Delta_Linf"]],
        "relative_D_Linf": ref_row[REF_COLUMNS["relative_D_Linf"]],
        "relative_WD_Linf": ref_row[REF_COLUMNS["relative_WD_Linf"]],
        "relative_source_Linf": ref_row[
            REF_COLUMNS["relative_source_Linf"]],
        "regular_max": ref_row[REF_COLUMNS["regular_max"]],
        "coordinate_condition_max": ref_row[
            REF_COLUMNS["coordinate_condition_max"]],
        "characteristic_speed_max": ref_row[
            REF_COLUMNS["characteristic_speed_max"]],
        "effective_CFL": ref_row[REF_COLUMNS["effective_CFL"]],
        "detg_min": -ref_row[REF_COLUMNS["minus_detg_min"]],
        "delta_q": user_row[USER_COLUMNS["delta_q"]],
        "delta_p": user_row[USER_COLUMNS["delta_p"]],
        "constraint_veto": user_row[USER_COLUMNS["constraint_veto"]],
    }
    result["common_constraints"] = {}
    for name, row in common_rows.items():
        for region, base in COMMON_LAYOUT[name]:
            result["common_constraints"][region] = common_region(row, base)
    return result


def log_fit(points, minimum_time):
    accepted = [(time, math.log(value)) for time, value in points
                if time >= minimum_time and math.isfinite(value) and value > 0.0]
    if len(accepted) < 3:
        return {"samples": len(accepted), "slope": None,
                "e_folding_time": None, "r_squared": None}
    mean_x = sum(item[0] for item in accepted)/len(accepted)
    mean_y = sum(item[1] for item in accepted)/len(accepted)
    denominator = sum((item[0] - mean_x)**2 for item in accepted)
    if denominator == 0.0:
        return {"samples": len(accepted), "slope": None,
                "e_folding_time": None, "r_squared": None}
    slope = sum((x - mean_x)*(y - mean_y)
                for x, y in accepted)/denominator
    intercept = mean_y - slope*mean_x
    residual = sum((y - intercept - slope*x)**2 for x, y in accepted)
    total = sum((y - mean_y)**2 for _, y in accepted)
    return {
        "samples": len(accepted), "minimum_time": minimum_time,
        "slope": slope,
        "e_folding_time": 1.0/slope if slope > 0.0 else None,
        "r_squared": 1.0 - residual/total if total > 0.0 else None,
    }


def allowed_finite_user(row):
    for index, value in enumerate(row):
        if index in (USER_COLUMNS["e_G"], USER_COLUMNS["e_alpha"]):
            if row[USER_COLUMNS["shell_valid"]] <= 0.5:
                continue
        if not math.isfinite(value):
            return False
    return True


def status(directory):
    result = {"directory": str(directory), "exit_status": None,
              "target": None}
    for name in ("run_status.txt", "segment_status.txt"):
        path = directory/name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in ("run_exit_status", "exit_status"):
                result["exit_status"] = int(value)
            elif key == "target":
                result["target"] = float(value)
    return result


def maxloc_summary(directories):
    records = []
    for directory in directories:
        matches = sorted(directory.glob("*.ref_gh_maxloc.tsv"))
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("{}: ambiguous maxloc files".format(directory))
        with matches[0].open(encoding="utf-8", newline="") as stream:
            records.extend(csv.DictReader(stream, delimiter="\t"))
    if not records:
        return {}
    final_time = max(float(row["time"]) for row in records)
    result = {}
    for row in records:
        if abs(float(row["time"]) - final_time) > 1.0e-12:
            continue
        result[row["diagnostic"]] = {
            "maximum": float(row["maximum"]),
            "radius": float(row["radius"]),
            "level": int(row["level"]),
            "rank": int(row["rank"]),
            "gid": int(row["gid"]),
            "coordinate": [
                float(row["x"]), float(row["y"]), float(row["z"])],
        }
    return {"time": final_time, "diagnostics": result}


def parse_case(specification):
    if "=" not in specification:
        raise ValueError("case must be LABEL=DIR[,DIR...]")
    label, paths = specification.split("=", 1)
    directories = [Path(path) for path in paths.split(",")]
    if not label or not directories or not all(path.is_dir() for path in directories):
        raise ValueError("invalid case {}".format(specification))
    return label, directories


def parse_spacing(specification):
    if "=" not in specification:
        raise ValueError("spacing must be LABEL=DX")
    label, value = specification.split("=", 1)
    spacing = float(value)
    if spacing <= 0.0 or not math.isfinite(spacing):
        raise ValueError("invalid spacing {}".format(specification))
    return label, spacing


def load_case(label, directories, targets, tolerance, outer_distance,
              late_fit_time):
    ref_rows = merged_rows(directories, ".ref_gh.hst", 34)
    user_rows = merged_rows(directories, ".user.hst", 34)
    common = {
        name: merged_rows(directories, "." + name + ".hst", 19 if name == "adm_common0" else 16)
        for name in COMMON_LAYOUT
    }
    records = []
    for target in targets:
        ref = nearest(ref_rows, target, tolerance)
        user = nearest(user_rows, target, tolerance)
        common_at_target = {
            name: nearest(rows, target, tolerance)
            for name, rows in common.items()
        }
        if ref is None or user is None or any(
                row is None for row in common_at_target.values()):
            continue
        item = extract_record(ref, user, common_at_target)
        item["target_time"] = target
        records.append(item)

    ref_series = []
    for ref in ref_rows:
        ref_series.append({
            "time": ref[0],
            "GH_RMS": safe_rms(ref[REF_COLUMNS["GH_L2sq"]],
                               ref[REF_COLUMNS["volume"]]),
            "reduction_RMS": safe_rms(
                ref[REF_COLUMNS["reduction_L2sq"]],
                ref[REF_COLUMNS["volume"]]),
            "curl_RMS": safe_rms(ref[REF_COLUMNS["curl_L2sq"]],
                                 ref[REF_COLUMNS["volume"]]),
            "Psi_error_RMS": safe_rms(
                ref[REF_COLUMNS["Psi_error_L2sq"]],
                ref[REF_COLUMNS["volume"]]),
            "relative_D_Linf": ref[REF_COLUMNS["relative_D_Linf"]],
            "relative_source_Linf": ref[REF_COLUMNS["relative_source_Linf"]],
        })
    minimum_fit_time = min(max(0.2, ref_rows[0][0]), ref_rows[-1][0])
    fast_fit = log_fit(
        ((row["time"], row["GH_RMS"]) for row in ref_series),
        minimum_fit_time)
    old_fast = (
        fast_fit["slope"] is not None
        and fast_fit["r_squared"] is not None
        and OLD_FAST_RATE*(1.0 - OLD_FAST_FRACTION) <= fast_fit["slope"]
        <= OLD_FAST_RATE*(1.0 + OLD_FAST_FRACTION)
        and fast_fit["r_squared"] >= OLD_FAST_R2)
    late_fits = {
        metric: log_fit(((row["time"], row[metric]) for row in ref_series),
                        late_fit_time)
        for metric in ("GH_RMS", "reduction_RMS", "curl_RMS",
                       "Psi_error_RMS", "relative_D_Linf",
                       "relative_source_Linf")
    }
    travel = 0.0
    for left, right in zip(user_rows, user_rows[1:]):
        travel += 0.5*(
            left[USER_COLUMNS["characteristic_speed_max"]]
            + right[USER_COLUMNS["characteristic_speed_max"]]
        )*(right[0] - left[0])
    all_finite = (
        all(math.isfinite(value) for row in ref_rows for value in row)
        and all(allowed_finite_user(row) for row in user_rows)
        and all(math.isfinite(value) for rows in common.values()
                for row in rows for value in row))
    exact_zero_controls = all(
        row[USER_COLUMNS[key]] == 0.0
        for row in user_rows for key in ("delta_q", "delta_p"))
    result = {
        "label": label,
        "directories": [str(path) for path in directories],
        "segments": [status(path) for path in directories],
        "initial_time": ref_rows[0][0],
        "final_time": ref_rows[-1][0],
        "history_rows": len(ref_rows),
        "all_required_values_finite": all_finite,
        "bad_state_clear": max(
            row[REF_COLUMNS["bad_state"]] for row in ref_rows) == 0.0,
        "exact_zero_exponent_controls": exact_zero_controls,
        "minimum_dt": min(row[REF_COLUMNS["dt"]] for row in ref_rows),
        "maximum_characteristic_speed": max(
            row[REF_COLUMNS["characteristic_speed_max"]] for row in ref_rows),
        "accumulated_characteristic_distance": travel,
        "outer_coordinate_distance": outer_distance,
        "remaining_causal_distance": outer_distance - travel,
        "old_fast_mode_fit": fast_fit,
        "old_fast_mode_recurrence": old_fast,
        "late_time_fits": late_fits,
        "records": records,
        "final_max_locations": maxloc_summary(directories),
    }
    if records:
        result["initial_record"] = records[0]
        result["final_record"] = records[-1]
    return result


def resolution_pairs(cases, spacings):
    labels = sorted(
        (label for label in cases if label in spacings),
        key=lambda label: spacings[label], reverse=True)
    metrics = ("GH_RMS", "reduction_RMS", "curl_RMS", "Psi_error_RMS")
    result = []
    by_label_target = {
        label: {record["target_time"]: record
                for record in payload["records"]}
        for label, payload in cases.items()
    }
    for coarse, fine in zip(labels, labels[1:]):
        common_targets = sorted(
            set(by_label_target[coarse]) & set(by_label_target[fine]))
        for target in common_targets:
            orders = {}
            for metric in metrics:
                left = by_label_target[coarse][target][metric]
                right = by_label_target[fine][target][metric]
                order = (math.log(left/right)
                         /math.log(spacings[coarse]/spacings[fine])
                         if left > 0.0 and right > 0.0 else math.nan)
                orders[metric] = {
                    "coarse": left, "fine": right, "signed_order": order,
                    "resolution_improving": left >= right,
                }
            result.append({
                "coarse": coarse, "fine": fine, "target_time": target,
                "orders": orders,
            })
    return result


def json_safe(value):
    """Replace diagnostic NaN/Inf sentinels with strict-JSON nulls."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True,
                        help="LABEL=DIR[,DIR...] chronological segments")
    parser.add_argument("--spacing", action="append", default=[],
                        help="LABEL=finest_DX for pairwise consistency")
    parser.add_argument("--target", nargs="+", type=float,
                        default=DEFAULT_TARGETS)
    parser.add_argument("--target-tolerance", type=float, default=0.03)
    parser.add_argument("--outer-distance", type=float, default=24.0)
    parser.add_argument("--late-fit-time", type=float, default=4.0)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    case_specs = [parse_case(value) for value in args.case]
    spacings = dict(parse_spacing(value) for value in args.spacing)
    cases = {
        label: load_case(
            label, directories, args.target, args.target_tolerance,
            args.outer_distance, args.late_fit_time)
        for label, directories in case_specs
    }
    payload = {
        "schema": "ref-gh-relative-damped-wormhole-history-v1",
        "claim_boundary": (
            "history-level finite/constraint/causal evidence only; physical "
            "trumpet settling requires the separate binary64 profile analysis"),
        "cases": cases,
        "spacings": spacings,
        "resolution_pairs": resolution_pairs(cases, spacings),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(args.output_prefix) + ".json")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    safe_payload = json_safe(payload)
    json_path.write_text(
        json.dumps(safe_payload, allow_nan=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    columns = (
        "target_time", "time", "GH_RMS", "reduction_RMS", "curl_RMS",
        "Psi_error_RMS", "physical_lapse_min", "physical_lapse_max",
        "relative_metric_condition_max", "relative_lapse_min",
        "relative_lapse_max", "relative_v2_max", "relative_D_Linf",
        "relative_source_Linf", "transition_amplitude", "e_G", "e_alpha",
        "bad_state", "characteristic_speed_max", "effective_CFL")
    with tsv_path.open("w", encoding="utf-8") as stream:
        stream.write("case\t" + "\t".join(columns) + "\n")
        for label in sorted(cases):
            for record in cases[label]["records"]:
                stream.write(label + "\t" + "\t".join(
                    str(record[key]) for key in columns) + "\n")
    print(json.dumps(
        safe_payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
