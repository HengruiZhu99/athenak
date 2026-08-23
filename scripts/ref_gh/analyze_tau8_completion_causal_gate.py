#!/usr/bin/env python3
"""Analyze the causally enlarged fixed-core tau-8 Ref-GH gate."""

import argparse
import csv
import json
import math
from pathlib import Path


SPACINGS = {"coarse": 1.0 / 16.0, "medium": 1.0 / 24.0,
            "fine": 1.0 / 32.0}
TARGETS = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
REGIONS = {
    "all": (0, 2), "lt1": (0, 9), "lt2": (1, 2),
    "r2to4": (1, 9), "r4to8": (2, 2), "gt8": (2, 9),
    "if4": (5, 2), "if2": (5, 9),
    "r0to1": (6, 2), "r1to2": (6, 9),
    "r2to3": (7, 2), "r3to4": (7, 9),
    "r4to6": (8, 2), "r6to8": (8, 9),
}
REGION_OUTER_RADIUS = {
    "lt1": 1.0, "r0to1": 1.0, "r1to2": 2.0, "lt2": 2.0,
    "r2to3": 3.0, "r3to4": 4.0, "r2to4": 4.0,
    "if2": 2.25, "if4": 4.5, "r4to6": 6.0,
    "r4to8": 8.0, "r6to8": 8.0,
}
ORDER_METRICS = ("GH_L2", "reduction_L2", "curl_L2")


def directories(value):
    result = [Path(item) for item in value.split(",")]
    if not all(path.is_dir() for path in result):
        raise ValueError(f"missing segment directory in {value}")
    return result


def one_file(directory, suffix):
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *{suffix} in {directory}, got {matches}")
    return matches[0]


def data_rows(path):
    result = []
    with path.open() as stream:
        for line in stream:
            if line.strip() and not line.startswith("#"):
                result.append([float(token) for token in line.split()])
    if not result:
        raise RuntimeError(f"no data rows in {path}")
    return result


def merged_rows(paths, suffix):
    by_time = {}
    for directory in paths:
        for row in data_rows(one_file(directory, suffix)):
            by_time[row[0]] = row
    return [by_time[time] for time in sorted(by_time)]


def nearest(rows, target, tolerance=0.026):
    row = min(rows, key=lambda item: abs(item[0] - target))
    return row if abs(row[0] - target) <= tolerance else None


def l2(total, volume):
    return math.sqrt(max(total, 0.0) / volume) if volume > 0.0 else math.nan


def region_values(row, base):
    volume = row[base + 6]
    return {
        "H_L1": row[base] / volume if volume > 0.0 else math.nan,
        "H_L2": l2(row[base + 1], volume), "H_Linf": row[base + 2],
        "M_L1": row[base + 3] / volume if volume > 0.0 else math.nan,
        "M_L2": l2(row[base + 4], volume), "M_Linf": row[base + 5],
        "volume": volume,
    }


def causal_history(adm0, boundary_radius, safety_buffer):
    result = []
    distance = 0.0
    previous_time = 0.0
    previous_speed = adm0[0][17]
    for row in adm0:
        time = row[0]
        speed = row[17]
        if time > previous_time:
            distance += (time - previous_time) * max(previous_speed, speed)
        result.append({
            "time": time, "characteristic_speed_max": speed,
            "travel_distance_upper_sum": distance,
            "causally_clean_radius": boundary_radius - distance - safety_buffer,
        })
        previous_time = time
        previous_speed = speed
    return result


def merged_maxloc(paths):
    by_key = {}
    for directory in paths:
        path = one_file(directory, ".adm_common_maxloc.tsv")
        with path.open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                key = (float(row["time"]), row["diagnostic"])
                by_key[key] = {
                    "time": key[0], "cycle": int(row["cycle"]),
                    "diagnostic": key[1], "maximum": float(row["maximum"]),
                    "radius": float(row["radius"]), "level": int(row["level"]),
                    "rank": int(row["rank"]), "gid": int(row["gid"]),
                    "x": float(row["x"]), "y": float(row["y"]),
                    "z": float(row["z"]),
                }
    return [by_key[key] for key in sorted(by_key)]


def load_case(paths, boundary_radius, safety_buffer):
    histories = {
        "ref": merged_rows(paths, ".ref_gh.hst"),
        "user": merged_rows(paths, ".user.hst"),
    }
    for instance in range(9):
        histories[f"adm{instance}"] = merged_rows(
            paths, f".adm_common{instance}.hst")
    histories["causal"] = causal_history(
        histories["adm0"], boundary_radius, safety_buffer)
    histories["maxloc"] = merged_maxloc(paths)
    return histories


def record_at(case, target):
    ref = nearest(case["ref"], target)
    user = nearest(case["user"], target)
    causal = min(case["causal"], key=lambda item: abs(item["time"] - target))
    if ref is None or user is None or abs(causal["time"] - target) > 0.026:
        return None
    volume = ref[11]
    result = {
        "target": target, "time": ref[0], "dt": ref[1],
        "GH_L2": l2(ref[2], volume), "reduction_L2": l2(ref[3], volume),
        "curl_L2": l2(ref[4], volume), "Psi_reference_L2": l2(ref[5], volume),
        "Pi_L2": l2(ref[6], volume), "Phi_L2": l2(ref[7], volume),
        "bad_state": ref[21], "physical_lapse_min": -ref[13],
        "metric_condition_max": ref[15], "coordinate_metric_max": ref[16],
        "characteristic_speed_max": ref[17], "effective_CFL": ref[18],
        "determinant_margin": -ref[19], "Q_Linf": ref[22],
        "Delta_Linf": ref[23], "delta_q": user[2], "delta_q_dot": user[3],
        "delta_p": user[4], "delta_p_dot": user[5],
        "relative_metric_lambda_min": user[9],
        "relative_metric_lambda_max": user[10],
        "relative_metric_condition_max": user[13],
        "relative_lapse_min": user[14], "relative_lapse_max": user[15],
        "relative_boost_v2_max": user[16], "transition_amplitude": user[23],
        "feedback": user[24], "controller_generation": user[26],
        "transition_rate": user[27], "transition_acceleration": user[28],
        "causal": causal, "regions": {}, "max_locations": {},
    }
    for name, (instance, base) in REGIONS.items():
        row = nearest(case[f"adm{instance}"], target)
        if row is None:
            continue
        values = region_values(row, base)
        if name in REGION_OUTER_RADIUS:
            values["causally_clean"] = (
                REGION_OUTER_RADIUS[name] < causal["causally_clean_radius"])
        result["regions"][name] = values
    maxloc_times = sorted({row["time"] for row in case["maxloc"]})
    if maxloc_times:
        selected = min(maxloc_times, key=lambda value: abs(value - target))
        if abs(selected - target) <= 0.026:
            result["max_locations"] = {
                row["diagnostic"]: row for row in case["maxloc"]
                if abs(row["time"] - selected) <= 1.0e-12
            }
    return result


def order(coarse, fine, h_coarse, h_fine):
    if coarse > 0.0 and fine > 0.0:
        return math.log(coarse / fine) / math.log(h_coarse / h_fine)
    return math.nan


def resolution_reversal_onsets(cases, start_time):
    if set(cases) != {"coarse", "medium", "fine"}:
        return {}
    times = [row[0] for row in cases["medium"]["ref"] if row[0] >= start_time]
    metrics = list(ORDER_METRICS)
    for region in REGIONS:
        metrics.extend((f"{region}.H_L2", f"{region}.M_L2"))
    onsets = {metric: None for metric in metrics}
    for target in times:
        records = {name: record_at(case, target) for name, case in cases.items()}
        if any(record is None for record in records.values()):
            continue
        for metric in metrics:
            if onsets[metric] is not None:
                continue
            if "." in metric:
                region, quantity = metric.split(".")
                values = [records[label]["regions"][region][quantity]
                          for label in ("coarse", "medium", "fine")]
            else:
                values = [records[label][metric]
                          for label in ("coarse", "medium", "fine")]
            if all(math.isfinite(value) for value in values) and not (
                    values[0] >= values[1] >= values[2]):
                onsets[metric] = {"time": target, "coarse": values[0],
                                  "medium": values[1], "fine": values[2]}
    return onsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse")
    parser.add_argument("--medium", required=True)
    parser.add_argument("--fine")
    parser.add_argument("--boundary-radius", type=float, default=12.0)
    parser.add_argument("--safety-buffer", type=float, default=0.5)
    parser.add_argument("--onset-start", type=float, default=0.2)
    parser.add_argument("--targets", nargs="+", type=float, default=TARGETS)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    cases = {}
    for label in ("coarse", "medium", "fine"):
        value = getattr(args, label)
        if value:
            cases[label] = load_case(
                directories(value), args.boundary_radius, args.safety_buffer)

    records = []
    for target in args.targets:
        values = {label: record_at(case, target) for label, case in cases.items()}
        values = {label: value for label, value in values.items() if value is not None}
        if not values:
            continue
        item = {"target": target, "resolutions": values, "orders": {}}
        if set(values) == {"coarse", "medium", "fine"}:
            metrics = list(ORDER_METRICS)
            for region in REGIONS:
                metrics.extend((f"{region}.H_L2", f"{region}.M_L2"))
            for metric in metrics:
                if "." in metric:
                    region, quantity = metric.split(".")
                    data = {label: values[label]["regions"][region][quantity]
                            for label in values}
                else:
                    data = {label: values[label][metric] for label in values}
                item["orders"][metric] = {
                    "coarse_medium": order(data["coarse"], data["medium"],
                                             SPACINGS["coarse"], SPACINGS["medium"]),
                    "medium_fine": order(data["medium"], data["fine"],
                                           SPACINGS["medium"], SPACINGS["fine"]),
                    "resolution_reversed": not (
                        data["coarse"] >= data["medium"] >= data["fine"]),
                }
        records.append(item)

    payload = {
        "schema": "ref-gh-tau8-completion-causal-gate-v1",
        "candidate": "fixed_core_tau8_compatible_open_loop",
        "boundary_radius": args.boundary_radius,
        "safety_buffer": args.safety_buffer,
        "spacings": {label: SPACINGS[label] for label in cases},
        "records": records,
        "resolution_reversal_onsets": resolution_reversal_onsets(
            cases, args.onset_start),
        "max_location_trajectories": {
            label: case["maxloc"] for label, case in cases.items()
        },
    }
    json_path = Path(str(args.output_prefix) + ".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    with tsv_path.open("w") as stream:
        stream.write("target\tresolution\tmetric\tvalue\tcausal_distance\t"
                     "causally_clean_radius\n")
        for item in records:
            for label, value in item["resolutions"].items():
                causal = value["causal"]
                for metric in ORDER_METRICS:
                    stream.write("\t".join(str(column) for column in (
                        item["target"], label, metric, value[metric],
                        causal["travel_distance_upper_sum"],
                        causal["causally_clean_radius"])) + "\n")
                for region, quantities in value["regions"].items():
                    for metric in ("H_L2", "M_L2"):
                        stream.write("\t".join(str(column) for column in (
                            item["target"], label, f"{region}.{metric}",
                            quantities[metric], causal["travel_distance_upper_sum"],
                            causal["causally_clean_radius"])) + "\n")
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
