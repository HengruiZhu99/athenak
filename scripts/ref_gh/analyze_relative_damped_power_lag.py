#!/usr/bin/env python3
"""Correlate Ref-GH continuation, paired power fits, and constraint growth.

The paired q history is optional so that pre-instrumentation trajectories can
be audited honestly.  In that case ``-e_G/2`` is published only as the
calibrated pure-power proxy for Delta q, never as a direct measurement.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from analyze_relative_damped_wormhole import (
    REF_COLUMNS, USER_COLUMNS, data_rows, merged_rows, one_file, safe_rms)


SHELLS = ("inner", "blend", "outside", "legacy")
MAXLOC_SECTORS = (
    "GH_constraint", "Pi_RHS_Linf", "Psi_RHS_Linf", "Phi_RHS_Linf",
    "Q", "Delta", "reduction_constraint", "curl_constraint",
    "source_curvature", "source_QQ", "source_DeltaDelta", "source_damping",
    "source_frame_correction", "reference_dt_frame",
    "reference_dt_connection", "reference_spatial_frame_gradient",
    "reference_window_gradient")
DEFAULT_TARGETS = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2)


def parse_case(specification):
    label, separator, value = specification.partition("=")
    if not separator or not label:
        raise ValueError("case must be LABEL=DIR[,DIR]")
    directories = [Path(item) for item in value.split(",")]
    if not all(item.is_dir() for item in directories):
        raise ValueError("missing case directory in {}".format(specification))
    return label, directories


def nearest(rows, time, tolerance=0.03):
    if not rows:
        return None
    row = min(rows, key=lambda item: abs(item[0] - time))
    return row if abs(row[0] - time) <= tolerance else None


def read_named_table(path):
    header = None
    by_time = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.rstrip("\n").split("\t")
            if header is None:
                header = values
                continue
            if len(values) != len(header):
                raise ValueError("{}: malformed row".format(path))
            record = {name: float(value) for name, value in zip(header, values)}
            by_time[record["time"]] = record
    if header is None:
        raise ValueError("{}: no header".format(path))
    return [by_time[time] for time in sorted(by_time)]


def merged_named(directories, suffix):
    by_time = {}
    found = []
    for directory in directories:
        matches = sorted(directory.glob("*" + suffix))
        if len(matches) > 1:
            raise ValueError("{}: multiple *{}".format(directory, suffix))
        if matches:
            found.append(str(matches[0]))
            for row in read_named_table(matches[0]):
                by_time[row["time"]] = row
    return [by_time[time] for time in sorted(by_time)], found


def merged_maxloc(directories):
    records = {}
    sources = []
    for directory in directories:
        matches = sorted(directory.glob("*.ref_gh_maxloc.tsv"))
        if len(matches) > 1:
            raise ValueError("{}: multiple maxloc tables".format(directory))
        if not matches:
            continue
        sources.append(str(matches[0]))
        with matches[0].open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                diagnostic = row["diagnostic"]
                if diagnostic not in MAXLOC_SECTORS:
                    continue
                time = float(row["time"])
                records[(time, diagnostic)] = {
                    "maximum": float(row["maximum"]),
                    "radius": float(row["radius"]),
                }
    return records, sources


def maxloc_nearest(records, time, diagnostic, tolerance=0.03):
    candidates = [(abs(key[0] - time), value)
                  for key, value in records.items() if key[1] == diagnostic]
    if not candidates:
        return None
    distance, value = min(candidates, key=lambda item: item[0])
    return value if distance <= tolerance else None


def record_at(ref, user, power, maxloc, time):
    ref_row = nearest(ref, time)
    user_row = nearest(user, time)
    if ref_row is None or user_row is None:
        return None
    volume = ref_row[REF_COLUMNS["volume"]]
    result = {
        "time": ref_row[0],
        "xi": user_row[USER_COLUMNS["xi"]],
        "xi_dot": user_row[USER_COLUMNS["xi_dot"]],
        "xi_ddot": user_row[USER_COLUMNS["xi_ddot"]],
        "transition_amplitude": user_row[USER_COLUMNS["transition_amplitude"]],
        "delta_q_controller": user_row[USER_COLUMNS["delta_q"]],
        "delta_p_controller": user_row[USER_COLUMNS["delta_p"]],
        "e_G": user_row[USER_COLUMNS["e_G"]],
        "e_alpha": user_row[USER_COLUMNS["e_alpha"]],
        "e_G_pure_power_Delta_q_proxy": -0.5*user_row[USER_COLUMNS["e_G"]],
        "GH_RMS": safe_rms(ref_row[REF_COLUMNS["GH_L2sq"]], volume),
        "reduction_RMS": safe_rms(
            ref_row[REF_COLUMNS["reduction_L2sq"]], volume),
        "curl_RMS": safe_rms(ref_row[REF_COLUMNS["curl_L2sq"]], volume),
        "relative_metric_condition_max": user_row[
            USER_COLUMNS["relative_metric_condition_max"]],
        "relative_lapse_min": user_row[USER_COLUMNS["relative_lapse_min"]],
        "relative_lapse_max": user_row[USER_COLUMNS["relative_lapse_max"]],
        "relative_v2_max": user_row[USER_COLUMNS["relative_v2_max"]],
        "controller_frozen": user_row[USER_COLUMNS["controller_frozen"]] > 0.5,
        "controller_completed": user_row[
            USER_COLUMNS["controller_completed"]] > 0.5,
        "bad_state": ref_row[REF_COLUMNS["bad_state"]],
        "power": None,
        "maxloc": {},
    }
    power_row = None
    if power:
        power_row = min(power, key=lambda item: abs(item["time"] - time))
        if abs(power_row["time"] - time) > 0.03:
            power_row = None
    if power_row is not None:
        result["reference_activation_dot"] = power_row.get(
            "transition_dot")
        result["reference_activation_ddot"] = power_row.get(
            "transition_ddot")
        result["power"] = {}
        for shell in SHELLS:
            fit = {}
            for quantity in ("qphys", "qref", "dq"):
                for statistic in ("mean", "var", "min", "max", "rms"):
                    name = "{}_{}_{}".format(shell, quantity, statistic)
                    fit["{}_{}".format(quantity, statistic)] = power_row[name]
            fit["cells"] = int(round(power_row[shell + "_cells"]))
            fit["effective_samples"] = power_row[shell + "_neff"]
            fit["valid"] = power_row[shell + "_valid"] > 0.5
            fit["pure_power_relation_residual"] = (
                result["e_G"] + 2.0*fit["dq_mean"])
            result["power"][shell] = fit
    for diagnostic in MAXLOC_SECTORS:
        value = maxloc_nearest(maxloc, time, diagnostic)
        if value is not None:
            result["maxloc"][diagnostic] = value
    return result


def pearson(records, left, right):
    pairs = [(left(item), right(item)) for item in records]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xbar = sum(item[0] for item in pairs)/len(pairs)
    ybar = sum(item[1] for item in pairs)/len(pairs)
    xx = sum((x - xbar)**2 for x, _ in pairs)
    yy = sum((y - ybar)**2 for _, y in pairs)
    if xx == 0.0 or yy == 0.0:
        return None
    xy = sum((x - xbar)*(y - ybar) for x, y in pairs)
    return {"samples": len(pairs), "pearson_r": xy/math.sqrt(xx*yy)}


def first_growth(records, getter, absolute_floor, factor=10.0,
                 baseline_end=0.5):
    baseline = [abs(getter(item)) for item in records
                if item["time"] <= baseline_end
                and math.isfinite(getter(item))]
    if not baseline:
        return {"time": None, "threshold": None, "definition": "unavailable"}
    ordered = sorted(baseline)
    median = ordered[len(ordered)//2]
    threshold = max(absolute_floor, factor*median)
    found = next((item["time"] for item in records
                  if item["time"] > baseline_end
                  and math.isfinite(getter(item))
                  and abs(getter(item)) > threshold), None)
    return {"time": found, "threshold": threshold,
            "definition": "first |value| > max(floor, 10*baseline_median), baseline t<=0.5M"}


def load_case(directories):
    ref = merged_rows(directories, ".ref_gh.hst", len(REF_COLUMNS))
    user = merged_rows(directories, ".user.hst", len(USER_COLUMNS))
    power, power_sources = merged_named(directories, ".ref_gh_power.hst")
    maxloc, maxloc_sources = merged_maxloc(directories)
    times = sorted(set(row[0] for row in ref))
    records = [record_at(ref, user, power, maxloc, time) for time in times]
    records = [item for item in records if item is not None]
    return records, power_sources, maxloc_sources


def summarize(label, directories, records, power_sources, maxloc_sources,
              targets, tolerance):
    samples = []
    for target in targets:
        item = min(records, key=lambda row: abs(row["time"] - target))
        if abs(item["time"] - target) <= tolerance:
            samples.append({"target": target, "record": item})
    direct = bool(power_sources)
    dq_getter = ((lambda item: item["power"]["legacy"]["dq_mean"])
                 if direct else
                 (lambda item: item["e_G_pure_power_Delta_q_proxy"]))
    correlations = {}
    quantities = {
        "xi": lambda item: item["xi"],
        "xi_dot": lambda item: item["xi_dot"],
        "e_G": lambda item: item["e_G"],
        "e_alpha": lambda item: item["e_alpha"],
        "Delta_q_direct_or_proxy": dq_getter,
        "GH_RMS": lambda item: item["GH_RMS"],
        "reduction_RMS": lambda item: item["reduction_RMS"],
        "curl_RMS": lambda item: item["curl_RMS"],
        "relative_condition": lambda item: item["relative_metric_condition_max"],
        "relative_lapse_max": lambda item: item["relative_lapse_max"],
        "relative_v2_max": lambda item: item["relative_v2_max"],
    }
    if direct:
        for shell in SHELLS:
            quantities["Delta_q_" + shell] = (
                lambda item, name=shell: item["power"][name]["dq_mean"])
            quantities["q_phys_" + shell] = (
                lambda item, name=shell: item["power"][name]["qphys_mean"])
            quantities["q_ref_" + shell] = (
                lambda item, name=shell: item["power"][name]["qref_mean"])
    for diagnostic in MAXLOC_SECTORS:
        quantities["max_" + diagnostic] = (
            lambda item, name=diagnostic:
                item["maxloc"].get(name, {}).get("maximum", math.nan))
    left_names = ["Delta_q_direct_or_proxy", "e_G", "e_alpha"]
    if direct:
        left_names.extend("Delta_q_" + shell for shell in SHELLS)
    for left_name in left_names:
        right_names = ("xi", "xi_dot", "GH_RMS", "reduction_RMS",
                       "curl_RMS", "relative_condition",
                       "relative_lapse_max", "relative_v2_max") + tuple(
                           "max_" + name for name in MAXLOC_SECTORS)
        for right_name in right_names:
            value = pearson(records, quantities[left_name], quantities[right_name])
            correlations[left_name + "__" + right_name] = value
    onsets = {
        "Delta_q_direct_or_proxy": first_growth(records, dq_getter, 1.0e-3),
        "GH_RMS": first_growth(records, quantities["GH_RMS"], 1.0e-3),
        "reduction_RMS": first_growth(records, quantities["reduction_RMS"], 1.0e-4),
        "curl_RMS": first_growth(records, quantities["curl_RMS"], 1.0e-3),
    }
    if direct:
        for shell in SHELLS:
            onsets["Delta_q_" + shell] = first_growth(
                records, quantities["Delta_q_" + shell], 1.0e-3)
    return {
        "label": label,
        "directories": [str(item.resolve()) for item in directories],
        "history_interval": [records[0]["time"], records[-1]["time"]],
        "direct_paired_power_available": direct,
        "power_sources": power_sources,
        "maxloc_sources": maxloc_sources,
        "proxy_warning": (None if direct else
            "Direct q_phys/q_ref was not recorded. -e_G/2 is only the calibrated pure-power proxy and is not assumed exact in the blended state."),
        "onsets": onsets,
        "correlations": correlations,
        "samples": samples,
        "final": records[-1],
    }


def write_tsv(path, summaries):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("case\ttarget\ttime\txi\txi_dot\txi_ddot\te_G\te_alpha\t"
                     "Delta_q_kind\tDelta_q\tGH_RMS\treduction_RMS\tcurl_RMS\t"
                     "condition\trel_lapse_min\trel_lapse_max\tv2_max\n")
        for summary in summaries:
            direct = summary["direct_paired_power_available"]
            for sample in summary["samples"]:
                item = sample["record"]
                dq = (item["power"]["legacy"]["dq_mean"] if direct
                      else item["e_G_pure_power_Delta_q_proxy"])
                values = (summary["label"], sample["target"], item["time"],
                          item["xi"], item["xi_dot"], item["xi_ddot"],
                          item["e_G"], item["e_alpha"],
                          "direct_legacy_shell" if direct else "minus_eG_over_2_proxy",
                          dq, item["GH_RMS"], item["reduction_RMS"],
                          item["curl_RMS"], item["relative_metric_condition_max"],
                          item["relative_lapse_min"], item["relative_lapse_max"],
                          item["relative_v2_max"])
                stream.write("\t".join(str(value) for value in values) + "\n")


def write_shell_tsv(path, summaries):
    statistics = ("mean", "var", "min", "max", "rms")
    with path.open("w", encoding="utf-8") as stream:
        columns = ["case", "target", "time", "shell"]
        columns.extend("{}_{}".format(quantity, statistic)
                       for quantity in ("qphys", "qref", "dq")
                       for statistic in statistics)
        columns.extend(("cells", "effective_samples", "valid", "e_G",
                        "e_G_plus_2Delta_q"))
        stream.write("\t".join(columns) + "\n")
        for summary in summaries:
            if not summary["direct_paired_power_available"]:
                continue
            for sample in summary["samples"]:
                item = sample["record"]
                for shell in SHELLS:
                    fit = item["power"][shell]
                    values = [summary["label"], sample["target"],
                              item["time"], shell]
                    values.extend(fit["{}_{}".format(quantity, statistic)]
                                  for quantity in ("qphys", "qref", "dq")
                                  for statistic in statistics)
                    values.extend((fit["cells"], fit["effective_samples"],
                                   fit["valid"], item["e_G"],
                                   fit["pure_power_relation_residual"]))
                    stream.write("\t".join(str(value) for value in values)
                                 + "\n")


def plot(path, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(8.0, 10.5), sharex=True)
    for label, records, direct in cases:
        times = [item["time"] for item in records]
        axes[0].plot(times, [item["xi"] for item in records], label=label)
        if direct:
            for shell, style in zip(SHELLS, ("-", "--", "-.", ":")):
                axes[1].plot(times,
                    [item["power"][shell]["dq_mean"] for item in records],
                    linestyle=style, label=label + ":" + shell)
        else:
            axes[1].plot(times,
                [item["e_G_pure_power_Delta_q_proxy"] for item in records],
                label=label + ":-eG/2 proxy")
        axes[2].plot(times, [abs(item["e_G"]) for item in records],
                     label=label + ":|eG|")
        axes[2].plot(times, [abs(item["e_alpha"]) for item in records],
                     linestyle="--", label=label + ":|ealpha|")
        for name, style in (("GH_RMS", "-"), ("reduction_RMS", "--"),
                            ("curl_RMS", ":")):
            axes[3].semilogy(times, [max(abs(item[name]), 1.0e-18)
                                     for item in records],
                            linestyle=style, label=label + ":" + name)
    axes[0].set_ylabel("xi")
    axes[1].set_ylabel("Delta q")
    axes[2].set_ylabel("fit mismatch")
    axes[3].set_ylabel("constraint RMS")
    axes[3].set_xlabel("t/M")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--target", nargs="+", type=float, default=DEFAULT_TARGETS)
    parser.add_argument("--target-tolerance", type=float, default=0.035)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    summaries = []
    plot_cases = []
    for specification in args.case:
        label, directories = parse_case(specification)
        records, power_sources, maxloc_sources = load_case(directories)
        summaries.append(summarize(label, directories, records, power_sources,
                                   maxloc_sources, args.target,
                                   args.target_tolerance))
        plot_cases.append((label, records, bool(power_sources)))
    payload = {
        "schema": "ref-gh-relative-damped-power-lag-v1",
        "claim_limit": "Correlations and threshold orderings are diagnostic evidence, not proof of causation.",
        "onset_definition_is_heuristic": True,
        "cases": {item["label"]: item for item in summaries},
    }
    json_path = Path(str(args.output_prefix) + ".json")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    shell_tsv_path = Path(str(args.output_prefix) + "_shells.tsv")
    plot_path = Path(str(args.output_prefix) + ".png")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    write_tsv(tsv_path, summaries)
    write_shell_tsv(shell_tsv_path, summaries)
    plot(plot_path, plot_cases)
    print(json_path)
    print(tsv_path)
    print(shell_tsv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
