#!/usr/bin/env python3
"""Compare direct-fixed, moving, hard-freeze, and smooth-stop Ref-GH cases.

The fit windows are explicit command-line data.  In particular, no smooth-stop
growth fit is inferred across the stopping transient.
"""

import argparse
import json
import math
from pathlib import Path

from analyze_reference_motion_freeze import LOCALIZED, log_fit
from analyze_relative_damped_power_lag import load_case, parse_case


QUANTITIES = (
    "GH_RMS", "reduction_RMS", "curl_RMS", "Pi_RHS_Linf",
    "Phi_RHS_Linf", "source_frame_correction",
    "reference_dt_frame", "reference_dt_connection")


def value(record, name):
    if name in record:
        return record[name]
    return record["maxloc"].get(name, {}).get("maximum", math.nan)


def parse_window(specification):
    label, separator, interval = specification.partition("=")
    if not separator:
        raise ValueError("window must be LABEL=TMIN:TMAX")
    start, separator, end = interval.partition(":")
    if not separator:
        raise ValueError("window must be LABEL=TMIN:TMAX")
    return label, float(start), float(end)


def nearest(records, target):
    return min(records, key=lambda record: abs(record["time"] - target))


def summarize(label, records, windows):
    final = records[-1]
    fits = {}
    for window_label, start, end in windows:
        fits[window_label] = {
            quantity: log_fit(records, quantity, start, end)
            for quantity in QUANTITIES
        }
    locations = {
        name: final["maxloc"][name]
        for name in LOCALIZED if name in final["maxloc"]
    }
    samples = {}
    for target in (0.0, 2.0, 3.0, 3.8, 4.2, 5.2):
        record = nearest(records, target)
        if abs(record["time"] - target) <= 0.03:
            samples[str(target)] = record
    return {
        "label": label,
        "history_interval": [records[0]["time"], final["time"]],
        "initial": records[0],
        "final": final,
        "samples": samples,
        "growth_fits": fits,
        "final_localized_maxima": locations,
    }


def write_table(path, cases):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("case\twindow\tquantity\tt_min\tt_max\tsamples\t"
                     "slope_per_M\te_folding_M\tr_squared\n")
        for case in cases:
            for window, quantities in case["growth_fits"].items():
                for quantity, fit in quantities.items():
                    row = (case["label"], window, quantity,
                           fit["interval"][0], fit["interval"][1],
                           fit["samples"], fit["slope_per_M"],
                           fit["e_folding_M"], fit["r_squared"])
                    stream.write("\t".join(str(item) for item in row) + "\n")


def plot(path, cases, stop_start, stop_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    for label, records in cases:
        times = [record["time"] for record in records]
        for quantity, style in zip(
                ("GH_RMS", "reduction_RMS", "curl_RMS"), ("-", "--", ":")):
            axes[0].semilogy(
                times, [max(value(record, quantity), 1.0e-18)
                        for record in records], style,
                label=f"{label}:{quantity}")
        axes[1].plot(times, [record["xi"] for record in records], label=label)
        axes[2].semilogy(
            times, [max(value(record, "reference_dt_frame"), 1.0e-30)
                    for record in records], label=f"{label}:dt-frame")
        axes[2].semilogy(
            times, [max(value(record, "reference_dt_connection"), 1.0e-30)
                    for record in records], "--", label=f"{label}:dt-Gamma")
    for axis in axes:
        axis.axvspan(stop_start, stop_end, color="grey", alpha=0.15)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("constraint RMS")
    axes[1].set_ylabel("xi")
    axes[2].set_ylabel("reference motion max")
    axes[2].set_xlabel("t/M")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--window", action="append", required=True,
                        help="CASE:NAME=TMIN:TMAX")
    parser.add_argument("--stop-start", type=float, default=2.0)
    parser.add_argument("--stop-end", type=float, default=3.0)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    windows = {}
    for specification in args.window:
        case_and_name, start, end = parse_window(specification)
        case, separator, name = case_and_name.partition(":")
        if not separator:
            raise ValueError("window must be CASE:NAME=TMIN:TMAX")
        windows.setdefault(case, []).append((name, start, end))

    summaries = []
    plot_cases = []
    for specification in args.case:
        label, directories = parse_case(specification)
        records, _, _ = load_case(directories)
        summaries.append(summarize(label, records, windows.get(label, [])))
        plot_cases.append((label, records))

    payload = {
        "schema": "ref-gh-fixed-reference-discriminator-v1",
        "stop_interval": [args.stop_start, args.stop_end],
        "fit_policy": (
            "Every fit uses an explicitly named interval; post-stop fits must "
            "begin only after the declared stopping transient."),
        "claim_limit": (
            "This bounded discriminator is not a stable or convergent trumpet "
            "claim and is not a formal operator stability analysis."),
        "cases": {summary["label"]: summary for summary in summaries},
    }
    json_path = Path(str(args.output_prefix) + ".json")
    table_path = Path(str(args.output_prefix) + "_growth.tsv")
    plot_path = Path(str(args.output_prefix) + ".png")
    write_table(table_path, summaries)
    try:
        plot(plot_path, plot_cases, args.stop_start, args.stop_end)
        payload["plot"] = {"written": True, "path": str(plot_path)}
    except ModuleNotFoundError as error:
        payload["plot"] = {"written": False, "reason": str(error)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(json_path)
    print(table_path)
    if payload["plot"]["written"]:
        print(plot_path)


if __name__ == "__main__":
    main()
