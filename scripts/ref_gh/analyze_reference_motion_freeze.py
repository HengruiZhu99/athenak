#!/usr/bin/env python3
"""Compare continued and hard-frozen Ref-GH reference trajectories.

This script reports log-linear growth fits and localized coefficient/source
maxima.  It deliberately does not turn those correlations into a formal
stability or causation claim.
"""

import argparse
import json
import math
from pathlib import Path

from analyze_relative_damped_power_lag import load_case, parse_case


CONSTRAINTS = ("GH_RMS", "reduction_RMS", "curl_RMS")
LOCALIZED = (
    "GH_constraint", "reduction_constraint", "curl_constraint",
    "Pi_RHS_Linf", "Phi_RHS_Linf", "source_curvature",
    "source_QQ", "source_DeltaDelta", "source_damping",
    "source_frame_correction", "reference_dt_frame",
    "reference_dt_connection", "reference_spatial_frame_gradient",
    "reference_window_gradient", "reference_Riemann")


def log_fit(records, name, minimum_time, maximum_time):
    samples = []
    for record in records:
        if not minimum_time <= record["time"] <= maximum_time:
            continue
        if name in record:
            value = record[name]
        else:
            value = record["maxloc"].get(name, {}).get("maximum", math.nan)
        if math.isfinite(value) and value > 0.0:
            samples.append((record["time"], math.log(value)))
    if len(samples) < 3:
        return {"samples": len(samples), "slope_per_M": None,
                "e_folding_M": None, "r_squared": None,
                "interval": [minimum_time, maximum_time]}
    tbar = sum(item[0] for item in samples)/len(samples)
    ybar = sum(item[1] for item in samples)/len(samples)
    tt = sum((time-tbar)**2 for time, _ in samples)
    ty = sum((time-tbar)*(value-ybar) for time, value in samples)
    slope = ty/tt if tt > 0.0 else math.nan
    intercept = ybar-slope*tbar
    residual = sum((value-(intercept+slope*time))**2
                   for time, value in samples)
    total = sum((value-ybar)**2 for _, value in samples)
    r_squared = 1.0-residual/total if total > 0.0 else None
    return {
        "samples": len(samples), "slope_per_M": slope,
        "e_folding_M": 1.0/slope if slope > 0.0 else None,
        "r_squared": r_squared, "interval": [minimum_time, maximum_time],
    }


def nearest(records, time):
    return min(records, key=lambda item: abs(item["time"]-time))


def summarize(label, records, freeze_time, post_start, post_end):
    # The merged history includes the common seed sample at exactly freeze_time.
    # Use the first evolved branch sample here so the reported motion sectors
    # describe the selected continuation rather than the pre-fork trajectory.
    tolerance = 1.0e-12*max(1.0, abs(freeze_time))
    post_records = [item for item in records
                    if item["time"] > freeze_time+tolerance]
    if not post_records:
        raise ValueError(
            f"{label}: no evolved samples strictly after t={freeze_time}")
    at_freeze = nearest(records, freeze_time)
    first = post_records[0]
    final = post_records[-1]
    fits = {}
    for name in CONSTRAINTS + ("Pi_RHS_Linf", "source_frame_correction"):
        fits[name] = {
            "full_post_freeze": log_fit(records, name, post_start, post_end),
            "late_post_freeze": log_fit(
                records, name, max(post_start, freeze_time+0.8), post_end),
        }
    locations = {}
    for name in LOCALIZED:
        entry = final["maxloc"].get(name)
        if entry is not None:
            locations[name] = entry
    activation_dot = first.get("reference_activation_dot")
    activation_ddot = first.get("reference_activation_ddot")
    return {
        "label": label,
        "history_interval": [records[0]["time"], records[-1]["time"]],
        "post_freeze_interval": [first["time"], final["time"]],
        "common_state_at_freeze": at_freeze,
        "first_post_freeze": first,
        "final": final,
        "growth_fits": fits,
        "final_localized_maxima": locations,
        "reference_motion_at_first_post_freeze": {
            "activation_dot": activation_dot,
            "activation_ddot": activation_ddot,
            "dt_frame_max": first["maxloc"].get(
                "reference_dt_frame", {}).get("maximum"),
            "dt_connection_max": first["maxloc"].get(
                "reference_dt_connection", {}).get("maximum"),
        },
    }


def write_table(path, summaries):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("case\tquantity\twindow\tt_min\tt_max\tsamples\t"
                     "slope_per_M\te_folding_M\tr_squared\n")
        for summary in summaries:
            for quantity, windows in summary["growth_fits"].items():
                for window, fit in windows.items():
                    values = (summary["label"], quantity, window,
                              fit["interval"][0], fit["interval"][1],
                              fit["samples"], fit["slope_per_M"],
                              fit["e_folding_M"], fit["r_squared"])
                    stream.write("\t".join(str(value) for value in values)
                                 + "\n")


def plot(path, cases, freeze_time):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True)
    for label, records in cases:
        times = [item["time"] for item in records]
        for name, style in zip(CONSTRAINTS, ("-", "--", ":")):
            axes[0].semilogy(
                times, [max(item[name], 1.0e-18) for item in records],
                style, label=label+":"+name)
        axes[1].plot(times, [item["xi"] for item in records], label=label)
        axes[2].semilogy(
            times,
            [max(item["maxloc"].get("reference_dt_frame", {})
                 .get("maximum", 0.0), 1.0e-30) for item in records],
            label=label+":dt-frame")
        axes[2].semilogy(
            times,
            [max(item["maxloc"].get("reference_dt_connection", {})
                 .get("maximum", 0.0), 1.0e-30) for item in records],
            "--", label=label+":dt-Gamma")
    for axis in axes:
        axis.axvline(freeze_time, color="black", alpha=0.35)
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
    parser.add_argument("--freeze-time", type=float, required=True)
    parser.add_argument("--post-start", type=float)
    parser.add_argument("--post-end", type=float)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    post_start = (args.post_start if args.post_start is not None
                  else args.freeze_time+0.15)

    summaries = []
    plot_cases = []
    for specification in args.case:
        label, directories = parse_case(specification)
        records, _, _ = load_case(directories)
        post_end = args.post_end if args.post_end is not None else records[-1]["time"]
        summaries.append(summarize(
            label, records, args.freeze_time, post_start, post_end))
        plot_cases.append((label, records))
    payload = {
        "schema": "ref-gh-reference-motion-freeze-v1",
        "freeze_time": args.freeze_time,
        "claim_limit": (
            "Log-linear fits and localized coefficient/source maxima are "
            "numerical evidence, not a formal stability or causation proof."),
        "cases": {item["label"]: item for item in summaries},
    }
    json_path = Path(str(args.output_prefix)+".json")
    table_path = Path(str(args.output_prefix)+"_growth.tsv")
    plot_path = Path(str(args.output_prefix)+".png")
    write_table(table_path, summaries)
    try:
        plot(plot_path, plot_cases, args.freeze_time)
        payload["plot"] = {"written": True, "path": str(plot_path)}
    except ModuleNotFoundError as error:
        payload["plot"] = {
            "written": False,
            "reason": f"optional plotting dependency unavailable: {error}",
        }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n",
                         encoding="utf-8")
    print(json_path)
    print(table_path)
    if payload["plot"]["written"]:
        print(plot_path)
    else:
        print(payload["plot"]["reason"])


if __name__ == "__main__":
    main()
