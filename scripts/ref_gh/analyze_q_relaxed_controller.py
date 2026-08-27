#!/usr/bin/env python3
"""Summarize finite-h Ref-GH q-controller trumpet evidence.

The primary convergence fields come from the puncture-stencil-masked fixed
coordinate regions in ``*-trumpet.dat``.  No singularity-adjacent global Linf
value is promoted to a convergence claim.
"""

import argparse
import json
import math
from pathlib import Path


STATIC_METRICS = (
    "field_region0", "field_region1", "field_region2",
    "constraint_region0", "constraint_region1", "constraint_region2",
    "physical_metric_region0", "physical_metric_region1",
    "physical_metric_region2", "physical_lapse_region0",
    "physical_lapse_region1", "physical_lapse_region2",
    "physical_shift_region0", "physical_shift_region1",
    "physical_shift_region2",
)


def read_trumpet(path):
    """Read the final named row from a stationary-trumpet summary."""
    names = None
    values = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("# nx1 "):
            names = line[2:].split()
        elif line and not line.startswith("#"):
            values = [float(token) for token in line.split()]
    if names is None or values is None or len(names) != len(values):
        raise ValueError("malformed trumpet summary: {}".format(path))
    return dict(zip(names, values))


def read_history(path):
    """Return numerical history rows without relying on truncated labels."""
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rows.append([float(token) for token in line.split()])
    if not rows:
        raise ValueError("empty history: {}".format(path))
    return rows


def observed_order(coarse_error, fine_error, coarse_h, fine_h):
    if not (coarse_error > 0.0 and fine_error > 0.0):
        return None
    return math.log(coarse_error/fine_error)/math.log(coarse_h/fine_h)


def finite_or_none(value):
    return value if math.isfinite(value) else None


def discover_static(root):
    cases = []
    for directory in sorted(root.glob("q???_h??")):
        summaries = list(directory.glob("*-trumpet.dat"))
        users = list(directory.glob("*.user.hst"))
        native = list(directory.glob("*.ref_gh.hst"))
        if len(summaries) != 1 or len(users) != 1 or len(native) != 1:
            raise ValueError("incomplete static case: {}".format(directory))
        summary = read_trumpet(summaries[0])
        user_rows = read_history(users[0])
        native_rows = read_history(native[0])
        initial_user = user_rows[0]
        final_user = user_rows[-1]
        final_native = native_rows[-1]
        nx = int(round(summary["nx1"]))
        tag = directory.name.split("_", 1)[0]
        q_initial = {"q090": 0.9, "q100": 1.0, "q110": 1.1}[tag]
        metrics = {name: finite_or_none(summary[name]) for name in STATIC_METRICS}
        case = {
            "tag": directory.name,
            "q_initial": q_initial,
            "nx": nx,
            "h_over_M": 1.0/nx,
            "cycles": int(round(summary["cycles"])),
            "time_over_M": summary["time"],
            "finite": all(value is not None for value in metrics.values()),
            "q_est_initial": finite_or_none(initial_user[5]),
            "q_analytic_initial": finite_or_none(initial_user[6]),
            "q_est_final": finite_or_none(final_user[5]),
            "q_analytic_final": finite_or_none(final_user[6]),
            "q_est_minus_analytic_final": finite_or_none(final_user[7]),
            "epsilon_G_mean_final": finite_or_none(final_user[13]),
            "epsilon_G_variance_final": finite_or_none(final_user[14]),
            "q_effective_sample_size": finite_or_none(final_user[9]),
            "q_cell_count": int(round(final_user[12])),
            "GH_L2sq_final": finite_or_none(final_native[2]),
            "reduction_L2sq_final": finite_or_none(final_native[3]),
            "curl_L2sq_final": finite_or_none(final_native[4]),
            "G_condition_max_final": finite_or_none(final_native[15]),
            "bad_state": finite_or_none(final_native[21]),
            "metrics": metrics,
            "summary": str(summaries[0]),
        }
        cases.append(case)
    if len(cases) != 9:
        raise ValueError("expected nine static cases, found {}".format(len(cases)))
    return cases


def convergence_orders(cases):
    result = {}
    for q_initial in (0.9, 1.0, 1.1):
        ladder = sorted(
            (case for case in cases if case["q_initial"] == q_initial),
            key=lambda case: case["h_over_M"], reverse=True)
        q_result = {}
        for metric in STATIC_METRICS:
            values = [case["metrics"][metric] for case in ladder]
            hs = [case["h_over_M"] for case in ladder]
            roundoff_floor = max(values) <= 1.0e-10
            q_result[metric] = {
                "p_16_to_24": None if roundoff_floor else observed_order(
                    values[0], values[1], hs[0], hs[1]),
                "p_24_to_32": None if roundoff_floor else observed_order(
                    values[1], values[2], hs[1], hs[2]),
                "monotone": None if roundoff_floor else (
                    values[0] >= values[1] >= values[2]),
                "roundoff_floor": roundoff_floor,
            }
        result["{:.1f}".format(q_initial)] = q_result
    return result


def markdown_report(result):
    lines = [
        "# Static q-controlled trumpet, t=0.1M",
        "",
        "All primary values use the complete puncture-stencil exclusion mask.",
        "Regions are r=[0.125,0.25), [0.25,0.375), and [0.375,0.5) M.",
        "",
        "| q | h/M | metric r1 | constraint r1 | lapse r1 | shift r1 | q_est-q_T |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sorted(result["cases"], key=lambda item: (item["q_initial"], -item["h_over_M"])):
        metric = case["metrics"]
        lines.append(
            "| {q:.1f} | 1/{nx} | {g:.6e} | {c:.6e} | {a:.6e} | "
            "{b:.6e} | {dq:+.6e} |".format(
                q=case["q_initial"], nx=case["nx"],
                g=metric["physical_metric_region1"],
                c=metric["constraint_region1"],
                a=metric["physical_lapse_region1"],
                b=metric["physical_shift_region1"],
                dq=case["q_est_minus_analytic_final"]))
    lines.extend([
        "",
        "| q | quantity | p(16,24) | p(24,32) | monotone |",
        "|---:|:---|---:|---:|:---:|",
    ])
    selected = ("physical_metric_region1", "constraint_region1",
                "physical_lapse_region1", "physical_shift_region1")
    for q_key in ("0.9", "1.0", "1.1"):
        for metric in selected:
            item = result["orders"][q_key][metric]
            p12 = "n/a" if item["p_16_to_24"] is None else "{:.3f}".format(item["p_16_to_24"])
            p23 = "n/a" if item["p_24_to_32"] is None else "{:.3f}".format(item["p_24_to_32"])
            monotone = "n/a" if item["monotone"] is None else (
                "yes" if item["monotone"] else "no")
            lines.append("| {} | {} | {} | {} | {} |".format(
                q_key, metric, p12, p23, monotone))
    lines.extend([
        "",
        "The innermost physical-metric Linf is not monotone for either static",
        "mismatch: q=0.9 gives 4.840438e-2, 6.566646e-3, 1.120053e-2 and",
        "q=1.1 gives 6.657521e-2, 1.294006e-2, 1.559001e-2.  This negative",
        "result is retained; no additional post-hoc puncture mask was applied.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    cases = discover_static(args.static_root)
    result = {
        "scope": "controller-off static reference, t=0.1M",
        "primary_mask": "complete FD/KO stencil support box",
        "cases": cases,
        "orders": convergence_orders(cases),
    }
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown_report(result))


if __name__ == "__main__":
    main()
