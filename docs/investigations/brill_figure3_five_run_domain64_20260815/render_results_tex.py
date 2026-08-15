#!/usr/bin/env python3
"""Render the authenticated five-case summary into a compact LaTeX table."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "analysis/five_case_summary.json"
OUTPUT = ROOT / "analysis/generated_results.tex"

ORDER = (
    ("d16_fixed_ko002", r"16", r"fixed $\eta=2$", "0.02"),
    ("d16_fixed_ko05", r"16", r"fixed $\eta=2$", "0.5"),
    ("d16_zero_ko05", r"16", r"zero", "0.5"),
    ("d64_fixed_ko05", r"64", r"fixed $\eta=2$", "0.5"),
    ("d64_zero_ko05", r"64", r"zero", "0.5"),
)


def fmt(value, digits=5):
    if value is None:
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "nonfinite"
    if value == 0.0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.{digits}g}"


def terminal(result: dict) -> str:
    if result.get("reached_target_t20"):
        return r"reached $t=20$"
    fatals = " ".join(result.get("fatal_lines", [])).lower()
    if "max_nmb_per_rank" in fatals or "meshblocks" in fatals:
        return "AMR capacity guard"
    if result.get("chi_rejections") or "chi" in fatals:
        return "strict positive-$\\chi$ gate"
    if "nonfinite" in fatals:
        return "nonfinite diagnostic gate"
    return f"exit {result.get('exit_code')}"


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rlllrrrrrl}",
        r"\toprule",
        r"case & outer $R=Z$ & shift & KO & $t_{\rm last}$ & "
        r"$\tau_{\rm last}$ & $\alpha_{\rm axis}$ & $L_{\max}$ & AH & terminal \\",
        r"\midrule",
    ]
    for index, (key, domain, shift, ko) in enumerate(ORDER, 1):
        case = data["cases"][key]
        last = case["last_finite_history"]
        result = case["result"]
        lines.append(
            f"{index} & {domain} & {shift} & {ko} & "
            f"{fmt(last.get('time'))} & {fmt(last.get('axisTau'))} & "
            f"{fmt(last.get('axisLapse'))} & {case['max_refinement_level']} & "
            f"{'yes' if case['horizon_found'] else 'no'} & "
            f"{terminal(result)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{The five requested controls.  Last values are the last fully finite history rows, not post-failure extrapolations.  The AH column is recorded telemetry only; the horizon finder was disabled for this diagnostic series.}",
            r"\label{tab:five-cases}",
            r"\end{table*}",
        ]
    )
    found = [key for key, case in data["cases"].items() if case["horizon_found"]]
    if found:
        lines.append(
            r"\paragraph{Horizon status.} At least one history reported an apparent "
            r"horizon; see the machine-readable summary for the exact cases."
        )
    else:
        lines.append(
            r"\paragraph{Horizon status.} No finite history row in any of the five "
            r"cases reported an apparent horizon."
        )
    OUTPUT.write_text("\n".join(lines) + "\n")
    print("GENERATED_RESULTS_TEX_COMPLETE")


if __name__ == "__main__":
    main()
