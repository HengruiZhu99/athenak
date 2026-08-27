#!/usr/bin/env python3
"""Reduce the bounded Rout16/Rout128 original-versus-CPBC discriminator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("A", "B", "C", "D")
LABELS = {
    "A": "Rout16 original",
    "B": "Rout16 Bjorhus",
    "C": "Rout128 original",
    "D": "Rout128 Bjorhus",
}
COLORS = {"A": "#984ea3", "B": "#ff7f00", "C": "#377eb8", "D": "#4daf4a"}
STYLES = {"A": "--", "B": "-", "C": "--", "D": "-"}
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def history(root: Path) -> dict[str, np.ndarray]:
    files = list(root.glob("*.z4c.user.hst"))
    require(len(files) == 1, f"{root}: expected one history")
    labels: dict[str, int] = {}
    rows: dict[float, list[float]] = {}
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
        elif line.strip():
            row = [float(value) for value in line.split()]
            rows[row[labels["time"]]] = row
    required = {"time", "axisTau", "axisKret", "dt", "maxRefLev", "nmb_total",
                "C-Linf", "C-rho", "C-z", *CONSTRAINTS}
    require(not required - labels.keys(), f"{root}: missing history fields")
    array = np.asarray([rows[key] for key in sorted(rows)])
    require(np.isfinite(array).all(), f"{root}: nonfinite history")
    return {name: array[:, index] for name, index in labels.items()}


def case_status(root: Path) -> dict[str, str]:
    def read(name: str) -> str:
        path = root / name
        return path.read_text(encoding="utf-8").strip() if path.exists() else "MISSING"
    return {
        "disposition": read("disposition"),
        "run_status": read("run-status"),
        "orchestration_status": read("orchestration-status"),
    }


def replay_rows(root: Path) -> list[dict]:
    files = list(root.glob("*.amr_history_replay.jsonl"))
    require(len(files) == 1, f"{root}: expected one replay log")
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    require(rows and all(row.get("exact_match") is True for row in rows),
            f"{root}: replay mismatch")
    return rows


def max_log_deviation(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    end = min(left["axisTau"][-1], right["axisTau"][-1])
    tau = np.linspace(max(left["axisTau"][0], right["axisTau"][0]), end, 801)
    a = np.interp(tau, left["axisTau"], left["axisKret"])
    b = np.interp(tau, right["axisTau"], right["axisKret"])
    valid = (np.abs(a) > 0.0) & (np.abs(b) > 0.0)
    return float(np.max(np.abs(np.log10(np.abs(a[valid])) - np.log10(np.abs(b[valid])))))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(rows, f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in CASES:
        parser.add_argument(f"--{case.lower()}", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {case: getattr(args, case.lower()) for case in CASES}
    tables = {case: history(root) for case, root in roots.items()}
    statuses = {case: case_status(root) for case, root in roots.items()}
    replay = {"B": replay_rows(roots["B"]), "D": replay_rows(roots["D"])}
    test_text = args.test_log.read_text(encoding="utf-8")
    match = re.search(r"induced_outgoing_rate=([0-9.eE+-]+)", test_text)
    require(match is not None and "tests passed" in test_text, "missing passing Bjorhus test")
    induced_outgoing = float(match.group(1))

    history_rows: list[dict] = []
    fields = ("time", "axisTau", "axisKret", "dt", "maxRefLev", "nmb_total",
              "C-Linf", "C-rho", "C-z", *CONSTRAINTS)
    for case, table in tables.items():
        for index in range(len(table["time"])):
            history_rows.append({"case": case, "label": LABELS[case], **{
                field: float(table[field][index]) for field in fields
            }})

    pair_rows: list[dict] = []
    for domain, original, cpbc in (("Rout16", "A", "B"), ("Rout128", "C", "D")):
        common_end = min(tables[original]["time"][-1], tables[cpbc]["time"][-1])
        for field in CONSTRAINTS:
            a = float(np.interp(common_end, tables[original]["time"], tables[original][field]))
            b = float(np.interp(common_end, tables[cpbc]["time"], tables[cpbc][field]))
            pair_rows.append({
                "domain": domain,
                "common_terminal_time": common_end,
                "field": field,
                "original": a,
                "bjorhus": b,
                "bjorhus_over_original": b / a if a != 0.0 else math.nan,
                "log10_ratio": math.log10(b / a) if a > 0.0 and b > 0.0 else math.nan,
                "qualified_complete_pair": False,
            })

    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cpbc_history_compact.csv", history_rows)
    write_csv(output / "cpbc_terminal_constraint_ratios.csv", pair_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for case in CASES:
            table = tables[case]
            axis.semilogy(table["time"], table[field], color=COLORS[case],
                          linestyle=STYLES[case], label=LABELS[case])
        axis.set_title(field)
        axis.set_xlabel("coordinate time")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figures / "cpbc_constraint_histories.png", dpi=220)
    fig.savefig(figures / "cpbc_constraint_histories.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for axis, cases, domain in zip(axes, (("A", "B"), ("C", "D")), ("Rout16", "Rout128")):
        for case in cases:
            table = tables[case]
            valid = np.abs(table["axisKret"]) > 0.0
            axis.plot(table["axisTau"][valid], np.log10(np.abs(table["axisKret"][valid])),
                      color=COLORS[case], linestyle=STYLES[case], label=LABELS[case])
        axis.set_title(domain)
        axis.set_xlabel(r"central proper time $\tau$")
        axis.set_ylabel(r"$\log_{10}|I(0)|$")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(figures / "cpbc_central_figure3_trace.png", dpi=220)
    fig.savefig(figures / "cpbc_central_figure3_trace.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    values = [0.7, 2.0e-13, induced_outgoing]
    labels = ["max incoming\nbefore", "incoming after\n(test bound)",
              "max induced\noutgoing"]
    axis.bar(labels, values, color=["#984ea3", "#4daf4a", "#ff7f00"])
    axis.set_yscale("log")
    axis.set_ylabel("manufactured characteristic-rate magnitude")
    axis.set_title("Bjorhus manufactured incoming/outgoing diagnostic")
    axis.grid(alpha=0.25, axis="y", which="both")
    fig.savefig(figures / "bjorhus_characteristic_diagnostic.png", dpi=220)
    fig.savefig(figures / "bjorhus_characteristic_diagnostic.pdf")
    plt.close(fig)

    summary = {
        "schema": "z4c_vc_ko05_cpbc_discriminator_v1",
        "terminal": {case: {field: float(table[field][-1]) for field in fields}
                     for case, table in tables.items()},
        "status": statuses,
        "replay": {case: {"events": len(rows), "all_exact": True,
                           "last_event": rows[-1]["event"]}
                   for case, rows in replay.items()},
        "central_log10_axisKret_max_deviation": {
            "Rout16": max_log_deviation(tables["A"], tables["B"]),
            "Rout128": max_log_deviation(tables["C"], tables["D"]),
        },
        "manufactured": {
            "incoming_after_absolute_bound": 2.0e-13,
            "maximum_induced_outgoing_rate": induced_outgoing,
        },
        "limitations": [
            "The planned four-way discriminator is incomplete and no CPBC comparison is qualified.",
            "A and C reached t=6.5; B failed closed at t=3.244461; D was intentionally cancelled at t=1.508791 and is excluded from conclusions.",
            "The Bjorhus option cancels incoming principal rates but does not preserve all outgoing rates.",
            "A replay-matched tree isolates boundary treatment but suppresses boundary-induced topology divergence.",
        ],
    }
    (output / "cpbc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
