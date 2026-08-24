#!/usr/bin/env python3
"""Reduce the bounded native-VC Figure-3 common-tree history experiment."""

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


CASES = ("n128", "n256", "n512")
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_history(path: Path) -> dict[str, np.ndarray]:
    labels: dict[str, int] = {}
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
        elif line.strip():
            rows.append([float(value) for value in line.split()])
    required = {"time", "dt", "axisTau", "axisKret", "axisLapse", "max_abs_K",
                "maxAbsKret", "maxRefLev", "nmb_total", *CONSTRAINTS}
    require(not required - labels.keys(), f"{path}: missing {sorted(required - labels.keys())}")
    array = np.asarray(rows, dtype=float)
    require(array.ndim == 2 and len(array) > 1 and np.isfinite(array).all(),
            f"{path}: invalid history")
    # Interrupted runs can contain repeated final-output cycles. Retain the last
    # row at each coordinate time and require the resulting series to be ordered.
    by_time = {float(row[labels["time"]]): row for row in array}
    array = np.asarray([by_time[key] for key in sorted(by_time)], dtype=float)
    require(np.all(np.diff(array[:, labels["time"]]) > 0.0), f"{path}: nonmonotone time")
    return {name: array[:, index] for name, index in labels.items()}


def load_replay(root: Path) -> list[dict]:
    files = list(root.glob("*.amr_history_replay.jsonl"))
    require(len(files) == 1, f"{root}: replay ledger inventory")
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    require(rows and all(row.get("exact_match") is True for row in rows),
            f"{root}: replay mismatch")
    return rows


def load_authority(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [row for row in rows if row.get("type") == "event"]
    require(events and events[0]["event"] == 0, "invalid authority")
    return events


def interp(table: dict[str, np.ndarray], x: str, y: str, points: np.ndarray) -> np.ndarray:
    return np.interp(points, table[x], table[y])


def write_csv(path: Path, rows: list[dict]) -> None:
    require(rows, f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timestep_causality(path: Path) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    by_time = {float(row["time"]): row for row in raw}
    rows = [by_time[key] for key in sorted(by_time)]
    time = np.asarray([float(row["time"]) for row in rows])
    speed = np.asarray([float(row["max_coordinate_speed"]) for row in rows])
    distance = np.zeros_like(time)
    distance[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(time))
    output = [{"time": t, "max_coordinate_speed": v, "causal_distance": d,
               "trusted_origin_margin": 16.0 - d,
               "trusted_r8_margin": 8.0 - d}
              for t, v, d in zip(time, speed, distance)]
    return output, {"terminal_time": float(time[-1]),
                    "maximum_coordinate_speed": float(np.max(speed)),
                    "integrated_causal_distance": float(distance[-1]),
                    "trusted_radius_at_terminal": float(max(0.0, 16.0 - distance[-1]))}


def compact_event(event: dict) -> dict:
    """Retain authority provenance without embedding the full leaf inventory."""
    return {key: event[key] for key in
            ("event", "time", "cycle", "leaf_count", "max_level", "tree_checksum",
             "created", "deleted", "balance_induced", "requested_refine",
             "requested_derefine") if key in event}


def event_jumps(histories: dict[str, dict[str, np.ndarray]],
                authority: list[dict]) -> list[dict]:
    """Bracket each recorded transaction by the nearest history samples."""
    rows: list[dict] = []
    for event in authority[1:]:
        event_time = float(event["time"])
        for case, table in histories.items():
            before = int(np.searchsorted(table["time"], event_time, side="right") - 1)
            after = before + 1
            if before < 0 or after >= len(table["time"]):
                continue
            for field in CONSTRAINTS:
                old = float(table[field][before]); new = float(table[field][after])
                rows.append({"resolution": case, "event": int(event["event"]),
                             "event_time": event_time,
                             "before_time": float(table["time"][before]),
                             "after_time": float(table["time"][after]),
                             "field": field, "before": old, "after": new,
                             "ratio": new / old if old != 0.0 else math.nan,
                             "delta_log10_abs": (math.log10(abs(new)) - math.log10(abs(old))
                                                  if old != 0.0 and new != 0.0
                                                  else math.nan),
                             "leaf_count": int(event["leaf_count"]),
                             "max_level": int(event["max_level"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = {case: getattr(args, case) for case in CASES}
    histories = {case: read_history(next(roots[case].glob(f"{case}.z4c.user.hst")))
                 for case in CASES}
    authority = load_authority(args.authority)
    replay = {case: load_replay(roots[case]) for case in ("n128", "n512")}
    for case, rows in replay.items():
        expected = authority[1:len(rows) + 1]
        require([row["event"] for row in rows] == [row["event"] for row in expected],
                f"{case}: event indices differ")
        require([row["tree_checksum"] for row in rows] ==
                [row["tree_checksum"] for row in expected], f"{case}: trees differ")

    low_tau = max(table["axisTau"][0] for table in histories.values())
    high_tau = min(table["axisTau"][-1] for table in histories.values())
    require(high_tau > low_tau, "empty common proper-time interval")
    tau = np.linspace(low_tau, high_tau, 401)
    convergence_rows: list[dict] = []
    series: dict[str, dict[str, np.ndarray]] = {case: {} for case in CASES}
    for case, table in histories.items():
        for field in (*CONSTRAINTS, "axisKret", "maxAbsKret", "dt", "max_abs_K",
                      "nmb_total", "maxRefLev", "axisLapse"):
            series[case][field] = interp(table, "axisTau", field, tau)
    for field in CONSTRAINTS:
        for index, proper_time in enumerate(tau):
            e1 = abs(series["n128"][field][index] - series["n256"][field][index])
            e2 = abs(series["n256"][field][index] - series["n512"][field][index])
            q = e1 / e2 if e2 > 0.0 else math.nan
            convergence_rows.append({"axisTau": proper_time, "field": field,
                                     "n128": series["n128"][field][index],
                                     "n256": series["n256"][field][index],
                                     "n512": series["n512"][field][index],
                                     "E_128_256": e1, "E_256_512": e2,
                                     "Q": q, "p": math.log2(q) if q > 0.0 else math.nan})

    history_rows: list[dict] = []
    for case, table in histories.items():
        for index, value in enumerate(table["time"]):
            history_rows.append({"resolution": case, **{
                field: float(table[field][index]) for field in
                ("time", "axisTau", "dt", *CONSTRAINTS, "axisKret", "maxAbsKret",
                 "max_abs_K", "nmb_total", "maxRefLev", "axisLapse")}})

    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "constraints_vs_time.csv", history_rows)
    write_csv(output / "convergence.csv", convergence_rows)
    jump_rows = event_jumps(histories, authority)
    write_csv(output / "amr_event_constraint_jumps.csv", jump_rows)
    causal_rows, causal_summary = timestep_causality(roots["n256"] /
                                                     "z4c_timestep_contract.csv")
    write_csv(output / "causal_protection.csv", causal_rows)

    colors = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
    event3_time = float(authority[3]["time"])
    event3_tau = float(np.interp(event3_time, histories["n256"]["time"],
                                 histories["n256"]["axisTau"]))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for case in CASES:
            axis.semilogy(histories[case]["axisTau"], histories[case][field],
                          color=colors[case], label=case.upper())
        axis.set_title(field); axis.set_xlabel(r"central proper time $\tau$")
        axis.axvline(event3_tau, color="black", linestyle=":", linewidth=0.9)
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    fig.savefig(figures / "constraints_vs_time_global.png", dpi=220)
    fig.savefig(figures / "constraints_vs_time_global.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        selected = [row for row in convergence_rows if row["field"] == field]
        axis.plot([row["axisTau"] for row in selected], [row["p"] for row in selected])
        axis.axhline(4.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set_title(field); axis.set_xlabel(r"central proper time $\tau$")
        axis.set_ylabel("effective order p"); axis.grid(alpha=0.25)
    fig.savefig(figures / "constraint_convergence_order_vs_time.png", dpi=220)
    fig.savefig(figures / "constraint_convergence_order_vs_time.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for case in CASES:
        table = histories[case]
        axes[0, 0].semilogy(table["axisTau"], np.maximum(abs(table["axisKret"]), 1e-300),
                           color=colors[case], label=case.upper())
        axes[0, 1].semilogy(table["axisTau"], np.maximum(table["maxAbsKret"], 1e-300),
                           color=colors[case])
        axes[1, 0].semilogy(table["axisTau"], table["dt"], color=colors[case])
        axes[1, 1].plot(table["axisTau"], table["maxRefLev"], color=colors[case])
    axes[0, 0].set_title("origin |Kretschmann|"); axes[0, 0].legend()
    axes[0, 1].set_title("domain max |Kretschmann|")
    axes[1, 0].set_title("timestep"); axes[1, 1].set_title("maximum physical AMR level")
    for axis in axes.flat:
        axis.set_xlabel(r"central proper time $\tau$"); axis.grid(alpha=0.25)
    fig.savefig(figures / "curvature_timestep_level.png", dpi=220)
    fig.savefig(figures / "curvature_timestep_level.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True,
                             constrained_layout=True)
    for case in CASES:
        table = histories[case]
        axes[0].semilogy(table["axisTau"], table["dt"], color=colors[case],
                         label=case.upper())
        axes[1].step(table["axisTau"], table["maxRefLev"], where="post",
                     color=colors[case])
        axes[2].step(table["axisTau"], table["nmb_total"], where="post",
                     color=colors[case])
    axes[0].set_ylabel("dt"); axes[0].legend()
    axes[1].set_ylabel("max level")
    axes[2].set_ylabel("MeshBlocks")
    axes[2].set_xlabel(r"central proper time $\tau$")
    for axis in axes:
        axis.axvline(event3_tau, color="black", linestyle=":", linewidth=0.9)
        axis.grid(alpha=0.25, which="both")
    fig.savefig(figures / "timestep_level_meshblocks.png", dpi=220)
    fig.savefig(figures / "timestep_level_meshblocks.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    event3_ratios: dict[str, dict[str, float]] = {}
    for axis, field in zip(axes.flat, CONSTRAINTS):
        selected = [row for row in jump_rows
                    if row["event"] == 3 and row["field"] == field]
        ratios = {row["resolution"]: float(row["ratio"]) for row in selected}
        event3_ratios[field] = ratios
        axis.bar(CASES, [ratios[case] for case in CASES],
                 color=[colors[case] for case in CASES])
        axis.set_yscale("log"); axis.set_title(field)
        axis.set_ylabel("post/pre bracketing ratio")
        axis.grid(alpha=0.25, axis="y", which="both")
    fig.suptitle("Authority event 3: level-5 to level-4 derefinement")
    fig.savefig(figures / "event3_constraint_jump_ratios.png", dpi=220)
    fig.savefig(figures / "event3_constraint_jump_ratios.pdf")
    plt.close(fig)

    reference: dict[str, tuple[list[float], list[float]]] = {}
    with args.reference.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values = reference.setdefault(row["series"], ([], []))
            values[0].append(float(row["tau"])); values[1].append(float(row["log10_abs_I"]))
    fig, axis = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        axis.plot(xx, yy, linewidth=1.2, alpha=0.70, label=f"paper {name}")
    for case in CASES:
        table = histories[case]
        mask = np.abs(table["axisKret"]) > 0.0
        axis.plot(table["axisTau"][mask], np.log10(np.abs(table["axisKret"][mask])),
                  color=colors[case], linewidth=1.5, label=f"VC {case.upper()}")
    axis.set_xlim(0.0, 15.0); axis.set_ylim(-8.0, 8.0)
    axis.set_xlabel(r"central proper time $\tau$")
    axis.set_ylabel(r"$\log_{10}|I(0)|$")
    axis.grid(alpha=0.22); axis.legend(ncol=2, fontsize=8)
    fig.savefig(figures / "figure3_vc_overlay.png", dpi=240)
    fig.savefig(figures / "figure3_vc_overlay.pdf")
    plt.close(fig)

    for case in CASES:
        table = histories[case]
        mask = np.abs(table["axisKret"]) > 0.0
        fig, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
        for name, (xx, yy) in reference.items():
            axis.plot(xx, yy, linewidth=1.1, alpha=0.55, label=f"paper {name}")
        axis.plot(table["axisTau"][mask], np.log10(np.abs(table["axisKret"][mask])),
                  color=colors[case], linewidth=1.6, label=f"VC {case.upper()}")
        axis.set_xlim(0.0, 15.0); axis.set_ylim(-8.0, 8.0)
        axis.set_xlabel(r"central proper time $\tau$")
        axis.set_ylabel(r"$\log_{10}|I(0)|$")
        axis.grid(alpha=0.22); axis.legend(fontsize=8)
        fig.savefig(figures / f"figure3_vc_{case.upper()}.png", dpi=240)
        fig.savefig(figures / f"figure3_vc_{case.upper()}.pdf")
        plt.close(fig)

    summary = {
        "schema": "z4c_vc_figure3_history_analysis_v1",
        "common_axis_proper_time": [float(low_tau), float(high_tau)],
        "authority": {"events": len(authority), "last": compact_event(authority[-1])},
        "replay": {case: {"events": len(rows), "exact": True,
                          "last_event": rows[-1]["event"]}
                   for case, rows in replay.items()},
        "terminal": {case: {field: float(histories[case][field][-1]) for field in
                             ("time", "axisTau", "dt", *CONSTRAINTS, "max_abs_K",
                              "maxAbsKret", "nmb_total", "maxRefLev")}
                     for case in CASES},
        "causality_n256": causal_summary,
        "median_constraint_order": {
            field: float(np.nanmedian([row["p"] for row in convergence_rows
                                       if row["field"] == field]))
            for field in CONSTRAINTS},
        "amr_event_jump_rows": len(jump_rows),
        "first_catastrophic_transaction": {
            "authority_event": 3,
            "coordinate_time": event3_time,
            "n256_axis_proper_time": event3_tau,
            "kind": "derefinement",
            "constraint_bracketing_ratios": event3_ratios,
        },
    }
    (output / "comparison_summary.history.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
