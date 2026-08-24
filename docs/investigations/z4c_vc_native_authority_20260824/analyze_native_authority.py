#!/usr/bin/env python3
"""Summarize the repaired native-authority Brill campaign."""

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


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
COLORS = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}


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
    array = np.asarray(rows, dtype=float)
    require(array.ndim == 2 and np.isfinite(array).all(), f"invalid history {path}")
    return {name: array[:, index] for name, index in labels.items()}


def read_events(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [row for row in rows if row.get("type") == "event"]
    require(events and [row["event"] for row in events] == list(range(len(events))),
            f"invalid authority {path}")
    return events


def first_at(table: dict[str, np.ndarray], field: str, threshold: float) -> dict | None:
    found = np.flatnonzero(table[field] >= threshold)
    if not len(found):
        return None
    index = int(found[0])
    return {name: float(table[name][index]) for name in
            ("time", "axisTau", "cycle", field, "dt", "maxRefLev", "nmb_total")}


def event_summary(event: dict) -> dict:
    return {key: event.get(key) for key in
            ("event", "time", "cycle", "leaf_count", "max_level", "tree_checksum",
             "requested_refine", "requested_derefine", "created", "deleted",
             "balance_induced")}


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=230)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root
    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    runs = root / "runs"
    early_authority_path = (runs / "n256_native_record_t2p5_v2_dethist" /
                            "n256_native_authority.jsonl")
    failure_authority_path = (runs / "n256_native_record_tau7_segment2_v1" /
                              "n256_native_authority.jsonl")
    historical_authority_path = root / "evidence/historical/n256_historical_authority.jsonl"
    early_events = read_events(early_authority_path)
    failure_events = read_events(failure_authority_path)
    historical_events = read_events(historical_authority_path)
    require(len(early_events) == 4, "early authority should end at event 3")
    for index in range(4):
        require(early_events[index]["tree_checksum"] ==
                historical_events[index]["tree_checksum"],
                f"pre-divergence checksum mismatch at event {index}")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True,
                             constrained_layout=True)
    for label, events, color in (("historical pre-repair authority", historical_events,
                                  "#984ea3"),
                                 ("fresh repaired authority", failure_events, "#ff7f00")):
        time = np.asarray([float(event["time"]) for event in events])
        leaves = np.asarray([event["leaf_count"] for event in events])
        level = np.asarray([event["max_level"] for event in events])
        axes[0].step(time, leaves, where="post", label=label, color=color)
        axes[1].step(time, level, where="post", color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("accepted leaves")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("logical max level")
    axes[1].set_xlabel("coordinate time")
    for axis in axes:
        axis.axvline(float(early_events[3]["time"]), color="black", linestyle=":",
                     linewidth=0.9)
        axis.grid(alpha=0.25, which="both")
    save(fig, figures / "authority_historical_vs_repaired")

    tau4_roots = {
        "n128": runs / "n128_native_replay_tau4_v1",
        "n256": runs / "n256_native_replay_tau4_v1",
        "n512": runs / "n512_native_replay_tau4_v1",
    }
    tau4 = {case: read_history(next(path.glob("*.z4c.user.hst")))
            for case, path in tau4_roots.items()}
    common_tau = np.linspace(0.0, min(table["axisTau"][-1] for table in tau4.values()), 501)
    orders: dict[str, np.ndarray] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        values = {}
        for case, table in tau4.items():
            values[case] = np.interp(common_tau, table["axisTau"], table[field])
            axis.semilogy(table["axisTau"], table[field], color=COLORS[case],
                          label=case.upper())
        e1 = np.abs(values["n128"] - values["n256"])
        e2 = np.abs(values["n256"] - values["n512"])
        with np.errstate(divide="ignore", invalid="ignore"):
            orders[field] = np.log2(e1 / e2)
        axis.set_title(field)
        axis.set_xlabel(r"central proper time $\tau$")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    save(fig, figures / "tau4_constraints")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        axis.plot(common_tau, orders[field], color="#4c72b0")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(4.0, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(field)
        axis.set_xlabel(r"central proper time $\tau$")
        axis.set_ylabel("effective order")
        axis.grid(alpha=0.25)
    save(fig, figures / "tau4_constraint_order")

    failure_history_path = (runs / "n256_native_record_tau7_segment2_v1" /
                            "n256_native_record.z4c.user.hst")
    failure = read_history(failure_history_path)
    event_time = np.asarray([float(event["time"]) for event in failure_events])
    event_leaves = np.asarray([event["leaf_count"] for event in failure_events])
    event_level = np.asarray([event["max_level"] for event in failure_events])
    leaf_delta = np.diff(event_leaves)
    leaf_sign = np.sign(leaf_delta)
    health = {
        "accepted_states": len(failure_events),
        "accepted_transactions": len(failure_events) - 1,
        "requested_refine_total": sum(int(event.get("requested_refine", 0))
                                      for event in failure_events),
        "requested_derefine_total": sum(int(event.get("requested_derefine", 0))
                                        for event in failure_events),
        "created_total": sum(int(event.get("created", 0))
                             for event in failure_events),
        "deleted_total": sum(int(event.get("deleted", 0))
                             for event in failure_events),
        "balance_induced_total": sum(int(event.get("balance_induced", 0))
                                     for event in failure_events),
        "transactions_growing_leaves": int(np.count_nonzero(leaf_delta > 0)),
        "transactions_shrinking_leaves": int(np.count_nonzero(leaf_delta < 0)),
        "transactions_unchanged_leaves": int(np.count_nonzero(leaf_delta == 0)),
        "leaf_change_sign_turnovers": int(np.count_nonzero(
            leaf_sign[:-1] * leaf_sign[1:] < 0)),
        "events_with_both_request_types": sum(
            int(event.get("requested_refine", 0)) > 0 and
            int(event.get("requested_derefine", 0)) > 0
            for event in failure_events),
        "initial_leaf_count": int(event_leaves[0]),
        "maximum_leaf_count": int(np.max(event_leaves)),
        "terminal_leaf_count": int(event_leaves[-1]),
        "maximum_logical_level": int(np.max(event_level)),
        "maximum_physical_refinement_level": int(np.max(failure["maxRefLev"])),
    }

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True,
                             constrained_layout=True)
    for field in CONSTRAINTS:
        axes[0].semilogy(failure["time"], failure[field], label=field)
    axes[0].set_ylabel("global constraint integral")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].semilogy(failure["time"], np.maximum(failure["maxAbsKret"], 1e-300),
                    label="domain max |Kretschmann|", color="#e41a1c")
    axes[1].semilogy(failure["time"], np.maximum(np.abs(failure["axisKret"]), 1e-300),
                    label="axis-center |Kretschmann|", color="#377eb8")
    axes[1].set_ylabel("curvature")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(failure["time"], failure["dt"], color="#4daf4a")
    axes[2].set_ylabel("dt")
    axes[2].set_xlabel("coordinate time")
    for axis in axes:
        axis.axvline(float(failure_events[4]["time"]), color="black", linestyle=":",
                     linewidth=0.9)
        axis.grid(alpha=0.25, which="both")
    save(fig, figures / "failure_constraints_curvature_timestep")

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True,
                             constrained_layout=True)
    axes[0].step(event_time, event_leaves, where="post", color="#ff7f00")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("accepted leaves")
    axes[1].step(event_time, event_level, where="post", color="#984ea3")
    axes[1].set_ylabel("logical max level")
    axes[2].step(failure["time"], failure["maxRefLev"], where="post", color="#377eb8")
    axes[2].set_ylabel("physical max level")
    axes[2].set_xlabel("coordinate time")
    for axis in axes:
        axis.set_xlim(9.0, float(failure["time"][-1]) + 0.02)
        axis.grid(alpha=0.25, which="both")
    save(fig, figures / "failure_native_amr_health")

    reference: dict[str, tuple[list[float], list[float]]] = {}
    with args.reference.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values = reference.setdefault(row["series"], ([], []))
            values[0].append(float(row["tau"]))
            values[1].append(float(row["log10_abs_I"]))
    fig, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    for name, (xvalues, yvalues) in reference.items():
        axis.plot(xvalues, yvalues, alpha=0.55, linewidth=1.0, label=f"paper {name}")
    for case, table in tau4.items():
        mask = np.abs(table["axisKret"]) > 0.0
        axis.plot(table["axisTau"][mask], np.log10(np.abs(table["axisKret"][mask])),
                  color=COLORS[case], linewidth=1.5, label=case.upper())
    axis.set_xlim(0.0, 4.1)
    axis.set_xlabel(r"central proper time $\tau$")
    axis.set_ylabel(r"$\log_{10}|I(0)|$")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    save(fig, figures / "figure3_tau4_overlay")

    first_constraint_growth = {
        field: first_at(failure, field, 1.0) for field in CONSTRAINTS
    }
    terminal = {field: float(failure[field][-1]) for field in
                ("time", "axisTau", "cycle", "dt", *CONSTRAINTS, "max_abs_K",
                 "maxAbsKret", "axisKret", "axisLapse", "maxRefLev", "nmb_total")}
    history_summary = json.loads(
        (root / "analysis/history-tau4-v1/comparison_summary.history.json").read_text())
    field_summary = json.loads(
        (root / "analysis/fields-tau4-v1/field_summary.json").read_text())
    summary = {
        "schema": "z4c_vc_native_authority_summary_v1",
        "verdict": "NATIVE_AMR_UNSTABLE",
        "authority": {
            "common_events": [event_summary(event) for event in early_events],
            "historical_first_post_repair_event": event_summary(historical_events[4]),
            "fresh_first_post_repair_event": event_summary(failure_events[4]),
            "failure_terminal_event": event_summary(failure_events[-1]),
            "failure_event_count": len(failure_events),
            "native_amr_health": health,
        },
        "tau4": {
            "history_median_constraint_order": history_summary["median_constraint_order"],
            "history_terminal": history_summary["terminal"],
            "field_minimum_trusted_core_order":
                field_summary["trusted_core_minimum_order"],
        },
        "failure": {
            "first_constraint_norm_ge_1": first_constraint_growth,
            "first_late_refinement": event_summary(failure_events[4]),
            "first_level7_event": event_summary(failure_events[10]),
            "first_level14_stagnation_cascade": event_summary(failure_events[22]),
            "terminal_history": terminal,
            "terminal_authority": event_summary(failure_events[-1]),
            "job": "57525753",
            "scheduler_disposition": "CANCELLED by user-agent after fail gate",
        },
        "limitations": [
            "The failed N256 stage was not replayed at N128 or N512 because the gate requires stopping.",
            "The evidence does not isolate bulk under-resolution from transfer/interface feedback.",
            "No convergence or Figure-3 reproduction claim extends beyond tau approximately 4.",
        ],
    }
    (output / "native_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
