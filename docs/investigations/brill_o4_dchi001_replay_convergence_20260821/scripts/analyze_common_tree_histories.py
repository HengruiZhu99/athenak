#!/usr/bin/env python3
"""Analyze authenticated common-tree Brill histories and replay ledgers.

This script deliberately limits itself to scalar histories and AMR ledgers.
Field-level convergence is a separate gate because it requires the binary
slice inventory and a common physical sampling operator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
CASES = ("n128", "n256", "n512")
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
REQUIRED_HISTORY = {
    "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
    "max_abs_K", "nmb_total", "maxAbsKret", "maxRefLev", "cycle",
    "axisLapse", "axisTau", "axisKret",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{number}: JSON object required")
        rows.append(row)
    return rows


def read_history(path: Path, segment: int) -> tuple[list[dict[str, float]], list[str]]:
    labels: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if not labels or len(values) <= max(labels.values()):
            raise RuntimeError(f"malformed history row in {path}")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"nonfinite history row in {path}")
        row = {name: values[index] for name, index in labels.items()}
        row["segment"] = float(segment)
        rows.append(row)
    missing = REQUIRED_HISTORY - labels.keys()
    if not rows or missing:
        raise RuntimeError(f"invalid history {path}; missing={sorted(missing)}")
    ordered = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    return rows, ordered


def segment_number(path: Path) -> int:
    match = re.fullmatch(r"segment(\d+)", path.parent.name)
    if not match:
        raise RuntimeError(f"history is not directly below segmentN: {path}")
    return int(match.group(1))


def merge_histories(case_root: Path, case: str) -> tuple[list[dict[str, float]], list[str], list[dict[str, Any]]]:
    paths = sorted(case_root.glob(f"segment*/{case}.z4c.user.hst"))
    if not paths:
        raise RuntimeError(f"no {case} history below {case_root}")
    by_cycle: dict[int, dict[str, float]] = {}
    order: list[str] | None = None
    inputs: list[dict[str, Any]] = []
    for path in paths:
        segment = segment_number(path)
        rows, names = read_history(path, segment)
        if order is None:
            order = names
        elif order != names:
            raise RuntimeError(f"history schema changed in {path}")
        inputs.append({
            "path": str(path), "sha256": sha256(path), "segment": segment,
            "rows": len(rows),
        })
        for row in rows:
            cycle = int(row["cycle"])
            previous = by_cycle.get(cycle)
            if previous is not None:
                for name in names:
                    if previous[name] != row[name]:
                        raise RuntimeError(
                            f"restart overlap mismatch: {case} cycle={cycle} field={name}"
                        )
                continue
            by_cycle[cycle] = row
    merged = [by_cycle[key] for key in sorted(by_cycle)]
    if any(b["time"] <= a["time"] for a, b in zip(merged, merged[1:])):
        raise RuntimeError(f"{case} history time is not strictly increasing")
    return merged, list(order or []), inputs


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def discover_jsonl(case_root: Path, pattern: str) -> list[Path]:
    return sorted(case_root.glob(f"segment*/{pattern}"))


def merge_replay_ledgers(case_root: Path, case: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = discover_jsonl(case_root, f"{case}.amr_history_replay.jsonl")
    if not paths:
        raise RuntimeError(f"no replay ledgers for {case}")
    by_event: dict[int, dict[str, Any]] = {}
    inputs = []
    for path in paths:
        rows = json_lines(path)
        inputs.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})
        for row in rows:
            event = int(row["event"])
            old = by_event.get(event)
            if old is not None and old != row:
                raise RuntimeError(f"replay overlap mismatch: {case} event={event}")
            by_event[event] = row
    rows = [by_event[event] for event in sorted(by_event)]
    if [int(row["event"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"non-contiguous replay events for {case}")
    return rows, inputs


def merge_shadow(case_root: Path, case: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = discover_jsonl(case_root, f"{case}.amr_native_shadow.rank*.jsonl")
    if not paths:
        raise RuntimeError(f"no native-AMR shadow records for {case}")
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    inputs = []
    for path in paths:
        payload = json_lines(path)
        inputs.append({"path": str(path), "sha256": sha256(path), "rows": len(payload)})
        for row in payload:
            key = (
                int(row["cycle"]), int(row["gid"]), int(row.get("authority_event_index", -1)),
                row["native_action"], row["authority_action"],
            )
            old = rows.get(key)
            if old is not None and old != row:
                raise RuntimeError(f"shadow overlap mismatch: {case} key={key}")
            rows[key] = row
    return sorted(rows.values(), key=lambda row: (row["cycle"], row["gid"])), inputs


def summarize_shadow_cycles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce the block-level shadow ledger without discarding its source hash."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["cycle"]), []).append(row)
    output = []
    for cycle in sorted(grouped):
        group = grouped[cycle]
        classifications = Counter(row["classification"] for row in group)
        actions = Counter(row["native_action"] for row in group)
        strongest = max(group, key=lambda row: float(row["raw_dchi"]))
        output.append({
            "cycle": cycle,
            "time": group[0]["time"],
            "tau_c": group[0]["tau_c"],
            "authority_event": any(bool(row.get("authority_event")) for row in group),
            "authority_event_index": max(int(row.get("authority_event_index", -1)) for row in group),
            "records": len(group),
            "agrees": classifications["AGREES"],
            "would_refine_earlier": classifications["WOULD_REFINE_EARLIER"],
            "would_derefine": classifications["WOULD_DEREFINE"],
            "would_not_refine": classifications["WOULD_NOT_REFINE"],
            "other": classifications["OTHER"],
            "native_refine": actions["refine"],
            "native_derefine": actions["derefine"],
            "native_same": actions["same"],
            "max_raw_dchi": strongest["raw_dchi"],
            "max_dchi_over_dx": max(float(row["dchi_over_dx"]) for row in group),
            "strongest_gid": strongest["gid"],
            "strongest_logical_location": strongest["logical_location"],
            "strongest_physical_location": strongest["strongest_physical_location"],
        })
    return output


def verify_replay(
    authority: list[dict[str, Any]], ledgers: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    auth = {int(row["event"]): row for row in authority if row.get("type") == "event"}
    output = []
    summary: dict[str, Any] = {}
    for case, rows in ledgers.items():
        for row in rows:
            event = int(row["event"])
            expected = auth.get(event)
            if expected is None:
                raise RuntimeError(f"{case} replayed unknown authority event {event}")
            exact = (
                row.get("exact_match") is True
                and abs(int(row["ulp_difference"])) <= 1
                and row["tree_checksum"] == expected["tree_checksum"]
                and row["authority_time_hex"] == expected["time_hex"]
            )
            output.append({
                "resolution": case,
                "event": event,
                "authority_time": expected["time"],
                "authority_time_hex": expected["time_hex"],
                "actual_time_hex": row["actual_mesh_time_hex"],
                "ulp_difference": row["ulp_difference"],
                "preceding_timestep_clipped": row["preceding_timestep_clipped"],
                "authority_leaves": expected["leaf_count"],
                "accepted_leaves": row["leaves"],
                "tree_checksum": row["tree_checksum"],
                "exact": exact,
            })
        summary[case] = {
            "events_replayed": len(rows),
            "authority_events_available": len(auth),
            "exact_executed_prefix": bool(rows) and all(item["exact"] for item in output if item["resolution"] == case),
            "max_abs_ulp_difference": max((abs(int(row["ulp_difference"])) for row in rows), default=None),
        }
    return output, summary


def interpolate(rows: list[dict[str, float]], xname: str, yname: str, xx: np.ndarray) -> np.ndarray:
    x = np.array([row[xname] for row in rows], dtype=float)
    y = np.array([row[yname] for row in rows], dtype=float)
    return np.interp(xx, x, y)


def convergence_rows(histories: dict[str, list[dict[str, float]]], trusted_tau_max: float) -> list[dict[str, Any]]:
    low = max(rows[0]["axisTau"] for rows in histories.values())
    high = min(trusted_tau_max, *(rows[-1]["axisTau"] for rows in histories.values()))
    if not high > low:
        raise RuntimeError("empty common trusted central-proper-time interval")
    authority_tau = np.array([
        row["axisTau"] for row in histories["n256"] if low <= row["axisTau"] <= high
    ])
    if authority_tau.size < 2:
        raise RuntimeError("too few authority rows in trusted interval")
    output = []
    for field in CONSTRAINTS:
        values = {
            case: interpolate(rows, "axisTau", field, authority_tau)
            for case, rows in histories.items()
        }
        for index, tau in enumerate(authority_tau):
            e_coarse = abs(values["n128"][index] - values["n256"][index])
            e_fine = abs(values["n256"][index] - values["n512"][index])
            p = math.log2(e_coarse / e_fine) if e_coarse > 0 and e_fine > 0 else math.nan
            output.append({
                "axisTau": tau, "field": field,
                "n128": values["n128"][index], "n256": values["n256"][index],
                "n512": values["n512"][index], "E_128_256": e_coarse,
                "E_256_512": e_fine, "Q": e_coarse / e_fine if e_fine > 0 else math.nan,
                "p": p,
            })
    return output


def bracket(rows: list[dict[str, float]], time: float) -> tuple[dict[str, float], dict[str, float]] | None:
    for before, after in zip(rows, rows[1:]):
        if before["time"] < time <= after["time"]:
            return before, after
    return None


def authority_jump_rows(
    authority: list[dict[str, Any]], histories: dict[str, list[dict[str, float]]]
) -> list[dict[str, Any]]:
    output = []
    for event in authority:
        if event.get("type") != "event" or int(event["event"]) == 0:
            continue
        time = float.fromhex(event["time_hex"])
        for case, rows in histories.items():
            pair = bracket(rows, time)
            if pair is None:
                continue
            before, after = pair
            for field in CONSTRAINTS:
                a, b = before[field], after[field]
                output.append({
                    "event": event["event"], "authority_time": time,
                    "resolution": case, "field": field,
                    "time_before": before["time"], "time_after": after["time"],
                    "tau_before": before["axisTau"], "tau_after": after["axisTau"],
                    "value_before": a, "value_after": b,
                    "ratio": b / a if a != 0 else math.nan,
                    "abs_log10_jump": abs(math.log10(b / a)) if a > 0 and b > 0 else math.nan,
                    "leaves_after": event["leaf_count"], "max_level_after": event["max_level"],
                    "created": event["created"], "deleted": event["deleted"],
                })
    return output


def first_threshold(rows: list[dict[str, float]], field: str, value: float, mode: str) -> dict[str, float] | None:
    for row in rows:
        if (mode == "le" and row[field] <= value) or (mode == "ge" and row[field] >= value):
            return row
    return None


def failure_landmarks(histories: dict[str, list[dict[str, float]]]) -> list[dict[str, Any]]:
    specs = [
        ("dt", 1e-3, "le"), ("dt", 1e-4, "le"), ("dt", 1e-5, "le"),
        ("max_abs_K", 1e2, "ge"), ("max_abs_K", 1e3, "ge"),
        ("max_abs_K", 1e4, "ge"), ("C-norm2", 1e2, "ge"),
        ("C-norm2", 1e4, "ge"), ("C-norm2", 1e6, "ge"),
        ("maxRefLev", 10, "ge"), ("maxRefLev", 15, "ge"),
        ("maxRefLev", 18, "ge"), ("maxRefLev", 20, "ge"),
    ]
    output = []
    for case, rows in histories.items():
        for field, threshold, mode in specs:
            row = first_threshold(rows, field, threshold, mode)
            output.append({
                "resolution": case, "field": field, "threshold": threshold,
                "direction": mode, "reached": row is not None,
                "time": row["time"] if row else "", "axisTau": row["axisTau"] if row else "",
                "cycle": int(row["cycle"]) if row else "", "value": row[field] if row else "",
            })
    return output


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_constraints(histories: dict[str, list[dict[str, float]]], xfield: str, path: Path) -> None:
    colors = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    for ax, field in zip(axes.flat, CONSTRAINTS):
        for case, rows in histories.items():
            xy = [(row[xfield], row[field]) for row in rows if row[field] > 0]
            ax.semilogy([x for x, _ in xy], [y for _, y in xy], label=case.upper(), color=colors[case])
        ax.set_title(field)
        ax.set_xlabel("central proper time" if xfield == "axisTau" else "coordinate time")
        ax.grid(alpha=0.22, which="both")
    axes[0, 0].legend()
    save_figure(fig, path)


def plot_scalar_panels(histories: dict[str, list[dict[str, float]]], output: Path) -> None:
    colors = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    for case, rows in histories.items():
        tau = [row["axisTau"] for row in rows]
        axes[0].semilogy(tau, [max(row["max_abs_K"], 1e-300) for row in rows], color=colors[case], label=f"{case.upper()} max|K|")
        axes[0].semilogy(tau, [max(row["maxAbsKret"], 1e-300) for row in rows], color=colors[case], linestyle="--", label=f"{case.upper()} max Kretschmann")
        axes[1].semilogy(tau, [row["dt"] for row in rows], color=colors[case], label=case.upper())
    axes[0].set_xlabel("central proper time"); axes[0].grid(alpha=.22, which="both"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("central proper time"); axes[1].set_ylabel("dt"); axes[1].grid(alpha=.22, which="both"); axes[1].legend()
    save_figure(fig, output / "curvature_and_timestep_vs_tau.png")

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for case, rows in histories.items():
        tau = [row["axisTau"] for row in rows]
        ax.semilogy(tau, [max(row["max_abs_K"], 1e-300) for row in rows],
                    color=colors[case], label=f"{case.upper()} max|K|")
        ax.semilogy(tau, [max(row["maxAbsKret"], 1e-300) for row in rows],
                    color=colors[case], linestyle="--",
                    label=f"{case.upper()} max Kretschmann")
    ax.set_xlabel("central proper time"); ax.grid(alpha=.22, which="both")
    ax.legend(fontsize=7)
    save_figure(fig, output / "curvature_vs_tau.png")

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for case, rows in histories.items():
        ax.semilogy([row["axisTau"] for row in rows], [row["dt"] for row in rows],
                    color=colors[case], label=case.upper())
    ax.set_xlabel("central proper time"); ax.set_ylabel("accepted dt")
    ax.grid(alpha=.22, which="both"); ax.legend()
    save_figure(fig, output / "timestep_vs_tau.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    for case, rows in histories.items():
        tau = [row["axisTau"] for row in rows]
        axes[0].step(tau, [row["maxRefLev"] for row in rows], where="post", color=colors[case], label=case.upper())
        axes[1].step(tau, [row["nmb_total"] for row in rows], where="post", color=colors[case], label=case.upper())
    axes[0].set_ylabel("maximum relative AMR level"); axes[1].set_ylabel("MeshBlocks")
    for ax in axes: ax.set_xlabel("central proper time"); ax.grid(alpha=.22); ax.legend()
    save_figure(fig, output / "amr_vs_tau.png")


def plot_constraint_order(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for field in CONSTRAINTS:
        selected = [row for row in rows if row["field"] == field and math.isfinite(row["p"])]
        ax.plot([row["axisTau"] for row in selected], [row["p"] for row in selected], label=field)
    ax.axhline(4, color="black", linestyle="--", linewidth=1, label="ideal O4")
    ax.set_xlabel("central proper time"); ax.set_ylabel("effective order p")
    ax.grid(alpha=.22); ax.legend()
    save_figure(fig, path)


def plot_shadow(shadows: dict[str, list[dict[str, Any]]], path: Path) -> None:
    classes = sorted({row["classification"] for rows in shadows.values() for row in rows})
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), constrained_layout=True, sharex=True)
    for ax, case in zip(axes, ("n128", "n512")):
        event_rows = [row for row in shadows[case] if row.get("authority_event")]
        events = sorted({int(row["authority_event_index"]) for row in event_rows})
        bottom = np.zeros(len(events))
        for cls in classes:
            values = np.array([
                sum(row["classification"] == cls and int(row["authority_event_index"]) == event for row in event_rows)
                for event in events
            ])
            ax.bar(events, values, bottom=bottom, label=cls)
            bottom += values
        ax.set_ylabel(f"{case.upper()} block requests"); ax.grid(alpha=.16, axis="y")
    axes[-1].set_xlabel("authority event")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    save_figure(fig, path)


def plot_shadow_sensor(shadows: dict[str, list[dict[str, Any]]], path: Path) -> None:
    colors = {"n128": "#377eb8", "n512": "#e41a1c"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    for case, rows in shadows.items():
        by_cycle: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_cycle.setdefault(int(row["cycle"]), []).append(row)
        groups = [by_cycle[cycle] for cycle in sorted(by_cycle)]
        tau = [float(group[0]["tau_c"]) for group in groups]
        raw = [max(float(row["raw_dchi"]) for row in group) for group in groups]
        scaled = [max(float(row["dchi_over_dx"]) for row in group) for group in groups]
        axes[0].semilogy(tau, raw, color=colors[case], label=case.upper(), linewidth=.9)
        axes[1].semilogy(tau, scaled, color=colors[case], label=case.upper(), linewidth=.9)
    axes[0].axhline(.01, color="black", linestyle="--", linewidth=.8,
                    label=r"$d\chi_{max}=0.01$")
    axes[0].set_ylabel("maximum native raw dchi")
    axes[1].set_ylabel("maximum native dchi/dx")
    for ax in axes:
        ax.set_xlabel("central proper time"); ax.grid(alpha=.2, which="both"); ax.legend()
    save_figure(fig, path)


def plot_authority_jump_convergence(rows: list[dict[str, Any]], path: Path) -> None:
    colors = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
    positions = {"n128": 0, "n256": 1, "n512": 2}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)
    for ax, field in zip(axes.flat, CONSTRAINTS):
        for case in CASES:
            values = [float(row["abs_log10_jump"]) for row in rows
                      if row["field"] == field and row["resolution"] == case
                      and math.isfinite(float(row["abs_log10_jump"]))]
            if values:
                ax.boxplot(values, positions=[positions[case]], widths=.55,
                           patch_artist=True,
                           boxprops={"facecolor": colors[case], "alpha": .55},
                           medianprops={"color": "black"}, showfliers=False)
        ax.set_xticks((0, 1, 2), ("N128\nh=0.25", "N256\nh=0.125", "N512\nh=0.0625"))
        ax.set_ylabel(r"$|\Delta\log_{10}\|C\|_2|$")
        ax.set_title(field); ax.grid(alpha=.2, axis="y")
    fig.suptitle("Constraint jumps bracketing corresponding authority events")
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--n128", type=Path, required=True)
    parser.add_argument("--n256", type=Path, required=True)
    parser.add_argument("--n512", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-tau-max", type=float, required=True)
    args = parser.parse_args()

    output = args.output
    data = output / "data"
    figures = output / "figures"
    histories: dict[str, list[dict[str, float]]] = {}
    inputs: dict[str, Any] = {"authority": {"path": str(args.authority), "sha256": sha256(args.authority)}}
    roots = {case: getattr(args, case) for case in CASES}
    for case in CASES:
        rows, fields, provenance = merge_histories(roots[case], case)
        histories[case] = rows
        inputs[f"{case}_histories"] = provenance
        write_csv(data / f"{case}_history.csv", fields + ["segment"], rows)

    authority = json_lines(args.authority)
    replay_ledgers = {}
    for case in ("n128", "n512"):
        rows, provenance = merge_replay_ledgers(roots[case], case)
        replay_ledgers[case] = rows
        inputs[f"{case}_replay_ledgers"] = provenance
    replay_rows, replay_summary = verify_replay(authority, replay_ledgers)
    replay_fields = list(replay_rows[0])
    write_csv(data / "replay_tree_verification.csv", replay_fields, replay_rows)

    shadows = {}
    shadow_summary = {}
    for case in ("n128", "n512"):
        rows, provenance = merge_shadow(roots[case], case)
        shadows[case] = rows
        inputs[f"{case}_shadow_ledgers"] = provenance
        reduced = summarize_shadow_cycles(rows)
        write_csv(data / f"native_amr_shadow_{case}.csv", list(reduced[0]), reduced)
        shadow_summary[case] = {
            "records": len(rows),
            "classification_counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
            "native_action_counts": dict(sorted(Counter(row["native_action"] for row in rows).items())),
            "max_raw_dchi": max(row["raw_dchi"] for row in rows),
            "max_dchi_over_dx": max(row["dchi_over_dx"] for row in rows),
        }
    (data / "native_authority_mismatch_summary.json").write_text(
        json.dumps(shadow_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    conv = convergence_rows(histories, args.trusted_tau_max)
    write_csv(data / "constraint_convergence.csv", list(conv[0]), conv)
    jumps = authority_jump_rows(authority, histories)
    write_csv(data / "authority_event_jump_comparison.csv", list(jumps[0]) if jumps else [], jumps)
    landmarks = failure_landmarks(histories)
    write_csv(data / "failure_landmarks.csv", list(landmarks[0]), landmarks)

    plot_constraints(histories, "time", figures / "constraints_vs_t.png")
    plot_constraints(histories, "axisTau", figures / "constraints_vs_tau.png")
    plot_scalar_panels(histories, figures)
    plot_constraint_order(conv, figures / "constraint_convergence_order.png")
    plot_shadow(shadows, figures / "native_amr_shadow.png")
    plot_shadow_sensor(shadows, figures / "native_amr_sensor_vs_tau.png")
    plot_authority_jump_convergence(jumps, figures / "authority_event_jump_convergence.png")

    summary = {
        "schema": "brill_o4_common_tree_history_analysis_v1",
        "trusted_tau_max": args.trusted_tau_max,
        "inputs": inputs,
        "replay": replay_summary,
        "terminal_history_rows": {case: rows[-1] for case, rows in histories.items()},
        "native_amr_shadow": shadow_summary,
        "qualification_boundary": {
            "field_convergence_included": False,
            "figure3_overlay_included": False,
            "reason": "field binary sampling and published-curve overlay are separate gates",
        },
    }
    (output / "comparison_summary.history_only.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
