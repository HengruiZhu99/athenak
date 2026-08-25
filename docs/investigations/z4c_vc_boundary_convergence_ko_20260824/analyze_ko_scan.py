#!/usr/bin/env python3
"""Analyze the independent N256 Rout=16 native-AMR KO scan."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("C", "H", "M", "Z")
CUTOFFS = (4.0, 8.0, 12.0)
PALETTE = {
    "0.02": "#000000", "0.05": "#377eb8", "0.10": "#4daf4a",
    "0.20": "#984ea3", "0.50": "#e41a1c",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_radial_module(script: Path):
    spec = importlib.util.spec_from_file_location("boundary_ko_radial", script)
    require(spec is not None and spec.loader is not None, f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def combine_histories(roots: list[Path], radial) -> dict[str, np.ndarray]:
    by_time: dict[float, dict[str, float]] = {}
    labels: set[str] | None = None
    for root in roots:
        files = list(root.glob("*.z4c.user.hst"))
        require(len(files) == 1, f"{root}: expected one history")
        table = radial.read_history(files[0])
        current = set(table)
        labels = current if labels is None else labels & current
        for index, time in enumerate(table["time"]):
            by_time[float(time)] = {name: float(values[index]) for name, values in table.items()}
    require(labels is not None and by_time, "empty KO history")
    ordered = [by_time[key] for key in sorted(by_time)]
    return {name: np.asarray([row[name] for row in ordered], dtype=float)
            for name in sorted(labels)}


def authority(roots: list[Path]) -> tuple[list[dict], Path]:
    candidates: list[tuple[int, Path, list[dict]]] = []
    for root in roots:
        for path in root.glob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            events = [row for row in rows if row.get("type") == "event"]
            if events:
                candidates.append((len(events), path, events))
    require(bool(candidates), "record run lacks authority")
    _, path, events = max(candidates, key=lambda item: item[0])
    return events, path


def paired_binaries(roots: list[Path], radial) -> list[tuple[Path, Path]]:
    state: dict[float, Path] = {}
    constraints: dict[float, Path] = {}
    for root in roots:
        binary_root = root / "bin/rank_00000000"
        for path in binary_root.glob("*.state.*.bin"):
            state[round(radial.binary_header(path)[0], 13)] = path
        for path in binary_root.glob("*.constraints.*.bin"):
            constraints[round(radial.binary_header(path)[0], 13)] = path
    common = sorted(set(state) & set(constraints))
    return [(state[time], constraints[time]) for time in common]


def at_time(table: dict[str, np.ndarray], field: str, target: float) -> float | None:
    if target < table["time"][0] - 1.0e-12 or target > table["time"][-1] + 1.0e-12:
        return None
    return float(np.interp(target, table["time"], table[field]))


def first_event(events: list[dict], *, after: float = -math.inf,
                refine_only: bool = True) -> dict | None:
    for event in events[1:]:
        if float(event["time"]) <= after:
            continue
        if not refine_only or int(event.get("created", 0)) > 0:
            return event
    return None


def exit_status(roots: list[Path]) -> list[int | None]:
    result = []
    for root in roots:
        path = root / "athena-exit"
        result.append(int(path.read_text().strip()) if path.exists() else None)
    return result


def peak_hbm(roots: list[Path]) -> float | None:
    values = []
    for root in roots:
        path = root / "evidence/peak-hbm-used-mib.txt"
        if path.exists():
            try:
                values.append(float(path.read_text().strip()))
            except ValueError:
                pass
    return max(values) if values else None


def log_axis_deviation(table: dict[str, np.ndarray], baseline: dict[str, np.ndarray]) -> float | None:
    low = max(float(table["axisTau"][0]), float(baseline["axisTau"][0]))
    high = min(float(table["axisTau"][-1]), float(baseline["axisTau"][-1]))
    if high <= low:
        return None
    tau = np.linspace(low, high, 1001)
    candidate = np.interp(tau, table["axisTau"], table["axisKret"])
    reference = np.interp(tau, baseline["axisTau"], baseline["axisKret"])
    mask = (np.abs(candidate) > 1.0e-300) & (np.abs(reference) > 1.0e-300)
    if not np.any(mask):
        return None
    return float(np.max(np.abs(np.log10(np.abs(candidate[mask]))
                               - np.log10(np.abs(reference[mask])))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON object mapping diss strings to ordered segment roots")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected = ("0.02", "0.05", "0.10", "0.20", "0.50")
    require(set(config) == set(expected), f"KO config must contain {expected}")
    roots = {diss: [Path(value) for value in config[diss]] for diss in expected}
    require(all(paths for paths in roots.values()), "empty KO segment inventory")

    radial = load_radial_module(Path(__file__).with_name("analyze_radial_constraints.py"))
    reader = radial.load_reader(args.source)
    histories = {diss: combine_histories(paths, radial) for diss, paths in roots.items()}
    authorities: dict[str, list[dict]] = {}
    authority_paths: dict[str, str] = {}
    radial_rows: list[dict] = []
    health_rows: list[dict] = []
    integration_errors: list[dict] = []
    for diss, paths in roots.items():
        events, path = authority(paths)
        authorities[diss] = events; authority_paths[diss] = str(path.resolve())
        for state_path, constraint_path in paired_binaries(paths, radial):
            try:
                rows, health = radial.integrate_snapshot(
                    state_path, constraint_path, reader, CUTOFFS, (0.0, 4.0, 8.0, 12.0, 16.0))
            except Exception as error:  # preserve the last valid prefix of a failed run
                integration_errors.append({"diss": diss, "state": str(state_path),
                                           "constraint": str(constraint_path),
                                           "error": str(error)})
                continue
            axis_tau = float(np.interp(health["time"], histories[diss]["time"],
                                       histories[diss]["axisTau"]))
            health_rows.append({"diss": diss, "axisTau": axis_tau, **health})
            for row in rows:
                if row["region"] in {"R4", "R8", "R12", "full"}:
                    radial_rows.append({"diss": diss, "time": health["time"],
                                        "axisTau": axis_tau, **row})

    output = args.output
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ko_radial_constraints.csv", radial_rows)
    write_csv(output / "ko_state_health.csv", health_rows)
    if integration_errors:
        write_csv(output / "ko_integration_errors.csv", integration_errors)

    baseline = histories["0.02"]
    summary_rows: list[dict] = []
    for diss in expected:
        table = histories[diss]; events = authorities[diss]
        first = first_event(events)
        late = first_event(events, after=6.5)
        created_total = sum(int(event.get("created", 0)) for event in events[1:])
        requested_total = sum(int(event.get("requested_refine", 0)) for event in events[1:])
        case_health = [row for row in health_rows if row["diss"] == diss]
        statuses = exit_status(roots[diss])
        terminal_time = float(table["time"][-1])
        status = ("REACHED_T11P3" if terminal_time >= 11.3 - 1.0e-10
                  else "REACHED_T9P5" if terminal_time >= 9.5 - 1.0e-10
                  else "REACHED_T6P5" if terminal_time >= 6.5 - 1.0e-10
                  else "INCOMPLETE_OR_FAILED")
        if any(value not in (0, None) for value in statuses):
            status += "_NONZERO_EXIT"
        summary_rows.append({
            "diss": diss,
            "C_t6p5": at_time(table, "C-norm2", 6.5),
            "C_t9p2": at_time(table, "C-norm2", 9.2),
            "first_transaction_time": float(events[1]["time"]) if len(events) > 1 else None,
            "first_accepted_refinement_time": float(first["time"]) if first else None,
            "first_late_refinement_after_t6p5": float(late["time"]) if late else None,
            "requested_refine_total": requested_total,
            "created_leaf_total": created_total,
            "max_leaf_count": max(int(event["leaf_count"]) for event in events),
            "max_ref_level": float(np.max(table["maxRefLev"])),
            "terminal_time": terminal_time,
            "terminal_axisTau": float(table["axisTau"][-1]),
            "terminal_C": float(table["C-norm2"][-1]),
            "terminal_H": float(table["H-norm2"][-1]),
            "terminal_M": float(table["M-norm2"][-1]),
            "terminal_Z": float(table["Z-norm2"][-1]),
            "minimum_chi": min((row["minimum_chi"] for row in case_health), default=None),
            "minimum_metric_pivot": min((row["minimum_conformal_spd_pivot"]
                                          for row in case_health), default=None),
            "max_axisKret_log10_deviation_from_diss0p02": log_axis_deviation(table, baseline),
            "peak_hbm_used_mib": peak_hbm(roots[diss]),
            "segment_exit_statuses": ";".join("missing" if value is None else str(value)
                                               for value in statuses),
            "status": status,
        })
    write_csv(output / "ko_summary.csv", summary_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for axis, field in zip(axes.flat, FIELDS):
        for diss in expected:
            table = histories[diss]
            axis.semilogy(table["axisTau"], table[f"{field}-norm2"],
                          color=PALETTE[diss], label=f"diss={diss}")
        axis.set_title(field); axis.set_xlabel(r"central proper time $\tau_c/M$")
        axis.grid(alpha=0.24, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(r"N256 $R_{out}=16M$ KO scan: global constraints")
    fig.savefig(figures / "ko_global_constraints.png", dpi=240)
    fig.savefig(figures / "ko_global_constraints.pdf")
    plt.close(fig)

    for region in ("R4", "R8", "R12"):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
        for axis, field in zip(axes.flat, FIELDS):
            for diss in expected:
                selected = sorted((row for row in radial_rows if row["diss"] == diss
                                   and row["region"] == region and row["field"] == field),
                                  key=lambda row: row["axisTau"])
                if selected:
                    axis.semilogy([row["axisTau"] for row in selected],
                                  [row["norm2"] for row in selected], "o-",
                                  markersize=2.5, color=PALETTE[diss], label=f"diss={diss}")
            axis.set_title(field); axis.set_xlabel(r"central proper time $\tau_c/M$")
            axis.grid(alpha=0.24, which="both")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"N256 KO scan: constraints in {region}")
        fig.savefig(figures / f"ko_constraints_{region}.png", dpi=240)
        fig.savefig(figures / f"ko_constraints_{region}.pdf")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    if args.reference is not None:
        reference: dict[str, tuple[list[float], list[float]]] = {}
        with args.reference.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                values = reference.setdefault(row["series"], ([], []))
                values[0].append(float(row["tau"])); values[1].append(float(row["log10_abs_I"]))
        for name, (xx, yy) in reference.items():
            axis.plot(xx, yy, linewidth=1.0, alpha=0.45, label=f"paper {name}")
    for diss in expected:
        table = histories[diss]; mask = np.abs(table["axisKret"]) > 0.0
        axis.plot(table["axisTau"][mask], np.log10(np.abs(table["axisKret"][mask])),
                  color=PALETTE[diss], label=f"diss={diss}")
    axis.set_xlim(left=0.0); axis.set_ylim(-8.0, 8.0)
    axis.set_xlabel(r"central proper time $\tau_c/M$")
    axis.set_ylabel(r"$\log_{10}|I(0)|$")
    axis.grid(alpha=0.22); axis.legend(ncol=2, fontsize=8)
    fig.savefig(figures / "ko_axisKret_figure3_overlay.png", dpi=240)
    fig.savefig(figures / "ko_axisKret_figure3_overlay.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.bar([row["diss"] for row in summary_rows],
             [row["first_late_refinement_after_t6p5"]
              if row["first_late_refinement_after_t6p5"] is not None else np.nan
              for row in summary_rows],
             color=[PALETTE[row["diss"]] for row in summary_rows])
    axis.axhline(10.278668702889213, color="black", linestyle="--", linewidth=0.9,
                 label="authenticated diss=0.02 late refinement")
    axis.set_xlabel("z4c/diss"); axis.set_ylabel("first refinement time after t=6.5")
    axis.grid(alpha=0.24, axis="y"); axis.legend(fontsize=8)
    fig.savefig(figures / "ko_first_late_refinement.png", dpi=240)
    fig.savefig(figures / "ko_first_late_refinement.pdf")
    plt.close(fig)

    result = {
        "schema": "z4c_vc_n256_ko_scan_v1",
        "config": str(args.config.resolve()),
        "authority_paths": authority_paths,
        "summary": summary_rows,
        "integration_errors": integration_errors,
    }
    (output / "ko_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
