#!/usr/bin/env python3
"""Build the matched A/B/C/B2 AMR-causality comparison from terminal evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = [
    ("A_dynamic", "A: dynamic/high-order"),
    ("B_frozen", "B: frozen/high-order"),
    ("C_buffered", "C: buffered frozen/high-order"),
    ("B2_limited", "B2: frozen/limited-O2"),
]
CONSTRAINTS = ("C", "H", "M", "Z")


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def reject(value: str) -> None:
        raise AnalysisError(f"nonfinite JSON token {value} in {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def history(path: Path) -> tuple[list[str], np.ndarray]:
    names = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#  [1]="):
            names = re.findall(r"\[\d+\]=([^ ]+)", line)
        elif line.strip() and not line.lstrip().startswith("#"):
            rows.append([float(value) for value in line.split()])
    if not names or not rows or any(len(row) != len(names) for row in rows):
        raise AnalysisError(f"invalid history {path}")
    return names, np.asarray(rows, dtype=np.float64)


def one_file(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise AnalysisError(f"expected one {pattern} under {root}, got {matches}")
    return matches[0]


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def target_jump(names: list[str], data: np.ndarray,
                pre_event_names: list[str], pre_event_row: np.ndarray,
                cycle: int = 1722) -> dict[str, Any]:
    idx = {name: i for i, name in enumerate(names)}
    pre_idx = {name: i for i, name in enumerate(pre_event_names)}
    cycles = np.rint(data[:, idx["cycle"]]).astype(int)
    after = np.where(cycles == cycle)[0]
    if len(after) != 1:
        raise AnalysisError("target history jump does not have one post-event row")
    if int(round(pre_event_row[pre_idx["cycle"]])) != cycle - 1:
        raise AnalysisError("pre-event history does not terminate at target cycle minus one")
    result = {}
    for family in CONSTRAINTS:
        a = float(pre_event_row[pre_idx[f"{family}-norm2"]])
        b = float(data[after[0], idx[f"{family}-norm2"]])
        result[family] = {
            "before": finite_or_none(a), "after": finite_or_none(b),
            "absolute_jump": finite_or_none(b-a),
            "log_abs_jump": finite_or_none(math.log(max(abs(b), 1e-300)) -
                                             math.log(max(abs(a), 1e-300)))
        }
    return result


def topology_cycle(path: Path) -> int:
    match = re.search(r"c(\d+)", path.stem)
    if not match:
        raise AnalysisError(f"cannot parse topology cycle from {path}")
    return int(match.group(1))


def point_segment_distance(x: float, y: float, segment: tuple[float, ...]) -> float:
    x1, y1, x2, y2 = segment
    dx, dy = x2-x1, y2-y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(x-x1, y-y1)
    t = max(0.0, min(1.0, ((x-x1)*dx + (y-y1)*dy)/(dx*dx+dy*dy)))
    return math.hypot(x-(x1+t*dx), y-(y1+t*dy))


def topology_geometry(rows: list[dict[str, str]]) -> tuple[list[dict[str, float]],
                                                            list[tuple[float, ...]]]:
    blocks = [{"level": int(row["level"]), "x1min": float(row["x1min"]),
               "x1max": float(row["x1max"]), "x2min": float(row["x2min"]),
               "x2max": float(row["x2max"])} for row in rows]
    interfaces: set[tuple[float, ...]] = set()
    tolerance = 1e-12
    for a_index, a in enumerate(blocks):
        for b in blocks[a_index+1:]:
            if a["level"] == b["level"]:
                continue
            for ax, bx in ((a["x1max"], b["x1min"]),
                           (b["x1max"], a["x1min"])):
                if abs(ax-bx) <= tolerance:
                    lo, hi = max(a["x2min"], b["x2min"]), min(a["x2max"], b["x2max"])
                    if hi-lo > tolerance:
                        interfaces.add((ax, lo, ax, hi))
            for ay, by in ((a["x2max"], b["x2min"]),
                           (b["x2max"], a["x2min"])):
                if abs(ay-by) <= tolerance:
                    lo, hi = max(a["x1min"], b["x1min"]), min(a["x1max"], b["x1max"])
                    if hi-lo > tolerance:
                        interfaces.add((lo, ay, hi, ay))
    return blocks, sorted(interfaces)


def location_distances(rho: float, z: float, geometry: tuple[list[dict[str, float]],
                                                              list[tuple[float, ...]]]) -> dict[str, float | None]:
    blocks, interfaces = geometry
    containing = [block for block in blocks
                  if block["x1min"]-1e-12 <= rho <= block["x1max"]+1e-12 and
                     block["x2min"]-1e-12 <= z <= block["x2max"]+1e-12]
    if not containing:
        raise AnalysisError(f"maximum location ({rho},{z}) lies outside topology")
    block = max(containing, key=lambda item: item["level"])
    edge = min(rho-block["x1min"], block["x1max"]-rho,
               z-block["x2min"], block["x2max"]-z)
    interface = min((point_segment_distance(rho, z, item) for item in interfaces),
                    default=math.inf)
    return {"meshblock_edge": max(0.0, edge),
            "coarse_fine_interface": interface if math.isfinite(interface) else None}


def case_summary(case_root: Path, name: str, label: str,
                 pre_event_names: list[str], pre_event_row: np.ndarray) -> tuple[dict[str, Any],
                                                                  list[str], np.ndarray,
                                                                  list[dict[str, Any]]]:
    history_path = one_file(case_root, "*.z4c.user.hst")
    names, data = history(history_path)
    idx = {column: i for i, column in enumerate(names)}
    result = strict_load(case_root / "result.json")
    diagnostic = one_file(case_root, "z4c_amr_jump_*") / "rank0000"
    exposure = read_jsonl(diagnostic / "rk_stage_exposure.jsonl")
    controls = read_jsonl(diagnostic / "hierarchy_control.jsonl")
    transactions = read_jsonl(diagnostic / "transactions.jsonl")
    topologies = sorted((diagnostic / "accepted_topologies").glob("*.csv"),
                        key=topology_cycle)
    if not topologies:
        raise AnalysisError(f"{name} has no accepted topology snapshots")
    change_cycles = {1722}
    for row in transactions:
        if int(row.get("nnew", 0)) or int(row.get("ndel", 0)):
            change_cycles.add(int(row["cycle"]))
    selected = [path for path in topologies if topology_cycle(path) in change_cycles]
    if not selected:
        selected = [topologies[0]]
    geometry_by_cycle = {topology_cycle(path): topology_geometry(read_csv(path))
                         for path in selected}
    available_cycles = sorted(geometry_by_cycle)

    max_distances: dict[str, dict[str, float | None]] = {}
    terminal_cycle = int(round(data[-1, idx["cycle"]]))
    geometry_cycle = max(cycle for cycle in available_cycles if cycle <= terminal_cycle)
    for family in CONSTRAINTS:
        rho = float(data[-1, idx[f"{family}-rho"]])
        z = float(data[-1, idx[f"{family}-z"]])
        max_distances[family] = location_distances(
            rho, z, geometry_by_cycle[geometry_cycle])
        max_distances[family].update({"rho": rho, "z": z,
                                      "value": finite_or_none(data[-1, idx[f"{family}-Linf"]])})

    actual_changes = [row for row in transactions
                      if int(row.get("nnew", 0)) or int(row.get("ndel", 0))]
    log = (case_root / "run.log").read_text(encoding="utf-8")
    fatal = [line.strip() for line in log.splitlines()
             if "FATAL ERROR" in line or "invalid_parent_stencils=" in line or
                "invalid_limited_groups=" in line]
    summary = {
        "case": name, "label": label,
        "hierarchy_control": result["hierarchy_control"],
        "amr_transfer": result["amr_transfer"],
        "exit_code": result["exit_code"], "reached_tlim": result["reached_tlim"],
        "terminal_time": result["terminal_time"], "terminal_cycle": terminal_cycle,
        "terminal_nmb": result["last_nmb"], "terminal_max_level": result["last_max_level"],
        "terminal_constraints": result["last_constraints"],
        "terminal_max_abs_K": result["last_max_abs_K"],
        "terminal_max_kretschmann": result["last_max_kretschmann"],
        "cumulative_X_CF": result["cumulative_X_CF"],
        "terminal_CF_face_incidents": result["last_coarse_fine_leaf_face_incidents"],
        "target_jump": target_jump(names, data, pre_event_names, pre_event_row),
        "actual_topology_change_count": len(actual_changes),
        "created_leaves": sum(int(row.get("nnew", 0)) for row in actual_changes),
        "deleted_leaves": sum(int(row.get("ndel", 0)) for row in actual_changes),
        "suppressed_refine": sum(int(row.get("suppressed_refine", 0)) for row in controls),
        "suppressed_derefine": sum(int(row.get("suppressed_derefine", 0)) for row in controls),
        "terminal_constraint_maximum_distances": max_distances,
        "fatal_lines": fatal,
        "history_sha256": sha256(history_path),
        "run_log_sha256": sha256(case_root / "run.log"),
        "qualification_claim": False,
    }
    return summary, names, data, exposure


def causal_verdict(cases: dict[str, dict[str, Any]]) -> tuple[str, str, list[str]]:
    a, b, c = (cases[name] for name in ("A_dynamic", "B_frozen", "C_buffered"))
    b2 = cases.get("B2_limited")
    evidence = []
    if not a["reached_tlim"] and b["reached_tlim"]:
        verdict, confidence = "repeated_regridding_is_causal", "high"
        evidence.append("A timed out in a refinement runaway while the same-event frozen hierarchy B reached 12.5 M")
    elif not b["reached_tlim"] and c["reached_tlim"]:
        verdict, confidence = "persistent_coarse_fine_interface_coupling_is_leading", "high"
        evidence.append("buffered frozen C survived while frozen B failed")
    elif not b["reached_tlim"] and not c["reached_tlim"]:
        delta = float(c["terminal_time"])-float(b["terminal_time"])
        if abs(delta) <= 0.15:
            verdict, confidence = "bulk_or_inherited_fine_representation_is_leading", "moderate"
            evidence.append(f"B and C failed within {abs(delta):.6g} M")
        elif delta > 0.15:
            verdict, confidence = "interface_proximity_materially_accelerates_failure", "moderate"
            evidence.append(f"buffering delayed failure by {delta:.6g} M but did not cure it")
        else:
            verdict, confidence = "buffering_did_not_stabilize_and_failed_earlier", "moderate"
            evidence.append(f"buffered C failed {-delta:.6g} M earlier than B")
        if b2:
            if b2["reached_tlim"]:
                verdict, confidence = "high_order_amr_transfer_is_causal", "high"
                evidence.append("limited-O2 B2 reached 12.5 M while high-order B failed")
            else:
                delay = float(b2["terminal_time"])-float(b["terminal_time"])
                evidence.append(f"limited-O2 changed survival time by {delay:.6g} M")
    elif a["reached_tlim"] and b["reached_tlim"] and c["reached_tlim"]:
        verdict, confidence = "known_failure_not_reproduced_by_12p5M", "high"
        evidence.append("all three required cases reached the requested terminal time")
    else:
        verdict, confidence = "inconclusive_mixed_survival", "low"
        evidence.append("the A/B/C survival pattern does not match a unique decision branch")
    return verdict, confidence, evidence


def analyze(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pre_event_history = one_file(root / "pre_event", "*.z4c.user.hst")
    pre_event_names, pre_event_data = history(pre_event_history)
    pre_idx = {name: i for i, name in enumerate(pre_event_names)}
    pre_rows = np.where(np.rint(pre_event_data[:, pre_idx["cycle"]]).astype(int) == 1721)[0]
    if len(pre_rows) < 1 or pre_rows[-1] != len(pre_event_data) - 1:
        raise AnalysisError("pre-event history must terminate at cycle 1721")
    if not np.allclose(pre_event_data[pre_rows],
                       np.repeat(pre_event_data[pre_rows[-1]][None, :],
                                 len(pre_rows), axis=0),
                       rtol=1.0e-13, atol=1.0e-15, equal_nan=True):
        raise AnalysisError("duplicate terminal cycle-1721 reductions disagree")
    pre_event_row = pre_event_data[pre_rows[-1]]
    summaries: dict[str, dict[str, Any]] = {}
    series: dict[str, tuple[list[str], np.ndarray, list[dict[str, Any]]]] = {}
    for name, label in CASES:
        case_root = root / "cases" / name
        if not case_root.is_dir():
            if name == "B2_limited" and (root / "B2-disposition.txt").read_text().strip().startswith("SKIPPED"):
                continue
            raise AnalysisError(f"missing required case {name}")
        summary, names, data, exposure = case_summary(
            case_root, name, label, pre_event_names, pre_event_row)
        summaries[name] = summary
        series[name] = (names, data, exposure)

    verdict, confidence, evidence = causal_verdict(summaries)
    table_rows = []
    for name, _ in CASES:
        if name not in summaries:
            continue
        row = summaries[name]
        table_rows.append({
            "case": name, "control": row["hierarchy_control"],
            "transfer": row["amr_transfer"], "exit": row["exit_code"],
            "reached_12p5M": row["reached_tlim"], "terminal_time": row["terminal_time"],
            "terminal_cycle": row["terminal_cycle"], "nmb": row["terminal_nmb"],
            "level": row["terminal_max_level"], "topology_changes": row["actual_topology_change_count"],
            "X_CF": row["cumulative_X_CF"], "CF_faces": row["terminal_CF_face_incidents"],
            "max_abs_K": row["terminal_max_abs_K"],
            "max_kretschmann": row["terminal_max_kretschmann"],
        })
    with (output / "comparison_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader(); writer.writerows(table_rows)

    colors = dict(zip(summaries, plt.cm.tab10.colors))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for family, axis in zip(CONSTRAINTS, axes.flat):
        for name, summary in summaries.items():
            names, data, _ = series[name]; idx = {column: i for i, column in enumerate(names)}
            axis.plot(data[:, idx["time"]], np.abs(data[:, idx[f"{family}-norm2"]]),
                      label=summary["label"], color=colors[name])
        axis.set_yscale("log"); axis.set_title(f"{family} constraint history")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("t/M"); axes[-1, 1].set_xlabel("t/M")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "constraint_histories.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, summary in summaries.items():
        names, data, _ = series[name]; idx = {column: i for i, column in enumerate(names)}
        axes[0].plot(data[:, idx["time"]], np.abs(data[:, idx["max_abs_K"]]),
                     label=summary["label"], color=colors[name])
        axes[1].plot(data[:, idx["time"]], np.abs(data[:, idx["maxAbsKret"]]),
                     label=summary["label"], color=colors[name])
    axes[0].set_ylabel("max |K|"); axes[1].set_ylabel("max Kretschmann")
    axes[1].set_xlabel("t/M")
    for axis in axes: axis.set_yscale("log"); axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "curvature_histories.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for name, summary in summaries.items():
        names, data, _ = series[name]; idx = {column: i for i, column in enumerate(names)}
        time = data[:, idx["time"]]
        axes[0].plot(time, data[:, idx["dt"]], label=summary["label"],
                     color=colors[name])
        axes[1].plot(time, data[:, idx["maxRefLev"]], color=colors[name])
        axes[2].plot(time, data[:, idx["nmb_total"]], color=colors[name])
        case_root = root / "cases" / name
        diagnostic = one_file(case_root, "z4c_amr_jump_*") / "rank0000"
        transactions = [row for row in read_jsonl(diagnostic / "transactions.jsonl")
                        if int(row.get("nnew", 0)) or int(row.get("ndel", 0))]
        if transactions:
            event_time = [float(row["time"]) for row in transactions]
            axes[3].step(event_time, np.arange(1, len(event_time) + 1), where="post",
                         color=colors[name])
    axes[0].set_yscale("log"); axes[0].set_ylabel(r"$\Delta t/M$")
    axes[1].set_ylabel("max refinement level")
    axes[2].set_ylabel("MeshBlocks")
    axes[3].set_ylabel("cumulative topology changes"); axes[3].set_xlabel("t/M")
    for axis in axes: axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "hierarchy_histories.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, summary in summaries.items():
        exposure = series[name][2]
        times = [row["time"] for row in exposure]
        axes[0].plot(times, [row["cumulative_X_CF"] for row in exposure],
                     label=summary["label"], color=colors[name])
        axes[1].plot(times, [row["coarse_fine_leaf_face_incidents"] for row in exposure],
                     label=summary["label"], color=colors[name])
    axes[0].set_ylabel(r"cumulative $X_{CF}$")
    axes[1].set_ylabel("CF face incidents / RK stage"); axes[1].set_xlabel("t/M")
    for axis in axes: axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "coarse_fine_exposure.png", dpi=180); plt.close(fig)

    result = {
        "schema": "athenak_brill_amr_abc_verdict_v1",
        "qualification_claim": False,
        "source_commit": args.source_commit,
        "restart_sha256": args.restart_sha256,
        "verdict": verdict, "confidence": confidence, "evidence": evidence,
        "cases": summaries,
        "limitations": [
            "Survival and diagnostic differences isolate numerical AMR mechanisms; they do not qualify Figure 3 or continuum convergence.",
            "A's step timeout and C's strict chi-parent failure are retained as distinct evidence; no positivity threshold, floor, gauge, damping, dissipation, or AMR threshold is altered."
        ]
    }
    strict_dump(output / "verdict.json", result)
    report = "# Brill N256 AMR hierarchy-causality result\n\n"
    report += f"Verdict: `{verdict}` (confidence: **{confidence}**).\n\n"
    for item in evidence:
        report += f"- {item}.\n"
    report += "\nThe parent-state audit is reported separately. No scientific qualification or production fix is claimed here.\n"
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    products = sorted(path for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in products:
            stream.write(f"{sha256(path)}  {path.name}\n")


def self_test() -> None:
    blocks = [{"level": 0, "x1min": 0.0, "x1max": 1.0, "x2min": 0.0, "x2max": 1.0},
              {"level": 1, "x1min": 1.0, "x1max": 1.5, "x2min": 0.0, "x2max": 0.5},
              {"level": 1, "x1min": 1.0, "x1max": 1.5, "x2min": 0.5, "x2max": 1.0}]
    geometry = topology_geometry([{key: str(value) for key, value in block.items()}
                                  for block in blocks])
    value = location_distances(0.5, 0.5, geometry)
    if not math.isclose(value["meshblock_edge"], 0.5) or \
       not math.isclose(value["coarse_fine_interface"], 0.5):
        raise AnalysisError("geometry self-test failed")
    print("campaign analyzer self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--restart-sha256")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if any(value is None for value in (
            args.run_root, args.output, args.source_commit, args.restart_sha256)):
        parser.error("run root, output, source commit, and restart hash are required")
    analyze(args)


if __name__ == "__main__":
    main()
