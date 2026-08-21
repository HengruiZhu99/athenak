#!/usr/bin/env python3
"""Build the authenticated O4/O6 Brill comparison from retained histories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / "evidence"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
INVESTIGATIONS = ROOT.parent
O6_HISTORY = (
    INVESTIGATIONS
    / "brill_r16_resolution_isolation_20260815"
    / "data/n256_history.csv"
)
PAPER_CURVES = (
    INVESTIGATIONS
    / "brill_r16_resolution_isolation_20260815"
    / "data/figure3_published_curves.csv"
)
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
FIELDS = (
    "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
    "max_abs_K", "nmb_total", "maxAbsKret", "maxRefLev", "cycle",
    "axisLapse", "axisTau", "axisKret",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_athena_history(path: Path, segment: int) -> list[dict[str, float]]:
    labels: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            for index, name in HEADER.findall(line):
                labels[name] = int(index) - 1
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"nonfinite history row in {path}")
        row = {name: values[index] for name, index in labels.items()}
        row["segment"] = float(segment)
        rows.append(row)
    missing = set(FIELDS) - labels.keys()
    if not rows or missing:
        raise RuntimeError(f"invalid history {path}; missing={sorted(missing)}")
    return rows


def read_o4() -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    histories = sorted(EVIDENCE.glob("segment*/*.z4c.user.hst"))
    if not histories:
        raise RuntimeError("no O4 history files")
    by_cycle: dict[int, dict[str, float]] = {}
    inputs: list[dict[str, object]] = []
    for path in histories:
        segment = int(path.parent.name.removeprefix("segment"))
        rows = read_athena_history(path, segment)
        inputs.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "segment": segment,
            "rows": len(rows),
        })
        for row in rows:
            cycle = int(row["cycle"])
            previous = by_cycle.get(cycle)
            if previous is not None:
                for name in FIELDS:
                    if previous[name] != row[name]:
                        raise RuntimeError(
                            f"restart overlap mismatch at cycle {cycle}, {name}: "
                            f"{previous[name]} != {row[name]}"
                        )
                continue
            by_cycle[cycle] = row
    rows = [by_cycle[cycle] for cycle in sorted(by_cycle)]
    if any(b["time"] <= a["time"] for a, b in zip(rows, rows[1:])):
        raise RuntimeError("O4 combined history time is not strictly increasing")
    return rows, inputs


def read_csv_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows or not set(FIELDS) <= rows[0].keys():
        raise RuntimeError(f"invalid CSV history: {path}")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220)
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def positive_xy(rows, x, y):
    pairs = [(row[x], row[y]) for row in rows if row[y] > 0.0]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def plot_figure3(curves) -> None:
    with PAPER_CURVES.open(newline="", encoding="utf-8") as stream:
        paper = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.7), constrained_layout=True)
    paper_styles = {
        "bamps": ("BAMPS (rendered paper curve)", "#999999", ":"),
        "prague": ("Prague (rendered paper curve)", "#666666", "--"),
        "sphGR": ("sphGR (rendered paper curve)", "#222222", "-."),
    }
    run_styles = {
        "O4": (r"O4, $d\chi_{\max}=0.02$", "#377eb8", "-"),
        "O6": (r"O6, $d\chi_{\max}=0.01$", "#e41a1c", "--"),
    }
    for ax in axes:
        for key, (label, color, linestyle) in paper_styles.items():
            selected = [row for row in paper if row["series"] == key]
            ax.plot(
                [float(row["tau"]) for row in selected],
                [float(row["log10_abs_I"]) for row in selected],
                label=label, color=color, linestyle=linestyle, linewidth=1.1,
            )
        for key, rows in curves.items():
            label, color, linestyle = run_styles[key]
            points = [
                (row["axisTau"], math.log10(abs(row["axisKret"])))
                for row in rows if row["axisKret"] != 0.0
            ]
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                label=label, color=color, linestyle=linestyle, linewidth=1.45,
            )
            ax.scatter(*points[-1], color=color, marker="x", s=40, zorder=5)
        ax.set_xlim(0.0, 15.05)
        ax.set_xlabel(r"central proper time $\tau/M$")
        ax.set_ylabel(r"$\log_{10}|I|$ at the origin")
        ax.grid(alpha=0.23)
    axes[0].set_ylim(-7.0, 6.0)
    axes[0].set_title("Published Figure 3 range")
    axes[1].set_ylim(-7.0, 34.0)
    axes[1].set_title("Full finite AthenaK range")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Brill A=-0.047: O4 and closest O6 history over Figure 3")
    save(fig, "fig3_reproduction_o4_o6_overlay")


def plot_constraints(curves) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
    specs = (
        ("C-norm2", r"$\|C\|_2$"),
        ("H-norm2", r"$\|H\|_2$"),
        ("M-norm2", r"$\|M\|_2$"),
        ("Z-norm2", r"$\|Z\|_2$"),
    )
    styles = {
        "O4": (r"O4, $d\chi_{\max}=0.02$", "#377eb8", "-"),
        "O6": (r"O6, $d\chi_{\max}=0.01$", "#e41a1c", "--"),
    }
    for ax, (field, title) in zip(axes.flat, specs):
        for key, rows in curves.items():
            xx, yy = positive_xy(rows, "time", field)
            label, color, linestyle = styles[key]
            ax.semilogy(xx, yy, label=label, color=color, linestyle=linestyle)
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.set_title(title)
        ax.grid(alpha=0.23, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Global constraints; contracts differ in AMR threshold")
    save(fig, "constraints_o4_o6")


def plot_gauge_amr(curves) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    specs = (
        ("axisLapse", "axis lapse", False),
        ("max_abs_K", r"$\max|K|$", True),
        ("maxAbsKret", r"$\max|I|$", True),
        ("maxRefLev", "maximum AMR level", False),
        ("nmb_total", "MeshBlock count", True),
        ("dt", r"time step $\Delta t/M$", True),
    )
    styles = {
        "O4": (r"O4, $d\chi_{\max}=0.02$", "#377eb8", "-"),
        "O6": (r"O6, $d\chi_{\max}=0.01$", "#e41a1c", "--"),
    }
    for ax, (field, title, log_y) in zip(axes.flat, specs):
        for key, rows in curves.items():
            label, color, linestyle = styles[key]
            xx, yy = positive_xy(rows, "time", field) if log_y else (
                [row["time"] for row in rows], [row[field] for row in rows]
            )
            (ax.semilogy if log_y else ax.plot)(
                xx, yy, label=label, color=color, linestyle=linestyle,
            )
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.set_title(title)
        ax.grid(alpha=0.23, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Gauge, curvature, hierarchy, and timestep")
    save(fig, "gauge_curvature_amr_dt_o4_o6")


def plot_jump_topology(jumps) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6), constrained_layout=True)
    specs = (
        ("C_abs_log10_jump", "C"),
        ("H_abs_log10_jump", "H"),
        ("M_abs_log10_jump", "M"),
        ("Z_abs_log10_jump", "Z"),
    )
    ordered = sorted(jumps, key=lambda row: row["time_after"])
    for ax, (field, label) in zip(axes.flat, specs):
        unchanged = [row for row in ordered if not row["topology_changed"]]
        changed = [row for row in ordered if row["topology_changed"]]
        ax.scatter(
            [row["time_after"] for row in unchanged],
            [row[field] for row in unchanged],
            s=7, alpha=0.32, color="#999999", label="same hierarchy",
        )
        ax.scatter(
            [row["time_after"] for row in changed],
            [row[field] for row in changed],
            s=13, alpha=0.72, color="#984ea3", label="hierarchy changed",
        )
        ax.set_title(f"{label}: consecutive |log10 ratio|")
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.set_ylabel("absolute log10 jump")
        ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("O4 history constraint jumps versus recorded topology changes")
    save(fig, "constraint_jumps_vs_topology_o4")


def landmark(rows, field, threshold, direction):
    for row in rows:
        value = row[field]
        if (direction == "ge" and value >= threshold) or (
            direction == "le" and value <= threshold
        ):
            return row["time"]
    return None


def interpolate(rows, coordinate: str, target: float, field: str) -> float | None:
    if target < rows[0][coordinate] or target > rows[-1][coordinate]:
        return None
    for left, right in zip(rows, rows[1:]):
        if left[coordinate] <= target <= right[coordinate]:
            if right[coordinate] == left[coordinate]:
                return left[field]
            fraction = (
                (target - left[coordinate])
                / (right[coordinate] - left[coordinate])
            )
            return left[field] + fraction * (right[field] - left[field])
    return rows[-1][field] if target == rows[-1][coordinate] else None


def common_samples(curves, coordinate: str, targets: tuple[float, ...]):
    samples = []
    for target in targets:
        if any(target > rows[-1][coordinate] for rows in curves.values()):
            continue
        row = {coordinate: target}
        for scheme, rows in curves.items():
            for field in ("C-norm2", "H-norm2", "M-norm2", "Z-norm2"):
                row[f"{scheme}_{field}"] = interpolate(rows, coordinate, target, field)
        samples.append(row)
    return samples


def constraint_jumps(rows) -> list[dict[str, object]]:
    events = []
    for before, after in zip(rows, rows[1:]):
        jumps = {}
        for field in ("C-norm2", "H-norm2", "M-norm2", "Z-norm2"):
            if before[field] > 0.0 and after[field] > 0.0:
                jumps[field] = abs(math.log10(after[field] / before[field]))
            else:
                jumps[field] = 0.0
        events.append({
            "cycle_before": int(before["cycle"]),
            "cycle_after": int(after["cycle"]),
            "time_before": before["time"],
            "time_after": after["time"],
            "max_abs_log10_constraint_jump": max(jumps.values()),
            "C_abs_log10_jump": jumps["C-norm2"],
            "H_abs_log10_jump": jumps["H-norm2"],
            "M_abs_log10_jump": jumps["M-norm2"],
            "Z_abs_log10_jump": jumps["Z-norm2"],
            "delta_meshblocks": int(after["nmb_total"] - before["nmb_total"]),
            "delta_max_level": int(after["maxRefLev"] - before["maxRefLev"]),
            "topology_changed": bool(
                after["nmb_total"] != before["nmb_total"]
                or after["maxRefLev"] != before["maxRefLev"]
            ),
        })
    return sorted(
        events, key=lambda row: row["max_abs_log10_constraint_jump"], reverse=True
    )


def main() -> None:
    o4, o4_inputs = read_o4()
    o6 = read_csv_history(O6_HISTORY)
    curves = {"O4": o4, "O6": o6}

    write_csv(DATA / "o4_history_combined.csv", list(o4[0]), o4)
    plotted = []
    for scheme, rows in curves.items():
        for row in rows:
            plotted.append({"scheme": scheme, **{name: row[name] for name in FIELDS}})
    write_csv(DATA / "o4_o6_plotted_history.csv", ["scheme", *FIELDS], plotted)
    jumps = constraint_jumps(o4)
    write_csv(DATA / "o4_constraint_jump_events.csv", list(jumps[0]), jumps)

    plot_figure3(curves)
    plot_constraints(curves)
    plot_gauge_amr(curves)
    plot_jump_topology(jumps)

    summary = {
        "schema": "athenak_brill_o4_o6_comparison_v1",
        "qualification_claim": False,
        "formal_disposition": "COMPARISON_NOT_MATCHED",
        "formal_reason": (
            "O4 uses dchi_max=0.02 while the closest authenticated O6 N256 "
            "history uses dchi_max=0.01; this is not an isolated order comparison"
        ),
        "inputs": {
            "O4_segments": o4_inputs,
            "O6_history": {
                "path": str(O6_HISTORY.relative_to(INVESTIGATIONS)),
                "sha256": sha256(O6_HISTORY),
                "spatial_order": 6,
                "dchi_max": 0.01,
            },
            "paper_curves": {
                "path": str(PAPER_CURVES.relative_to(INVESTIGATIONS)),
                "sha256": sha256(PAPER_CURVES),
                "status": "rendered PDF polyline reconstruction, not raw paper data",
            },
        },
        "terminal": {
            key: {
                "time": rows[-1]["time"],
                "axis_proper_time": rows[-1]["axisTau"],
                "cycle": int(rows[-1]["cycle"]),
                "constraint_C": rows[-1]["C-norm2"],
                "dt": rows[-1]["dt"],
                "max_abs_K": rows[-1]["max_abs_K"],
                "max_kretschmann": rows[-1]["maxAbsKret"],
                "max_refinement_level": int(max(r["maxRefLev"] for r in rows)),
                "max_meshblocks": int(max(r["nmb_total"] for r in rows)),
                "topology_change_rows": sum(
                    1 for a, b in zip(rows, rows[1:])
                    if a["nmb_total"] != b["nmb_total"]
                    or a["maxRefLev"] != b["maxRefLev"]
                ),
                "minimum_dt": min(r["dt"] for r in rows if r["dt"] > 0.0),
                "maximum_constraints": {
                    field: max(r[field] for r in rows)
                    for field in ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
                },
            }
            for key, rows in curves.items()
        },
        "matched_coordinate_time_constraints": common_samples(
            curves, "time", (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
        ),
        "matched_axis_proper_time_constraints": common_samples(
            curves, "axisTau", (5.0, 10.0)
        ),
        "landmarks": {
            key: {
                "C_ge_1e6": landmark(rows, "C-norm2", 1.0e6, "ge"),
                "dt_le_1e-4": landmark(rows, "dt", 1.0e-4, "le"),
                "level_ge_10": landmark(rows, "maxRefLev", 10.0, "ge"),
                "level_ge_18": landmark(rows, "maxRefLev", 18.0, "ge"),
                "K_ge_1e3": landmark(rows, "max_abs_K", 1.0e3, "ge"),
            }
            for key, rows in curves.items()
        },
        "o4_constraint_jumps": {
            "event_count": len(jumps),
            "top_20_topology_changed_count": sum(
                bool(row["topology_changed"]) for row in jumps[:20]
            ),
            "top_event": jumps[0],
        },
        "test_evidence": {
            "zero_pde_fourier_max_error": {
                "O2": 1.2965737272678386,
                "O4": 3.4443389472700869,
                "O6": 3.7965698987568763,
            },
            "o4_raw_fixed_radius_derivative_order": 4.01794,
        },
        "limitations": [
            "No isolated O4/O6 production comparison because dchi_max differs.",
            "Published Figure 3 curves are reconstructed from rendered PDF vectors.",
            "No convergence or Figure 3 reproduction claim is made.",
            "Operational MeshBlock-capacity stops are not numerical-invariant failures.",
            (
                "The O4 state-admissibility scan detected an invalid state, but the "
                "SYCL host extraction of the selected noncontiguous cell failed before "
                "the checkpoint, component, coordinates, and reason could be recorded."
            ),
        ],
    }
    summary["terminal"]["O4"].update({
        "job_id": 8769961,
        "athena_exit": 134,
        "failure_mechanism": "STATE_INADMISSIBILITY_DETECTED_DETAIL_EXTRACTION_FAILED",
        "failure_evidence": {
            "first_guard": "Z4c state-admissibility scan selected an inadmissible cell",
            "detail_extraction": (
                "Kokkos SYCL rejected the noncontiguous 25-component device subview "
                "copy to Host before z4c_state_failure.json was written"
            ),
            "exact_state_reason_known": False,
            "exact_checkpoint_known": False,
            "exact_cell_known": False,
            "printed_chi_transfer_invalid_parent_groups": 0,
            "printed_chi_transfer_invalid_limited_groups": 0,
        },
    })
    summary["terminal"]["O6"].update({
        "failure_mechanism": "CENTRAL_AXIS_DIAGNOSTIC_SUPPORT_NONFINITE_OR_INVALID",
        "failure_source_status": "authenticated existing comparator evidence",
    })
    summary["segments"] = [
        {
            "segment": 0,
            "job_id": 8769918,
            "start": "fresh",
            "result": "OPERATIONAL_MESHBLOCK_CAPACITY_STOP",
            "max_nmb_per_rank": 4096,
            "proposed_meshblocks": 4100,
        },
        {
            "segment": 1,
            "job_id": 8769947,
            "start_restart_sha256": (
                "fb12aa1704204fa1b3b96346a2b064245d47676e4495ef78ff53a7ecf53c11f0"
            ),
            "result": "OPERATIONAL_MESHBLOCK_CAPACITY_STOP",
            "max_nmb_per_rank": 8192,
            "proposed_meshblocks": 8246,
        },
        {
            "segment": 2,
            "job_id": 8769961,
            "start_restart_sha256": (
                "340be0e5f851af226c7a39d00c5df27796e6c46f9a4d28552079e2513a1f5e75"
            ),
            "result": "STATE_INADMISSIBILITY_DETECTED_DETAIL_EXTRACTION_FAILED",
            "max_nmb_per_rank": 16384,
            "athena_exit": 134,
        },
    ]
    summary["diagnostic_measure"] = {
        "cartoon_history_cell_measure": "2*pi*rho*dx_rho*dx_z*sqrt(abs(det(gamma)))",
        "fictitious_collapsed_y_width": False,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
