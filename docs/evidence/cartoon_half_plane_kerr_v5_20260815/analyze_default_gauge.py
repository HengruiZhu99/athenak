#!/usr/bin/env python3
"""Package the existing half-plane Kerr default-gauge evidence.

This script is intentionally read-only with respect to the numerical evidence.
It parses the frozen V5 histories/horizon rows and the bounded CPU restart
diagnostics, then writes CSV/JSON summaries and publication-quality figures.
It performs no AthenaK evolution and makes no convergence qualification claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "v5_evidence"
OUTPUT = ROOT / "analysis_default_gauge"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> bool:
    return math.isfinite(value)


def strict_dump(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_history(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    names: dict[int, str] = {}
    rows: list[dict[str, float]] = []
    header_pattern = re.compile(r"\[(\d+)\]=([^ ]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            for index, name in header_pattern.findall(line):
                names[int(index)] = name
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != len(names):
            raise ValueError(f"{path}: expected {len(names)} columns, got {len(values)}")
        rows.append({names[index + 1]: value for index, value in enumerate(values)})
    expected = set(range(1, len(names) + 1))
    if set(names) != expected or not rows:
        raise ValueError(f"{path}: invalid or empty history schema")
    return [names[index] for index in sorted(names)], rows


HORIZON_FIELDS = [
    "cycle",
    "time",
    "branch",
    "accepted",
    "center_z",
    "axis_extremum_z",
    "center_lapse",
    "area",
    "irreducible_mass",
    "horizon_mass",
    "spin_z",
    "mean_radius",
    "minimum_radius",
    "direct_residual",
    "flow_residual",
    "failure",
]


def parse_horizon(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < len(HORIZON_FIELDS):
            raise ValueError(f"{path}: short horizon row")
        row: dict[str, Any] = {
            "cycle": int(values[0]),
            "time": float(values[1]),
            "branch": values[2],
            "accepted": int(values[3]),
            "failure": values[15],
        }
        for index, field in enumerate(HORIZON_FIELDS[4:15], start=4):
            row[field] = float(values[index])
        row["coefficients"] = [float(value) for value in values[16:]]
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no horizon rows")
    return rows


def rms(numerator: float, denominator: float) -> float | None:
    if not finite(numerator) or not finite(denominator) or denominator <= 0 or numerator < 0:
        return None
    return math.sqrt(numerator / denominator)


def load_cycle_slices() -> dict[int, dict[str, Any]]:
    def science_payload(payload: dict[str, Any]) -> dict[str, Any]:
        canonical = {key: value for key, value in payload.items() if key != "sources"}
        canonical["rows"] = [
            {key: value for key, value in row.items() if key != "sources"}
            for row in payload["rows"]
        ]
        return canonical

    slices: dict[int, dict[str, Any]] = {}
    for path in sorted(ROOT.glob("cycle_near_axis_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cycle = int(payload["cycle"])
        if cycle in slices:
            canonical = json.dumps(science_payload(slices[cycle]), sort_keys=True, separators=(",", ":"))
            duplicate = json.dumps(science_payload(payload), sort_keys=True, separators=(",", ":"))
            if canonical != duplicate:
                raise ValueError(f"nonidentical duplicate cycle {cycle}")
            continue
        slices[cycle] = payload
    if set(slices) != set(range(2162, 2178)):
        raise ValueError(f"unexpected diagnostic cycles: {sorted(slices)}")
    return slices


EVEN_Z4C = {
    "z4c_chi", "z4c_gxx", "z4c_gxz", "z4c_gyy", "z4c_gzz",
    "z4c_Khat", "z4c_Axx", "z4c_Axz", "z4c_Ayy", "z4c_Azz",
    "z4c_Gamx", "z4c_Gamz", "z4c_Theta", "z4c_alpha",
    "z4c_betax", "z4c_betaz", "z4c_Bx", "z4c_Bz",
}
ODD_Z4C = {
    "z4c_gxy", "z4c_gyz", "z4c_Axy", "z4c_Ayz", "z4c_Gamy",
    "z4c_betay", "z4c_By",
}


def family_max(rows: list[dict[str, Any]], prefix: str) -> float:
    values: list[float] = []
    for row in rows:
        for name, value in row["z4c"].items():
            if name.startswith(prefix) and finite(float(value)):
                values.append(abs(float(value)))
    return max(values) if values else math.nan


def constraint_max(rows: list[dict[str, Any]], name: str) -> float:
    values = [abs(float(row["constraints"][name])) for row in rows]
    return max(values)


def parity_residual(rows: list[dict[str, Any]]) -> float:
    keyed = {(round(row["rho"], 14), round(row["z"], 14)): row for row in rows}
    numer = 0.0
    scale = 0.0
    used = 0
    for (rho, z), plus in keyed.items():
        if z <= 0:
            continue
        minus = keyed.get((rho, round(-z, 14)))
        if minus is None:
            continue
        for name, parity in [(name, 1.0) for name in EVEN_Z4C] + [
            (name, -1.0) for name in ODD_Z4C
        ]:
            pvalue = float(plus["z4c"][name])
            mvalue = float(minus["z4c"][name])
            numer = max(numer, abs(pvalue - parity * mvalue))
            scale = max(scale, abs(pvalue), abs(mvalue))
            used += 1
    if used == 0:
        raise ValueError("no mirror pairs")
    return numer / max(scale, 1.0e-300)


KV_PATTERN = re.compile(r"(\w+)=([^ ]+)")


def load_term_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = ROOT / "run_stage_split"
    for path in sorted(source.glob("z4c_rhs_stage_rank*.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("Z4C_RHS_TERM_DIAGNOSTIC"):
                continue
            raw = dict(KV_PATTERN.findall(line))
            record = {
                "rank": int(raw["rank"]),
                "cycle": int(raw["cycle"]),
                "time": float(raw["time"]),
                "stage": int(raw["stage"]),
                "term": raw["term"],
                "selected_cells": int(raw["selected_cells"]),
                "nonfinite": int(raw["nonfinite"]),
                "abs_max": float(raw["abs_max"]),
                "value": float(raw["value"]),
                "rho": float(raw["rho"]),
                "z": float(raw["z"]),
                "source": path.name,
            }
            records.append(record)
    terms = {record["term"] for record in records}
    if len(terms) != 75 or len(records) != 75 * 4 * 6 * 4:
        raise ValueError(f"unexpected term inventory: {len(terms)} terms, {len(records)} rows")
    return records


def term_category(term: str) -> str | None:
    ordered = [
        "A_trace_ricci", "A_ricci_tensor", "A_hessian", "A_geometric",
        "A_nonlinear", "A_lie", "Gamma_contraction", "Gamma_expansion",
        "Gamma_second", "Gamma_advective", "Gamma_ddiv", "Gamma_damping",
        "Gamma_lapse_gradient",
    ]
    for category in ordered:
        if term.startswith(category + "_"):
            return category
    return None


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        OUTPUT / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_constraint_history(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    colors = {"h32": "#1f77b4", "h48": "#d62728"}
    styles = {"C-norm2": "-", "H-norm2": "--", "M-norm2": ":", "Z-norm2": "-."}
    for resolution, rows in histories.items():
        times = [row["time"] for row in rows]
        for field, style in styles.items():
            values = [abs(row[field]) if finite(row[field]) and row[field] != 0 else math.nan for row in rows]
            axes[0].semilogy(times, values, style, color=colors[resolution], lw=1.2,
                            label=f"{resolution} |{field.replace('-norm2', '')}|" if field == "C-norm2" else None,
                            alpha=1.0 if field == "C-norm2" else 0.58)
        axis_values = [rms(row["ax-C2"], row["ax-N"]) or math.nan for row in rows]
        off_values = [rms(row["off-C2"], row["off-Vol"]) or math.nan for row in rows]
        axes[1].semilogy(times, axis_values, color=colors[resolution], lw=1.5,
                        label=f"{resolution} axis RMS(C)")
        axes[1].semilogy(times, off_values, color=colors[resolution], lw=1.3, ls="--",
                        label=f"{resolution} off-axis RMS(C)")
    axes[0].set_title("Reported global constraint diagnostics")
    axes[0].set_xlabel(r"$t/M$")
    axes[0].set_ylabel("absolute reported norm2")
    axes[0].grid(True, which="both", alpha=0.25)
    # Compact legend: the line styles are described in the caption/README.
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_title("Axis versus off-axis C diagnostic")
    axes[1].set_xlabel(r"$t/M$")
    axes[1].set_ylabel("reported RMS(C)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    save_figure(fig, "kerr_default_gauge_constraints")


def plot_horizons(horizons: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 3.9), constrained_layout=True)
    colors = {"h32": "#1f77b4", "h48": "#d62728"}
    for resolution, rows in horizons.items():
        origin = [row for row in rows if row["branch"] == "origin"]
        accepted = [row for row in origin if row["accepted"] == 1]
        rejected = [row for row in origin if row["accepted"] == 0]
        axes[0].plot([row["time"] for row in accepted], [row["horizon_mass"] for row in accepted],
                     color=colors[resolution], marker="o", ms=2.7, lw=1.2, label=resolution)
        axes[1].plot([row["time"] for row in accepted], [row["spin_z"] for row in accepted],
                     color=colors[resolution], marker="o", ms=2.7, lw=1.2, label=resolution)
        axes[2].semilogy([row["time"] for row in origin], [row["direct_residual"] for row in origin],
                        color=colors[resolution], lw=1.2, label=f"{resolution} direct")
        axes[2].semilogy([row["time"] for row in origin], [row["flow_residual"] for row in origin],
                        color=colors[resolution], lw=1.2, ls="--", label=f"{resolution} flow")
        axes[2].scatter([row["time"] for row in rejected], [row["direct_residual"] for row in rejected],
                        facecolors="none", edgecolors=colors[resolution], s=18, zorder=3)
    axes[0].axhline(1.0, color="0.35", ls=":", lw=1)
    axes[1].axhline(0.5, color="0.35", ls=":", lw=1)
    axes[0].set_title("Accepted origin horizon mass")
    axes[1].set_title("Accepted origin horizon spin")
    axes[2].set_title("Origin finder residuals")
    for axis in axes:
        axis.set_xlabel(r"$t/M$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"$M_{\rm AH}/M$")
    axes[1].set_ylabel(r"$S_z/M_{\rm AH}^2$")
    axes[2].set_ylabel("residual")
    save_figure(fig, "kerr_default_gauge_horizons")


def plot_near_axis_growth(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    cycles = [row["cycle"] for row in rows]
    axes[0].semilogy(cycles, [row["max_abs_C"] for row in rows], "o-", label=r"max $|C|$")
    axes[0].semilogy(cycles, [row["max_abs_A"] for row in rows], "s-", label=r"max $|\tilde A_{ij}|$")
    axes[0].semilogy(cycles, [row["max_abs_Gamma"] for row in rows], "^-", label=r"max $|\tilde\Gamma^i|$")
    axes[0].set_title("Near-axis puncture-interior growth")
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("maximum absolute value")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].semilogy(cycles, [max(row["parity_residual"], 1.0e-18) for row in rows], "o-",
                    label="equatorial parity residual")
    axes[1].semilogy(cycles, [max(row["max_abs_detg_minus_one"], 1.0e-18) for row in rows], "s-",
                    label=r"max $|\det\tilde\gamma-1|$")
    axes[1].semilogy(cycles, [max(row["max_abs_traceA"], 1.0e-18) for row in rows], "^-",
                    label=r"max $|\tilde\gamma^{ij}\tilde A_{ij}|$")
    axes[1].set_title("Symmetry and algebraic constraints")
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("relative or absolute residual")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(fig, "kerr_default_gauge_near_axis_growth")


def plot_near_axis_slices(slices: dict[int, dict[str, Any]]) -> None:
    selected_cycles = [2168, 2172, 2177]
    fields = [
        ("constraints", "con_C", r"$|C|$"),
        ("z4c_A", None, r"$\max|\tilde A_{ij}|$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True, sharey=True,
                             constrained_layout=True)
    all_values: dict[str, list[float]] = {"constraints": [], "z4c_A": []}
    values_by_panel: dict[tuple[str, int], list[float]] = {}
    for cycle in selected_cycles:
        rows = slices[cycle]["rows"]
        con_values = [max(abs(float(row["constraints"]["con_C"])), 1.0e-18) for row in rows]
        a_values = [max(abs(float(value)) for name, value in row["z4c"].items()
                        if name.startswith("z4c_A")) for row in rows]
        values_by_panel[("constraints", cycle)] = con_values
        values_by_panel[("z4c_A", cycle)] = a_values
        all_values["constraints"].extend(con_values)
        all_values["z4c_A"].extend(a_values)
    norms = {
        key: LogNorm(vmin=max(min(values), 1.0e-12), vmax=max(values))
        for key, values in all_values.items()
    }
    for column, cycle in enumerate(selected_cycles):
        rows = slices[cycle]["rows"]
        for row_index, (kind, _, label) in enumerate(fields):
            axis = axes[row_index, column]
            patches = [Rectangle((row["rho"] - row["h_rho"] / 2,
                                  row["z"] - row["h_z"] / 2),
                                 row["h_rho"], row["h_z"]) for row in rows]
            collection = PatchCollection(patches, cmap="magma", norm=norms[kind],
                                         edgecolor="none")
            collection.set_array(values_by_panel[(kind, cycle)])
            axis.add_collection(collection)
            axis.set_xlim(0.0, 0.105)
            axis.set_ylim(-0.5, 0.5)
            axis.set_aspect("equal")
            axis.set_title(f"{label}, cycle {cycle}" if row_index == 0 else label)
            axis.set_xlabel(r"$\rho/M$")
            if column == 0:
                axis.set_ylabel(r"$z/M$")
            fig.colorbar(collection, ax=axis, fraction=0.046, pad=0.03)
    save_figure(fig, "kerr_default_gauge_near_axis_slices")


def plot_rhs_terms(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    a_categories = ["A_trace_ricci", "A_ricci_tensor", "A_hessian", "A_nonlinear", "A_lie"]
    g_categories = ["Gamma_contraction", "Gamma_expansion", "Gamma_second",
                    "Gamma_advective", "Gamma_damping", "Gamma_lapse_gradient"]
    for category in a_categories:
        selected = [row for row in rows if row["category"] == category]
        axes[0].semilogy([row["cycle"] for row in selected], [row["abs_max"] for row in selected],
                        marker="o", ms=3, lw=1.2, label=category.replace("_", " "))
    for category in g_categories:
        selected = [row for row in rows if row["category"] == category]
        axes[1].semilogy([row["cycle"] for row in selected], [row["abs_max"] for row in selected],
                        marker="o", ms=3, lw=1.2, label=category.replace("_", " "))
    axes[0].set_title(r"Dominant $\tilde A_{ij}$ RHS pieces")
    axes[1].set_title(r"Dominant $\tilde\Gamma^i$ RHS pieces")
    for axis in axes:
        axis.set_xlabel("cycle")
        axis.set_ylabel("maximum absolute stage contribution")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7.5)
    save_figure(fig, "kerr_default_gauge_rhs_terms")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    state_path = EVIDENCE / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    physics = state["contract"]["physics"]
    gauge = physics["gauge"]
    if not (
        physics["precollapsed_lapse"] is True
        and gauge["name"] == "athenak_default_moving_puncture"
        and gauge["lapse_oplog"] == 2.0
        and gauge["shift_Gamma"] == 1.0
        and gauge["shift_eta"] == 2.0
        and gauge["telegraph_lapse"] is False
        and gauge["sss_damping_amp"] == 0.0
        and physics["dchi_max"] == 0.02
    ):
        raise ValueError("campaign does not match the requested default-gauge contract")

    histories: dict[str, list[dict[str, float]]] = {}
    horizons: dict[str, list[dict[str, Any]]] = {}
    source_paths: list[Path] = [state_path]
    history_csv_rows: list[dict[str, Any]] = []
    horizon_csv_rows: list[dict[str, Any]] = []
    resolution_summary: dict[str, Any] = {}
    for resolution in ("h32", "h48"):
        history_path = EVIDENCE / f"kerr_half_plane_{resolution}_moving_puncture.z4c.user.hst"
        horizon_path = EVIDENCE / f"kerr_half_plane_{resolution}_moving_puncture.cartoon_m0_horizon_0.txt"
        _, history_rows = parse_history(history_path)
        horizon_rows = parse_horizon(horizon_path)
        histories[resolution] = history_rows
        horizons[resolution] = horizon_rows
        source_paths.extend([history_path, horizon_path])
        for row in history_rows:
            history_csv_rows.append({
                "resolution": resolution,
                "time": row["time"],
                "cycle": int(row["cycle"]),
                "C_norm2": row["C-norm2"],
                "H_norm2": row["H-norm2"],
                "M_norm2": row["M-norm2"],
                "Z_norm2": row["Z-norm2"],
                "C_Linf": row["C-Linf"],
                "axis_C_rms": rms(row["ax-C2"], row["ax-N"]),
                "off_axis_C_rms": rms(row["off-C2"], row["off-Vol"]),
                "axis_lapse": row["axisLapse"],
                "horizon_status": int(row["ahStatus"]),
            })
        for row in horizon_rows:
            horizon_csv_rows.append({"resolution": resolution, **{field: row[field] for field in HORIZON_FIELDS}})
        origin = [row for row in horizon_rows if row["branch"] == "origin"]
        accepted = [row for row in origin if row["accepted"] == 1]
        resolution_summary[resolution] = {
            "history_rows": len(history_rows),
            "terminal_time": history_rows[-1]["time"],
            "terminal_cycle": int(history_rows[-1]["cycle"]),
            "terminal_C_norm2": history_rows[-1]["C-norm2"] if finite(history_rows[-1]["C-norm2"]) else None,
            "terminal_C_Linf": history_rows[-1]["C-Linf"] if finite(history_rows[-1]["C-Linf"]) else None,
            "origin_horizon": {
                "first_accepted": accepted[0],
                "last_accepted": accepted[-1],
                "last_attempt": origin[-1],
            },
        }

    cycle_slices = load_cycle_slices()
    source_paths.extend(sorted(ROOT.glob("cycle_near_axis_*.json")))
    near_axis_rows: list[dict[str, Any]] = []
    for cycle, payload in sorted(cycle_slices.items()):
        rows = payload["rows"]
        near_axis_rows.append({
            "cycle": cycle,
            "time": float(payload["time"]),
            "selected_cells": len(rows),
            "max_abs_C": constraint_max(rows, "con_C"),
            "max_abs_H": constraint_max(rows, "con_H"),
            "max_abs_M": constraint_max(rows, "con_M"),
            "max_abs_Z": constraint_max(rows, "con_Z"),
            "max_abs_A": family_max(rows, "z4c_A"),
            "max_abs_Gamma": family_max(rows, "z4c_Gam"),
            "max_abs_alpha": family_max(rows, "z4c_alpha"),
            "max_abs_chi": family_max(rows, "z4c_chi"),
            "max_abs_detg_minus_one": max(abs(float(row["derived"]["det_conformal_metric_minus_one"])) for row in rows),
            "max_abs_traceA": max(abs(float(row["derived"]["trace_conformal_A"])) for row in rows),
            "parity_residual": parity_residual(rows),
        })

    term_records = load_term_records()
    source_paths.extend(sorted((ROOT / "run_stage_split").glob("z4c_rhs_stage_rank*.log")))
    term_max_by_cycle: dict[tuple[int, str], dict[str, Any]] = {}
    for record in term_records:
        category = term_category(record["term"])
        if category is None or record["selected_cells"] == 0:
            continue
        key = (record["cycle"], category)
        current = term_max_by_cycle.get(key)
        if current is None or record["abs_max"] > current["abs_max"]:
            term_max_by_cycle[key] = {**record, "category": category}
    term_rows = [term_max_by_cycle[key] for key in sorted(term_max_by_cycle)]

    write_csv(
        OUTPUT / "constraint_history.csv", history_csv_rows,
        ["resolution", "time", "cycle", "C_norm2", "H_norm2", "M_norm2", "Z_norm2",
         "C_Linf", "axis_C_rms", "off_axis_C_rms", "axis_lapse", "horizon_status"],
    )
    write_csv(
        OUTPUT / "horizon_history.csv", horizon_csv_rows,
        ["resolution", *HORIZON_FIELDS],
    )
    write_csv(
        OUTPUT / "near_axis_growth.csv", near_axis_rows,
        list(near_axis_rows[0]),
    )
    write_csv(
        OUTPUT / "rhs_term_growth.csv", term_rows,
        ["cycle", "time", "stage", "rank", "category", "term", "selected_cells", "nonfinite",
         "abs_max", "value", "rho", "z", "source"],
    )

    plot_constraint_history(histories)
    plot_horizons(horizons)
    plot_near_axis_growth(near_axis_rows)
    plot_near_axis_slices(cycle_slices)
    plot_rhs_terms(term_rows)

    dominant_terms: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in term_rows}):
        row = max((item for item in term_rows if item["category"] == category), key=lambda item: item["abs_max"])
        dominant_terms[category] = {key: row[key] for key in
                                    ["term", "cycle", "time", "stage", "rank", "abs_max", "value", "rho", "z"]}

    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(source_paths))}
    summary = {
        "schema": "athenak-cartoon-half-plane-kerr-default-gauge-diagnostic-v1",
        "claim_scope": "existing_data_only_no_new_evolution_no_convergence_qualification",
        "source": state["contract"]["source"],
        "executable": state["contract"]["executable"],
        "physics": physics,
        "campaign_cases": state["cases"],
        "resolution_summary": resolution_summary,
        "near_axis": {
            "cycles": [row["cycle"] for row in near_axis_rows],
            "terminal_row": near_axis_rows[-1],
            "selection": cycle_slices[2168]["selection"],
            "interpretation": "puncture-interior Gamma/A-Ricci mode; not outer-boundary arrival or equatorial-reflection asymmetry",
        },
        "rhs_term_diagnosis": {
            "records": len(term_records),
            "cycles": sorted({row["cycle"] for row in term_records}),
            "stages": sorted({row["stage"] for row in term_records}),
            "dominant_by_category": dominant_terms,
            "interpretation": "A trace-Ricci and Gamma Lie/shift-derivative terms dominate; explicit Gamma damping is orders of magnitude smaller",
        },
        "disposition": {
            "status": "failed_convergence_qualification",
            "h32": "completed_t5_but_large_late_constraints_and_lost_horizon_acceptance",
            "h48": "failed_t4.560417_nonfinite_central_diagnostic_after_puncture_interior_mode",
            "h64": "not_run_because_convergence_already_failed",
            "qualification_claim": False,
            "safe_fix_identified": False,
            "threshold_or_gauge_waiver": False,
        },
        "source_hashes": source_hashes,
    }
    strict_dump(OUTPUT / "diagnostic_summary.json", summary)

    payloads = sorted(path for path in OUTPUT.iterdir()
                      if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    manifest_lines = [f"{sha256(path)}  {path.name}" for path in payloads]
    (OUTPUT / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (OUTPUT / "SHA256SUMS.sha256").write_text(
        f"{sha256(OUTPUT / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
