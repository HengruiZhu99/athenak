#!/usr/bin/env python3
"""Plot the authenticated three original-domain and two domain-64 controls."""

from __future__ import annotations

import csv
import bisect
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
REFERENCE = EVIDENCE / "reference"
FIGURES = ROOT / "figures"
ANALYSIS = ROOT / "analysis"
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")

CASES: dict[str, tuple[str, ...]] = {
    "d16_fixed_ko002": (
        "domain16/run/cases/fixed_eta2_tau1_kappa1_l20_nocd_ko002/"
        "brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko002_n128.z4c.user.hst",
    ),
    "d16_fixed_ko05": (
        "domain16/run/cases/fixed_eta2_tau1_kappa1_l20_nocd_ko05/"
        "brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko05_n128.z4c.user.hst",
    ),
    "d16_zero_ko05": (
        "domain16/run/cases/zero_shift_tau1_kappa1_l20_nocd_ko05/"
        "brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_n128.z4c.user.hst",
    ),
    "d64_fixed_ko05": (
        "domain64/run/cases/fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64/"
        "brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64_n128.z4c.user.hst",
    ),
    "d64_zero_ko05": (
        "domain64/run/cases/zero_shift_tau1_kappa1_l20_nocd_ko05_domain64/"
        "brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_n128.z4c.user.hst",
        "domain64/continuation/cases/"
        "zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_restart_v8/"
        "brill_fig3_zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_restart_v8_n128.z4c.user.hst",
    ),
}

STYLES = {
    "d16_fixed_ko002": (r"$R=Z=16$: fixed shift, KO=0.02", "#377eb8", "-"),
    "d16_fixed_ko05": (r"$R=Z=16$: fixed shift, KO=0.5", "#e41a1c", "-"),
    "d16_zero_ko05": (r"$R=Z=16$: zero shift, KO=0.5", "#4daf4a", "-"),
    "d64_fixed_ko05": (r"$R=Z=64$: fixed shift, KO=0.5", "#e41a1c", "--"),
    "d64_zero_ko05": (r"$R=Z=64$: zero shift, KO=0.5", "#4daf4a", "--"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_history(path: Path) -> list[dict[str, float]]:
    labels: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="strict").splitlines():
        if line.startswith("#"):
            for index, name in HEADER.findall(line):
                labels[name] = int(index) - 1
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if all(math.isfinite(value) for value in values):
            rows.append({name: values[index] for name, index in labels.items()})
    required = {
        "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
        "max_abs_K", "maxAbsKret", "nmb_total", "maxRefLev", "cycle",
        "axisLapse", "axisTau", "axisKret",
    }
    if not rows or not required <= labels.keys():
        raise RuntimeError(f"invalid history: {path}")
    return rows


def merge_history_segments(paths: tuple[str, ...]) -> list[dict[str, float]]:
    """Merge restart segments, preferring the later segment on overlap."""
    merged: list[dict[str, float]] = []
    for relpath in paths:
        segment = read_history(EVIDENCE / relpath)
        first = segment[0]["time"]
        merged = [row for row in merged if row["time"] < first]
        merged.extend(segment)
    if any(b["time"] <= a["time"] for a, b in zip(merged, merged[1:])):
        raise RuntimeError(f"non-monotonic merged history: {paths}")
    return merged


def load_cases() -> dict[str, list[dict[str, float]]]:
    return {name: merge_history_segments(paths) for name, paths in CASES.items()}


def finite_xy(rows: list[dict[str, float]], x: str, y: str, positive=False):
    pairs = []
    for row in rows:
        xv, yv = row[x], row[y]
        if math.isfinite(xv) and math.isfinite(yv) and (not positive or yv > 0.0):
            pairs.append((xv, yv))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def interpolate(rows: list[dict[str, float]], field: str, time: float) -> float:
    times = [row["time"] for row in rows]
    index = bisect.bisect_left(times, time)
    if index == 0:
        return rows[0][field]
    if index == len(rows):
        return rows[-1][field]
    left, right = rows[index - 1], rows[index]
    weight = (time - left["time"]) / (right["time"] - left["time"])
    return left[field] + weight * (right[field] - left[field])


def boundary_metrics(
    small: list[dict[str, float]],
    large: list[dict[str, float]],
    comparison_end: float,
) -> dict[str, float]:
    end = min(comparison_end, small[-1]["time"], large[-1]["time"])
    samples = [row["time"] for row in small if row["time"] <= end]
    samples.extend(row["time"] for row in large if row["time"] <= end)
    samples = sorted(set(samples))
    lapse_abs = []
    curvature_rel = []
    constraint_rel = []
    for time in samples:
        values = {
            (side, field): interpolate(rows, field, time)
            for side, rows in (("small", small), ("large", large))
            for field in ("axisLapse", "axisKret", "C-norm2")
        }
        lapse_abs.append(abs(values["small", "axisLapse"] - values["large", "axisLapse"]))
        for field, target in (("axisKret", curvature_rel), ("C-norm2", constraint_rel)):
            a, b = values["small", field], values["large", field]
            target.append(abs(a - b) / max(abs(a), abs(b), 1.0e-300))
    endpoint = {
        (side, field): interpolate(rows, field, end)
        for side, rows in (("small", small), ("large", large))
        for field in ("axisLapse", "axisKret", "C-norm2")
    }
    return {
        "comparison_end_time": end,
        "samples": len(samples),
        "max_abs_axis_lapse_difference": max(lapse_abs),
        "max_relative_axis_curvature_difference": max(curvature_rel),
        "max_relative_global_C_difference": max(constraint_rel),
        "endpoint_abs_axis_lapse_difference": abs(
            endpoint["small", "axisLapse"] - endpoint["large", "axisLapse"]
        ),
        "endpoint_relative_axis_curvature_difference": abs(
            endpoint["small", "axisKret"] - endpoint["large", "axisKret"]
        ) / max(
            abs(endpoint["small", "axisKret"]),
            abs(endpoint["large", "axisKret"]),
            1.0e-300,
        ),
        "endpoint_relative_global_C_difference": abs(
            endpoint["small", "C-norm2"] - endpoint["large", "C-norm2"]
        ) / max(
            abs(endpoint["small", "C-norm2"]),
            abs(endpoint["large", "C-norm2"]),
            1.0e-300,
        ),
    }


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=200)
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_figure3(curves: dict[str, list[dict[str, float]]]) -> None:
    paper = read_csv(REFERENCE / "figure3_published_curves.csv")
    paper_styles = {
        "bamps": ("BAMPS (rendered paper curve)", "#aaaaaa", ":"),
        "prague": ("Prague (rendered paper curve)", "#777777", "--"),
        "sphGR": ("sphGR (rendered paper curve)", "#444444", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), constrained_layout=True)
    for ax in axes:
        for key, (label, color, linestyle) in paper_styles.items():
            selected = [row for row in paper if row["series"] == key]
            ax.plot(
                [float(row["tau"]) for row in selected],
                [float(row["log10_abs_I"]) for row in selected],
                label=label, color=color, linestyle=linestyle, linewidth=1.2,
            )
        for key, rows in curves.items():
            label, color, linestyle = STYLES[key]
            xx, raw = finite_xy(rows, "axisTau", "axisKret")
            pairs = [(xv, math.log10(abs(yv))) for xv, yv in zip(xx, raw) if yv != 0.0]
            ax.plot(
                [p[0] for p in pairs], [p[1] for p in pairs],
                label=label, color=color, linestyle=linestyle, linewidth=1.55,
            )
            ax.scatter(pairs[-1][0], pairs[-1][1], color=color, marker="x", s=30)
        ax.set_xlim(0.0, 15.05)
        ax.set_xlabel(r"central proper time $\tau$")
        ax.set_ylabel(r"$\log_{10}|I|$ on the symmetry axis")
        ax.grid(alpha=0.23)
    axes[0].set_ylim(-7.0, 16.5)
    axes[0].set_title("Published Figure-3 scale")
    axes[1].set_ylim(-7.0, 51.0)
    axes[1].set_title("Full finite AthenaK range")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Brill A=-0.047: zero-damping KO/shift/domain controls")
    save(fig, "figure3_five_case_overlay")


def plot_constraints(curves: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
    for ax, field, title in zip(
        axes.flat,
        ("C-norm2", "H-norm2", "M-norm2", "Z-norm2"),
        (r"$\|C\|_2$", r"$\|H\|_2$", r"$\|M\|_2$", r"$\|Z\|_2$"),
    ):
        for key, rows in curves.items():
            label, color, linestyle = STYLES[key]
            xx, yy = finite_xy(rows, "time", field, positive=True)
            ax.semilogy(xx, yy, label=label, color=color, linestyle=linestyle, linewidth=1.45)
        ax.set_title(title)
        ax.set_xlabel(r"coordinate time $t$")
        ax.grid(alpha=0.23, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Brill A=-0.047: global constraint histories")
    save(fig, "constraints_five_case")


def plot_gauge_amr(curves: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    specs = (
        ("axisLapse", "axis lapse", False),
        ("max_abs_K", r"$\max |K|$", True),
        ("maxAbsKret", r"$\max |I|$", True),
        ("maxRefLev", "maximum physical AMR level", False),
        ("nmb_total", "total MeshBlocks", True),
        ("dt", r"time step $\Delta t$", True),
    )
    for ax, (field, title, log_y) in zip(axes.flat, specs):
        for key, rows in curves.items():
            label, color, linestyle = STYLES[key]
            xx, yy = finite_xy(rows, "time", field, positive=log_y)
            plot = ax.semilogy if log_y else ax.plot
            plot(xx, yy, label=label, color=color, linestyle=linestyle, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel(r"coordinate time $t$")
        ax.grid(alpha=0.23, which="both")
    axes[1, 0].axhline(20, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Brill A=-0.047: lapse, curvature, AMR, and timestep")
    save(fig, "gauge_amr_five_case")


def plot_boundary_pairs(curves: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), constrained_layout=True)
    pairs = (
        ("d16_fixed_ko05", "d64_fixed_ko05", "fixed shift"),
        ("d16_zero_ko05", "d64_zero_ko05", "zero shift"),
    )
    for column, (small, large, title) in enumerate(pairs):
        for key in (small, large):
            label, color, linestyle = STYLES[key]
            xx, yy = finite_xy(curves[key], "time", "C-norm2", positive=True)
            axes[0, column].semilogy(xx, yy, label=label, color=color, linestyle=linestyle)
            xx, yy = finite_xy(curves[key], "axisTau", "axisKret")
            good = [(xv, math.log10(abs(yv))) for xv, yv in zip(xx, yy) if yv != 0.0]
            axes[1, column].plot(
                [p[0] for p in good], [p[1] for p in good],
                label=label, color=color, linestyle=linestyle,
            )
        axes[0, column].set_title(f"{title}: global C norm")
        axes[0, column].set_xlabel(r"coordinate time $t$")
        axes[0, column].set_ylabel(r"$\|C\|_2$")
        axes[1, column].set_title(f"{title}: central curvature")
        axes[1, column].set_xlabel(r"central proper time $\tau$")
        axes[1, column].set_ylabel(r"$\log_{10}|I|$")
        end = min(curves[small][-1]["time"], curves[large][-1]["time"])
        times = [row["time"] for row in curves[small] if row["time"] <= end]
        lapse_difference = []
        curvature_difference = []
        for time in times:
            lapse_small = interpolate(curves[small], "axisLapse", time)
            lapse_large = interpolate(curves[large], "axisLapse", time)
            curvature_small = interpolate(curves[small], "axisKret", time)
            curvature_large = interpolate(curves[large], "axisKret", time)
            lapse_difference.append(max(abs(lapse_small - lapse_large), 1.0e-30))
            curvature_difference.append(
                max(
                    abs(curvature_small - curvature_large)
                    / max(abs(curvature_small), abs(curvature_large), 1.0e-300),
                    1.0e-30,
                )
            )
        axes[2, column].semilogy(
            times, lapse_difference, color="#377eb8", label=r"$|\Delta\alpha_{\rm axis}|$"
        )
        axes[2, column].semilogy(
            times, curvature_difference, color="#984ea3",
            label=r"relative $|\Delta I_{\rm axis}|$",
        )
        axes[2, column].set_title(f"{title}: matched-time differences")
        axes[2, column].set_xlabel(r"coordinate time $t$")
        axes[2, column].legend(fontsize=8)
        for row in range(3):
            axes[row, column].grid(alpha=0.23, which="both")
        for row in range(2):
            axes[row, column].legend(fontsize=8)
    fig.suptitle("Original versus enlarged outer boundary at fixed base spacing")
    save(fig, "boundary_distance_pairwise")


def write_summary(curves: dict[str, list[dict[str, float]]]) -> None:
    d16 = json.loads((EVIDENCE / "domain16/run/comparison.json").read_text())
    d64 = json.loads((EVIDENCE / "domain64/run/comparison.json").read_text())
    continuation_path = (
        EVIDENCE / "domain64/continuation/cases/"
        "zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_restart_v8/result.json"
    )
    continuation = json.loads(continuation_path.read_text())
    fixed64 = [
        case for case in d64["cases"]
        if case["name"] == "fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64"
    ]
    if len(fixed64) != 1:
        raise RuntimeError("missing unique enlarged-domain fixed-shift result")
    results = d16["cases"] + [fixed64[0], continuation]
    result_by_name = {case["name"]: case for case in results}
    key_to_name = {
        "d16_fixed_ko002": "fixed_eta2_tau1_kappa1_l20_nocd_ko002",
        "d16_fixed_ko05": "fixed_eta2_tau1_kappa1_l20_nocd_ko05",
        "d16_zero_ko05": "zero_shift_tau1_kappa1_l20_nocd_ko05",
        "d64_fixed_ko05": "fixed_eta2_tau1_kappa1_l20_nocd_ko05_domain64",
        "d64_zero_ko05": "zero_shift_tau1_kappa1_l20_nocd_ko05_domain64_restart_v8",
    }
    cases = {}
    for key, rows in curves.items():
        result = result_by_name[key_to_name[key]]
        cases[key] = {
            "result": result,
            "history_segments": [
                {"path": path, "sha256": sha256(EVIDENCE / path)}
                for path in CASES[key]
            ],
            "last_finite_history": rows[-1],
            "max_refinement_level": int(max(row["maxRefLev"] for row in rows)),
            "max_meshblocks": int(max(row["nmb_total"] for row in rows)),
            "horizon_found": any(row.get("ahStatus", 0.0) > 0.5 for row in rows),
        }
    summary = {
        "schema": "athenak_brill_figure3_five_run_domain_comparison_v1",
        "initial_data": {
            "amplitude": -0.047,
            "adm_mass": 2.660301967997158,
            "direct_irisk_interpolation": True,
            "precollapsed_lapse": "psi^-2",
        },
        "common": {
            "tau": 1.0,
            "kappa": 1.0,
            "scale": "max_domain_abs_K",
            "constraint_damping": False,
            "dchi_max": 0.02,
            "max_physical_level": 20,
            "base_spacing": 0.25,
        },
        "cases": cases,
        "pairwise_boundary_metrics": {
            "fixed_shift_through_t16": boundary_metrics(
                curves["d16_fixed_ko05"], curves["d64_fixed_ko05"], 16.0
            ),
            "zero_shift_through_t7": boundary_metrics(
                curves["d16_zero_ko05"], curves["d64_zero_ko05"], 7.0
            ),
            "zero_shift_full_common_interval": boundary_metrics(
                curves["d16_zero_ko05"], curves["d64_zero_ko05"],
                min(
                    curves["d16_zero_ko05"][-1]["time"],
                    curves["d64_zero_ko05"][-1]["time"],
                ),
            ),
        },
        "domain16_comparison_sha256": sha256(EVIDENCE / "domain16/run/comparison.json"),
        "domain64_comparison_sha256": sha256(EVIDENCE / "domain64/run/comparison.json"),
        "domain64_continuation_result_sha256": sha256(continuation_path),
        "qualification_claim": False,
        "figure3_reproduction_claim": False,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "five_case_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    curves = load_cases()
    plot_figure3(curves)
    plot_constraints(curves)
    plot_gauge_amr(curves)
    plot_boundary_pairs(curves)
    write_summary(curves)
    print("FIVE_CASE_PLOTS_COMPLETE")


if __name__ == "__main__":
    main()
