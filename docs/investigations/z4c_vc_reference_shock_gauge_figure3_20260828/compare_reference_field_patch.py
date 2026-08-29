#!/usr/bin/env python3
"""Compare physically aligned N256/N512 stitched-patch diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = (
    "eta4_rho_l2", "eta4_z_l2", "eta4_max_direction",
    "fluctuation_rms",
    "f2d_above_n256_nyquist_50", "f2d_above_n256_nyquist_65",
    "f2d_above_n256_nyquist_80", "f2d_above_own_nyquist_50",
    "d4_rho_seam_over_away", "k90", "k95", "k99",
)
DISPLAY = {
    "z4c_chi": r"$\chi$", "z4c_Khat": r"$\hat K$",
    "z4c_Theta": r"$\Theta$", "z4c_Gamx": r"$\tilde\Gamma^x$",
    "z4c_Gamz": r"$\tilde\Gamma^z$", "z4c_Axx": r"$\tilde A_{xx}$",
    "z4c_Azz": r"$\tilde A_{zz}$", "z4c_alpha": r"$\alpha$",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path) -> list[dict[str, object]]:
    require(path.is_file(), f"missing metrics: {path}")
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            output: dict[str, object] = dict(row)
            for key in ("time", "axisTau", *METRICS, "duplicate_max_relative"):
                output[key] = float(row[key])
            rows.append(output)
    require(bool(rows), f"empty metrics: {path}")
    return rows


def snapshots(rows: list[dict[str, object]]) -> dict[float, dict[str, dict[str, object]]]:
    output: dict[float, dict[str, dict[str, object]]] = {}
    for row in rows:
        tau = float(row["axisTau"])
        output.setdefault(tau, {})[str(row["variable"])] = row
    require(all(len(group) == 25 for group in output.values()),
            "a field snapshot does not contain all 25 variables")
    return output


def nearest(groups: dict[float, dict[str, dict[str, object]]], target: float,
            tolerance: float) -> tuple[float, dict[str, dict[str, object]]]:
    tau = min(groups, key=lambda value: abs(value - target))
    require(abs(tau - target) <= tolerance,
            f"nearest snapshot tau={tau} misses target={target}")
    return tau, groups[tau]


def save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=230)
    fig.savefig(root / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n256", type=Path, required=True)
    parser.add_argument("--n512", type=Path, required=True)
    parser.add_argument("--target-tau", type=float, nargs="+",
                        default=(8.0, 9.5, 10.3, 11.0))
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {"n256": snapshots(read(args.n256)),
              "n512": snapshots(read(args.n512))}
    selected: dict[float, dict[str, tuple[float, dict[str, dict[str, object]]]]] = {}
    for target in args.target_tau:
        selected[target] = {
            case: nearest(values, target, args.tolerance)
            for case, values in groups.items()
        }

    rows: list[dict[str, object]] = []
    summary_targets: dict[str, object] = {}
    for target, cases in selected.items():
        tau256, values256 = cases["n256"]
        tau512, values512 = cases["n512"]
        require(values256.keys() == values512.keys(), "field schemas differ")
        for variable in values256:
            row: dict[str, object] = {
                "target_tau": target, "n256_tau": tau256, "n512_tau": tau512,
                "variable": variable,
            }
            for metric in METRICS:
                a = float(values256[variable][metric])
                b = float(values512[variable][metric])
                row[f"n256_{metric}"] = a
                row[f"n512_{metric}"] = b
                row[f"ratio_n512_over_n256_{metric}"] = (
                    b / a if a != 0.0 else (1.0 if b == 0.0 else math.inf))
            rows.append(row)
        worst = {}
        for case, (_, values) in cases.items():
            variable = max(values, key=lambda name: float(values[name]["eta4_max_direction"]))
            worst[case] = {
                "variable": variable,
                "eta4_max_direction": float(values[variable]["eta4_max_direction"]),
                "direction": values[variable]["eta4_dominant_direction"],
            }
        summary_targets[f"{target:g}"] = {
            "n256_tau": tau256, "n512_tau": tau512, "largest_eta4": worst,
        }

    with (args.output / "field_patch_comparison.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    last_target = min(args.target_tau, key=lambda value: abs(value - 10.3))
    peak_rows = [row for row in rows if row["target_tau"] == last_target]
    ranked = sorted(peak_rows,
                    key=lambda row: max(float(row["n256_eta4_max_direction"]),
                                        float(row["n512_eta4_max_direction"])),
                    reverse=True)[:12]
    y = np.arange(len(ranked))
    fig, axis = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    axis.barh(y + 0.18, [row["n256_eta4_max_direction"] for row in ranked],
              height=0.34, color="#377eb8", label="N256")
    axis.barh(y - 0.18, [row["n512_eta4_max_direction"] for row in ranked],
              height=0.34, color="#e41a1c", label="N512")
    axis.set_yticks(y, [DISPLAY.get(str(row["variable"]), str(row["variable"]))
                        for row in ranked])
    axis.invert_yaxis(); axis.set_xscale("log")
    axis.set_xlabel(r"$\max(\|\Delta^4_\rho u\|/\|\Delta^2_\rho u\|,"
                    r"\|\Delta^4_z u\|/\|\Delta^2_z u\|)$")
    axis.set_title(f"stitched rho=4--6 patch near tau={last_target:g}")
    axis.grid(alpha=0.22, axis="x", which="both"); axis.legend()
    save(fig, args.output, "field_eta4_ranking_near_peak")

    tracked = [name for name in ("z4c_chi", "z4c_Khat", "z4c_Theta",
                                  "z4c_Gamx", "z4c_Gamz", "z4c_Axx", "z4c_Azz")
               if any(row["variable"] == name for row in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    styles = {"n256": ("#377eb8", "-"), "n512": ("#e41a1c", "--")}
    for variable in tracked:
        for case in ("n256", "n512"):
            xx = [float(row[f"{case}_tau"]) for row in rows if row["variable"] == variable]
            yy = [float(row[f"{case}_eta4_max_direction"])
                  for row in rows if row["variable"] == variable]
            color, linestyle = styles[case]
            axes[0].semilogy(xx, yy, color=color, linestyle=linestyle,
                            alpha=0.75, label=f"{case.upper()} {DISPLAY.get(variable, variable)}")
            yy = [max(float(row[f"{case}_f2d_above_n256_nyquist_50"]), 1.0e-30)
                  for row in rows if row["variable"] == variable]
            axes[1].semilogy(xx, yy, color=color, linestyle=linestyle, alpha=0.75,
                            label=f"{case.upper()} {DISPLAY.get(variable, variable)}")
    axes[0].set_ylabel("fourth/second-difference indicator")
    axes[1].set_ylabel("power above 0.5 N256 Nyquist")
    for axis in axes:
        axis.set_xlabel("central proper time"); axis.grid(alpha=0.22, which="both")
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, args.output, "tracked_field_high_frequency_vs_tau")

    summary = {
        "schema": "athenak.z4c.reference_field_patch_comparison.v1",
        "targets": summary_targets,
        "physical_high_k_reference": "fractions use the N256 physical Nyquist",
        "claim_boundary": "two-resolution descriptive discriminator; not convergence",
    }
    (args.output / "field_patch_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
