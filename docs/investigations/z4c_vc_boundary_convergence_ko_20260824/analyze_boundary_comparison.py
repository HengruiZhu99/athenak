#!/usr/bin/env python3
"""Compare production-matched radial constraints for Rout=16 and Rout=128."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("C", "H", "M", "Z")
REGIONS = ("R4", "R8", "R12", "R14", "full")
RESOLUTIONS = ("n128", "n256", "n512")
COLORS = {"R4": "#1b9e77", "R8": "#377eb8", "R12": "#984ea3",
          "R14": "#e6ab02", "full": "#e41a1c"}
STYLES = {"n128": ":", "n256": "--", "n512": "-"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def series(rows: list[dict[str, str]], resolution: str, region: str,
           field: str, column: str = "norm2") -> tuple[np.ndarray, np.ndarray]:
    selected = sorted((row for row in rows if row["resolution"] == resolution
                       and row["region"] == region and row["field"] == field),
                      key=lambda row: float(row["axisTau"]))
    require(bool(selected), f"missing {resolution}/{region}/{field}")
    return (np.asarray([float(row["axisTau"]) for row in selected]),
            np.asarray([float(row[column]) for row in selected]))


def convergence_series(rows: list[dict[str, str]], region: str, field: str,
                       column: str) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted((row for row in rows if row["region"] == region
                       and row["field"] == field), key=lambda row: float(row["axisTau"]))
    require(bool(selected), f"missing order {region}/{field}/{column}")
    return (np.asarray([float(row["axisTau"]) for row in selected]),
            np.asarray([float(row[column]) for row in selected]))


def matched_grid(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    low = max(float(x1[0]), float(x2[0]))
    high = min(float(x1[-1]), float(x2[-1]))
    require(high >= low, "series do not overlap")
    return x1[(x1 >= low - 1.0e-13) & (x1 <= high + 1.0e-13)]


def history(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#  [1]"))
    labels = re.findall(r"\[\d+\]=([^\s]+)", header)
    data = np.loadtxt(path, comments="#")
    require(data.ndim == 2 and data.shape[1] == len(labels), f"bad history {path}")
    return {label: data[:, index] for index, label in enumerate(labels)}


def one_history(root: Path) -> Path:
    paths = list(root.glob("*.z4c.user.hst"))
    require(len(paths) == 1, f"expected one history under {root}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--small-history-root", type=Path, required=True)
    parser.add_argument("--large-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    small_integrals = read_csv(args.small / "radial_constraint_integrals.csv")
    large_integrals = read_csv(args.large / "radial_constraint_integrals.csv")
    small_orders = read_csv(args.small / "radial_constraint_convergence.csv")
    large_orders = read_csv(args.large / "radial_constraint_convergence.csv")
    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict] = []
    for resolution in RESOLUTIONS:
        for region in REGIONS:
            for field in FIELDS:
                xs, ys = series(small_integrals, resolution, region, field)
                xl, yl = series(large_integrals, resolution, region, field)
                tau = matched_grid(xs, xl)
                small_values = np.interp(tau, xs, ys)
                large_values = np.interp(tau, xl, yl)
                for t, small_value, large_value in zip(tau, small_values, large_values):
                    comparison_rows.append({
                        "axisTau": float(t), "resolution": resolution,
                        "region": region, "field": field,
                        "rout16_norm2": float(small_value),
                        "rout128_norm2": float(large_value),
                        "rout16_over_rout128": (float(small_value / large_value)
                                                 if large_value != 0.0 else math.nan),
                    })
    write_csv(output / "boundary_sensitivity.csv", comparison_rows)

    terminal_rows: list[dict] = []
    terminal_tau = min(max(float(row["axisTau"]) for row in small_orders),
                       max(float(row["axisTau"]) for row in large_orders))
    for domain, rows in (("Rout16", small_orders), ("Rout128", large_orders)):
        for region in REGIONS:
            for field in FIELDS:
                selected = [row for row in rows if row["region"] == region
                            and row["field"] == field]
                selected.sort(key=lambda row: abs(float(row["axisTau"]) - terminal_tau))
                row = selected[0]
                terminal_rows.append({
                    "domain": domain, "axisTau": float(row["axisTau"]),
                    "region": region, "field": field,
                    "n128": float(row["n128"]), "n256": float(row["n256"]),
                    "n512": float(row["n512"]),
                    "p_128_256": float(row["p_128_256"]),
                    "p_256_512": float(row["p_256_512"]),
                    "p_self": float(row["p_self"]),
                })
    write_csv(output / "boundary_terminal_tables.csv", terminal_rows)

    small_shells = read_csv(args.small / "radial_shell_budget.csv")
    large_shells = read_csv(args.large / "radial_shell_budget.csv")
    shell_rows: list[dict] = []
    for domain, rows in (("Rout16", small_shells), ("Rout128", large_shells)):
        for field in FIELDS:
            candidates = [row for row in rows if row["resolution"] == "n512"
                          and row["field"] == field]
            shell_names = sorted({row["region"] for row in candidates})
            for shell_name in shell_names:
                selected = [row for row in candidates if row["region"] == shell_name]
                selected.sort(key=lambda row: abs(float(row["axisTau"]) - terminal_tau))
                row = selected[0]
                shell_rows.append({
                    "domain": domain, "axisTau": float(row["axisTau"]),
                    "resolution": "n512", "field": field, "shell": shell_name,
                    "norm2": float(row["norm2"]),
                    "fraction_of_full": float(row["fraction_of_full"]),
                    "weighted_rho": float(row["weighted_rho"]),
                    "weighted_z": float(row["weighted_z"]),
                })
    write_csv(output / "terminal_shell_budget.csv", shell_rows)

    shell_orders = {
        "Rout16": ("shell_0_4", "shell_4_8", "shell_8_12", "shell_12_16"),
        "Rout128": ("shell_0_4", "shell_4_8", "shell_8_12", "shell_12_16",
                    "shell_16_32", "shell_32_64", "shell_64_120", "shell_gt_120"),
    }
    for domain in ("Rout16", "Rout128"):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
        for axis, field in zip(axes.flat, FIELDS):
            names = shell_orders[domain]
            values = []
            for name in names:
                selected = [row for row in shell_rows if row["domain"] == domain
                            and row["field"] == field and row["shell"] == name]
                values.append(selected[0]["norm2"] if selected else math.nan)
            axis.bar(range(len(names)), values, color="#377eb8")
            axis.set_yscale("log")
            axis.set_xticks(range(len(names)))
            axis.set_xticklabels([name.replace("shell_", "").replace("_", "-")
                                  for name in names], rotation=32, ha="right")
            axis.set_title(field)
            axis.set_ylabel("N512 squared-constraint inventory")
            axis.grid(alpha=0.24, axis="y", which="both")
        fig.suptitle(f"{domain} radial shell budget at matched terminal central time")
        stem = ("small_domain_radial_shell_budget" if domain == "Rout16"
                else "large_domain_radial_shell_budget")
        fig.savefig(figures / f"{stem}.png", dpi=240)
        fig.savefig(figures / f"{stem}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for axis, field in zip(axes.flat, FIELDS):
        for region in ("R4", "R8", "R12", "full"):
            selected = sorted((row for row in comparison_rows
                               if row["resolution"] == "n512"
                               and row["region"] == region and row["field"] == field),
                              key=lambda row: row["axisTau"])
            axis.semilogy([row["axisTau"] for row in selected],
                          [row["rout16_over_rout128"] for row in selected],
                          color=COLORS[region], label=region)
        axis.axhline(1.0, color="black", linewidth=0.7)
        axis.set_title(field)
        axis.set_xlabel(r"central proper time $\tau_c/M$")
        axis.set_ylabel(r"$Q(R_{out}=16)/Q(R_{out}=128)$")
        axis.grid(alpha=0.24, which="both")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("N512 boundary sensitivity of production-matched constraint inventories")
    fig.savefig(figures / "boundary_sensitivity_n512.png", dpi=240)
    fig.savefig(figures / "boundary_sensitivity_n512.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for axis, field in zip(axes.flat, FIELDS):
        for region in ("R4", "R8", "R12", "full"):
            for domain, rows, style in (("Rout16", small_orders, "--"),
                                        ("Rout128", large_orders, "-")):
                tau, order = convergence_series(rows, region, field, "p_256_512")
                axis.plot(tau, order, color=COLORS[region], linestyle=style,
                          label=f"{region} {domain}")
        axis.set_ylim(-1.0, 9.0)
        axis.set_title(field)
        axis.set_xlabel(r"central proper time $\tau_c/M$")
        axis.set_ylabel(r"direct $p_{256,512}$")
        axis.grid(alpha=0.24)
    axes[0, 0].legend(fontsize=6.8, ncol=2)
    fig.suptitle("Fine-pair order: small versus large outer boundary")
    fig.savefig(figures / "boundary_fine_pair_orders.png", dpi=240)
    fig.savefig(figures / "boundary_fine_pair_orders.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    for region in REGIONS:
        for domain, rows, style in (("Rout16", small_orders, "--"),
                                    ("Rout128", large_orders, "-")):
            tau, order = convergence_series(rows, region, "Z", "p_256_512")
            axis.plot(tau, order, color=COLORS[region], linestyle=style,
                      label=f"{region} {domain}")
    axis.set_ylim(-1.0, 9.0)
    axis.set_xlabel(r"central proper time $\tau_c/M$")
    axis.set_ylabel(r"Z direct $p_{256,512}$")
    axis.grid(alpha=0.24)
    axis.legend(fontsize=7.2, ncol=2)
    fig.savefig(figures / "z_boundary_contamination.png", dpi=240)
    fig.savefig(figures / "z_boundary_contamination.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    axis_deviation_rows: list[dict] = []
    for resolution in RESOLUTIONS:
        small_root = args.small_history_root / f"{resolution}_native_replay_tau4_v1"
        large_name = (f"boundary_{resolution}_record_t6p5_v1" if resolution == "n256"
                      else f"boundary_{resolution}_replay_t6p5_v1")
        small_table = history(one_history(small_root))
        large_table = history(one_history(args.large_history_root / large_name))
        low = max(float(small_table["axisTau"][0]), float(large_table["axisTau"][0]))
        high = min(float(small_table["axisTau"][-1]), float(large_table["axisTau"][-1]))
        tau = np.linspace(low, high, 2001)
        small_k = np.interp(tau, small_table["axisTau"], small_table["axisKret"])
        large_k = np.interp(tau, large_table["axisTau"], large_table["axisKret"])
        mask = (np.abs(small_k) > 1.0e-300) & (np.abs(large_k) > 1.0e-300)
        deviation = float(np.max(np.abs(np.log10(np.abs(small_k[mask]))
                                        - np.log10(np.abs(large_k[mask])))))
        axis_deviation_rows.append({"resolution": resolution,
                                    "maximum_log10_abs_axisKret_deviation": deviation})
        axis.plot(small_table["axisTau"], np.log10(np.abs(small_table["axisKret"])),
                  color={"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}[resolution],
                  linestyle="--", linewidth=1.0, label=f"{resolution} Rout16")
        axis.plot(large_table["axisTau"], np.log10(np.abs(large_table["axisKret"])),
                  color={"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}[resolution],
                  linewidth=1.0, label=f"{resolution} Rout128")
    write_csv(output / "axisKret_boundary_deviation.csv", axis_deviation_rows)
    axis.set_xlabel(r"central proper time $\tau_c/M$")
    axis.set_ylabel(r"$\log_{10}|I(0)|$")
    axis.set_ylim(-8.0, 8.0)
    axis.grid(alpha=0.24)
    axis.legend(fontsize=7.2, ncol=2)
    fig.savefig(figures / "axisKret_small_vs_large_boundary.png", dpi=240)
    fig.savefig(figures / "axisKret_small_vs_large_boundary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
