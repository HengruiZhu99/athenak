#!/usr/bin/env python3
"""Deterministic qualification diagnostics for native-VC midpoint transfer."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RULES = {
    4: (-1, np.asarray((-1.0, 9.0, 9.0, -1.0)) / 16.0),
    6: (-2, np.asarray((3.0, -25.0, 150.0, 150.0, -25.0, 3.0)) / 256.0),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def midpoint_symbol(order: int, theta: np.ndarray) -> np.ndarray:
    left, weights = RULES[order]
    offsets = left + np.arange(weights.size)
    return np.sum(weights[:, None] * np.exp(1j * offsets[:, None] * theta), axis=0)


def branch_amplitudes(order: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relative_midpoint = midpoint_symbol(order, theta) * np.exp(-0.5j * theta)
    return 0.5 * (1.0 + relative_midpoint), 0.5 * (1.0 - relative_midpoint)


def polynomial_checks(order: int) -> dict[str, object]:
    left, weights = RULES[order]
    offsets = left + np.arange(weights.size, dtype=float)
    errors = []
    for degree in range(order):
        errors.append(abs(float(np.dot(weights, offsets ** degree)) - 0.5 ** degree))
    symmetry_error = float(np.max(np.abs(weights - weights[::-1])))
    sum_error = abs(float(np.sum(weights)) - 1.0)
    tolerance = 64.0 * np.finfo(float).eps
    require(max(errors) <= tolerance, f"O{order} polynomial reproduction failed")
    require(symmetry_error == 0.0 and sum_error == 0.0,
            f"O{order} symmetry/constant preservation failed")
    return {
        "order": order,
        "reproduced_degree": order - 1,
        "degree_errors": errors,
        "symmetry_error": symmetry_error,
        "weight_sum_error": sum_error,
        "l1_weight_norm": float(np.sum(np.abs(weights))),
        "tensor_l1_2d": float(np.sum(np.abs(weights)) ** 2),
        "tensor_l1_3d": float(np.sum(np.abs(weights)) ** 3),
        "collapsed_dimension_is_identity": True,
        "coincident_injection_is_exact": True,
    }


def periodic_prolong(coarse: np.ndarray, order: int) -> np.ndarray:
    left, weights = RULES[order]
    fine = np.empty(2 * coarse.size, dtype=coarse.dtype)
    fine[0::2] = coarse
    for index in range(coarse.size):
        value = 0.0
        for point, weight in enumerate(weights):
            value += weight * coarse[(index + left + point) % coarse.size]
        fine[2 * index + 1] = value
    return fine


def high_fraction(values: np.ndarray, threshold: float = 0.5) -> float:
    centered = values - np.mean(values)
    power = np.abs(np.fft.fft(centered)) ** 2
    frequency = np.abs(np.fft.fftfreq(values.size)) / 0.5
    total = float(np.sum(power))
    return 0.0 if total == 0.0 else float(np.sum(power[frequency >= threshold]) / total)


def cycle_checks(order: int) -> list[dict[str, object]]:
    cases = {
        "smooth_fourier": np.cos(2.0 * math.pi * 3.0 * np.arange(64) / 64.0),
        "near_nyquist": np.cos(2.0 * math.pi * 29.0 * np.arange(64) / 64.0),
        "localized_pulse": np.exp(-((np.arange(64) - 31.5) / 7.0) ** 2),
    }
    rows = []
    for name, initial in cases.items():
        coarse = initial.copy()
        first_fine = periodic_prolong(coarse, order)
        for _ in range(16):
            fine = periodic_prolong(coarse, order)
            restricted = fine[0::2].copy()
            require(np.array_equal(restricted, coarse),
                    f"O{order} injection did not recover coincident nodes")
            coarse = restricted
        final_fine = periodic_prolong(coarse, order)
        rows.append({
            "order": order,
            "case": name,
            "coarse_cycle_linf_change": float(np.max(np.abs(coarse - initial))),
            "fine_cycle_linf_change": float(np.max(np.abs(final_fine - first_fine))),
            "fine_high_fraction_50": high_fraction(first_fine, 0.5),
            "fine_high_fraction_80": high_fraction(first_fine, 0.8),
            "fine_overshoot_above_parent_range": float(max(
                0.0, np.max(first_fine) - np.max(initial),
                np.min(initial) - np.min(first_fine))),
        })
    fine_index = np.arange(128)
    for kh_fraction in (0.125, 0.5, 0.8, 1.0):
        initial_fine = np.cos(math.pi * kh_fraction * fine_index)
        restricted = initial_fine[0::2].copy()
        reconstructed = periodic_prolong(restricted, order)
        rows.append({
            "order": order,
            "case": f"fine_restrict_prolong_kh_{kh_fraction:.3f}",
            "coarse_cycle_linf_change": float("nan"),
            "fine_cycle_linf_change": float(
                np.max(np.abs(reconstructed - initial_fine))),
            "fine_high_fraction_50": high_fraction(reconstructed, 0.5),
            "fine_high_fraction_80": high_fraction(reconstructed, 0.8),
            "fine_overshoot_above_parent_range": float(max(
                0.0, np.max(reconstructed) - np.max(initial_fine),
                np.min(initial_fine) - np.min(reconstructed))),
        })
    return rows


def tensor_response_rows(order: int) -> list[dict[str, object]]:
    modes = {
        "2d_low": (0.25, 0.25),
        "2d_mixed": (0.75, 0.25),
        "2d_high": (0.9, 0.9),
        "3d_low": (0.25, 0.25, 0.25),
        "3d_mixed": (0.75, 0.5, 0.25),
        "3d_high": (0.9, 0.9, 0.9),
    }
    rows = []
    for name, fractions in modes.items():
        thetas = np.asarray(fractions) * math.pi
        symbol = np.prod(midpoint_symbol(order, thetas))
        branches = [branch_amplitudes(order, np.asarray((theta,)))
                    for theta in thetas]
        physical = float(np.prod([abs(branch[0][0]) for branch in branches]))
        branch_power = 1.0
        for physical_1d, image_1d in branches:
            branch_power *= abs(physical_1d[0]) ** 2 + abs(image_1d[0]) ** 2
        physical_power = physical ** 2
        rows.append({
            "order": order,
            "mode": name,
            "dimensions": len(fractions),
            "kH_over_pi": ";".join(str(value) for value in fractions),
            "midpoint_magnitude": float(abs(symbol)),
            "midpoint_phase": float(np.angle(symbol)),
            "physical_branch_amplitude": physical,
            "image_branch_power_fraction": float(
                0.0 if branch_power == 0.0 else
                (branch_power - physical_power) / branch_power),
        })
    return rows


def nonlinear_checks(order: int) -> list[dict[str, object]]:
    x = np.arange(128, dtype=float) / 128.0
    profiles = {
        "chi_smooth_positive": 0.18 + 0.76 * np.exp(-((x - 0.49) / 0.12) ** 2),
        "lapse_smooth_positive": 1.0 - 0.68 *
                                 np.exp(-((x - 0.52) / 0.11) ** 2),
        "metric_smooth_positive": 1.0 + 0.24 * np.cos(2.0 * math.pi * x) +
                                  0.05 * np.cos(6.0 * math.pi * x),
    }
    rows = []
    for name, coarse in profiles.items():
        fine = periodic_prolong(coarse, order)
        below = max(0.0, float(np.min(coarse) - np.min(fine)))
        above = max(0.0, float(np.max(fine) - np.max(coarse)))
        rows.append({
            "order": order,
            "profile": name,
            "parent_min": float(np.min(coarse)),
            "child_min": float(np.min(fine)),
            "parent_max": float(np.max(coarse)),
            "child_max": float(np.max(fine)),
            "undershoot": below,
            "overshoot": above,
            "positive": bool(np.min(fine) > 0.0),
            "high_fraction_50": high_fraction(fine, 0.5),
            "high_fraction_80": high_fraction(fine, 0.8),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"cannot write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    theta = np.linspace(0.0, math.pi, 4097)
    summary = {"schema": 1, "rules": {}}
    response_rows = []
    cycle_rows = []
    nonlinear_rows = []
    tensor_rows = []

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), constrained_layout=True)
    for order, color in ((4, "tab:blue"), (6, "tab:orange")):
        symbol = midpoint_symbol(order, theta)
        physical, image = branch_amplitudes(order, theta)
        phase_error = np.angle(symbol * np.exp(-0.5j * theta))
        polynomial = polynomial_checks(order)
        cycles = cycle_checks(order)
        nonlinear = nonlinear_checks(order)
        tensor = tensor_response_rows(order)
        cycle_rows.extend(cycles)
        nonlinear_rows.extend(nonlinear)
        tensor_rows.extend(tensor)
        for index in range(theta.size):
            response_rows.append({
                "order": order,
                "kH_over_pi": float(theta[index] / math.pi),
                "midpoint_magnitude": float(abs(symbol[index])),
                "midpoint_phase_error": float(phase_error[index]),
                "physical_branch_amplitude": float(abs(physical[index])),
                "image_branch_amplitude": float(abs(image[index])),
            })
        summary["rules"][str(order)] = {
            "polynomial": polynomial,
            "max_midpoint_magnitude": float(np.max(np.abs(symbol))),
            "max_image_amplitude": float(np.max(np.abs(image))),
            "near_nyquist_midpoint_magnitude": float(abs(symbol[-1])),
            "near_nyquist_image_amplitude": float(abs(image[-1])),
            "cycles": cycles,
            "nonlinear": nonlinear,
            "tensor_response": tensor,
        }
        axes[0, 0].plot(theta / math.pi, np.abs(symbol), color=color,
                        label=f"O{order}")
        axes[0, 1].plot(theta / math.pi, phase_error, color=color,
                        label=f"O{order}")
        axes[1, 0].plot(theta / math.pi, np.abs(physical), color=color,
                        label=f"O{order}")
        axes[1, 1].plot(theta / math.pi, np.abs(image), color=color,
                        label=f"O{order}")

    axes[0, 0].set_title("Midpoint magnitude")
    axes[0, 1].set_title("Midpoint phase error")
    axes[1, 0].set_title("Physical branch amplitude")
    axes[1, 1].set_title("Fine-grid image amplitude")
    for axis in axes.flat:
        axis.set_xlabel(r"$kH/\pi$")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output / "fourier_transfer_response.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    names = [row["case"] for row in cycle_rows
             if row["order"] == 4 and not row["case"].startswith("fine_")]
    xloc = np.arange(len(names))
    width = 0.36
    for offset, order, color in ((-width / 2, 4, "tab:blue"),
                                 (width / 2, 6, "tab:orange")):
        selected = [row for row in cycle_rows
                    if row["order"] == order and
                    not row["case"].startswith("fine_")]
        axes[0].bar(xloc + offset,
                    [row["fine_high_fraction_50"] for row in selected],
                    width, label=f"O{order}", color=color)
        axes[1].bar(xloc + offset,
                    [row["fine_overshoot_above_parent_range"] for row in selected],
                    width, label=f"O{order}", color=color)
    axes[0].set_title("Fine high-k fraction")
    axes[1].set_title("Pointwise overshoot")
    for axis in axes:
        axis.set_xticks(xloc, names, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.savefig(output / "repeated_transfer_controls.png", dpi=180)
    plt.close(figure)

    write_csv(output / "fourier_response.csv", response_rows)
    write_csv(output / "repeated_cycles.csv", cycle_rows)
    write_csv(output / "nonlinear_profiles.csv", nonlinear_rows)
    write_csv(output / "tensor_response.csv", tensor_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "o4_max_image": summary["rules"]["4"]["max_image_amplitude"],
        "o6_max_image": summary["rules"]["6"]["max_image_amplitude"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
