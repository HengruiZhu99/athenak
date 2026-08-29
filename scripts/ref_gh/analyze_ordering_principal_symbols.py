#!/usr/bin/env python3
"""Frozen-coefficient Ref-GH ordering/gauge principal-symbol audit.

The matrix is assembled directly from the gamma1=-1 production principal
terms.  Repeated eigenspaces are represented by orthonormal null-space bases,
so the reported basis condition is invariant under rescaling or mixing within
one repeated-speed family.  The stationary q=1 trumpet profiles come from the
independently tested table generator used by the Ref-GH source-unit suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np
from scipy.linalg import null_space


PAIR = [(a, b) for a in range(4) for b in range(a, 4)]
RADII = (0.03, 0.05, 0.08, 0.125, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0)


def assemble_symbol(alpha: float, beta: np.ndarray, normal: np.ndarray,
                    gamma2: float, compatible: bool,
                    gauge_driver: bool) -> np.ndarray:
    """Return A(n) for dt(u)+A(n) d_n(u)=lower order."""
    size = 61 if gauge_driver else 50
    matrix = np.zeros((size, size))
    beta_s = float(beta @ normal)
    psi_offset, pi_offset, phi_offset = 0, 10, 20
    hhat_offset, theta_offset = 50, 54
    spacetime_normal = np.concatenate(([0.0], normal))

    for component, (a, b) in enumerate(PAIR):
        psi = psi_offset + component
        pi = pi_offset + component
        matrix[pi, psi] = gamma2 * beta_s
        matrix[pi, pi] = -beta_s
        for i in range(3):
            matrix[pi, phi_offset + 10 * i + component] = alpha * normal[i]
        if gauge_driver:
            matrix[pi, hhat_offset + b] += alpha * spacetime_normal[a]
            matrix[pi, hhat_offset + a] += alpha * spacetime_normal[b]

        for i in range(3):
            phi_i = phi_offset + 10 * i + component
            matrix[phi_i, psi] = -alpha * gamma2 * normal[i]
            matrix[phi_i, pi] = alpha * normal[i]
            if compatible:
                for j in range(3):
                    matrix[phi_i, phi_offset + 10 * j + component] = (
                        -normal[i] * beta[j]
                    )
            else:
                matrix[phi_i, phi_i] = -beta_s

    if gauge_driver:
        for a in range(4):
            matrix[hhat_offset + a, hhat_offset + a] = -beta_s
            # eta=1 in the controlling matrix.  Eta changes eigenvectors but
            # not speeds or diagonalizability; this is the production value.
            matrix[theta_offset + a, hhat_offset + a] = beta_s
    return matrix


def cluster(values: list[float], scale: float) -> list[float]:
    tolerance = 2.0e-11 * max(1.0, scale)
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
        else:
            result[-1] = 0.5 * (result[-1] + value)
    return result


def eigenspace_audit(matrix: np.ndarray, speeds: list[float]) -> dict[str, float]:
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    bases = []
    dimensions = []
    for speed in cluster(speeds, scale):
        basis = null_space(matrix - speed * np.eye(matrix.shape[0]), rcond=1.0e-10)
        bases.append(basis)
        dimensions.append(int(basis.shape[1]))
    geometric_dimension = sum(dimensions)
    condition = math.inf
    if geometric_dimension == matrix.shape[0]:
        condition = float(np.linalg.cond(np.hstack(bases)))
    eigenvalues = np.linalg.eigvals(matrix)
    return {
        "dimension": matrix.shape[0],
        "geometric_dimension": geometric_dimension,
        "basis_condition": condition,
        "maximum_imaginary_eigenvalue": float(np.max(np.abs(eigenvalues.imag))),
    }


def expected_speeds(beta_s: float, alpha: float, compatible: bool,
                    gauge_driver: bool) -> list[float]:
    if compatible:
        speeds = [0.0] * 30 + [-beta_s + alpha] * 10 + [-beta_s - alpha] * 10
    else:
        speeds = ([0.0] * 10 + [-beta_s] * 20
                  + [-beta_s + alpha] * 10 + [-beta_s - alpha] * 10)
    if gauge_driver:
        speeds += [-beta_s] * 4 + [0.0] * 7
    return speeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite = pathlib.Path(__file__).resolve().parents[2] / "tst/test_suite/ref_gh"
    sys.path.insert(0, str(suite))
    import generate_trumpet_table as trumpet  # pylint: disable=import-outside-toplevel
    from trumpet_reference_audit import interpolate  # pylint: disable=import-outside-toplevel

    log_r, data, *_ = trumpet.build_table(1025, 1.0e-4, 128.0)
    systems = (
        ("compatible_gamma2_0", 0.0, True, False),
        ("compatible_gamma2_1", 1.0, True, False),
        ("standard_gamma2_1", 1.0, False, False),
        ("standard_gamma2_1_driver", 1.0, False, True),
    )
    rows = []
    crossing_rows = []
    angles = np.linspace(0.0, 0.5 * math.pi, 37)
    for radius in RADII:
        alpha = float(interpolate(log_r, data, 0, radius)[0])
        psi2 = float(interpolate(log_r, data, 3, radius)[0] / radius)
        shift_q = float(interpolate(log_r, data, 6, radius)[0])
        beta_magnitude = psi2 * shift_q * radius
        beta = np.array([beta_magnitude, 0.0, 0.0])
        chi_beta = beta_magnitude / alpha
        for name, gamma2, compatible, gauge_driver in systems:
            worst_condition = 0.0
            minimum_geometric_dimension = 10**9
            maximum_imaginary = 0.0
            worst_angle = 0.0
            for angle in angles:
                normal = np.array([math.cos(angle), math.sin(angle), 0.0])
                beta_s = float(beta @ normal)
                matrix = assemble_symbol(
                    alpha, beta, normal, gamma2, compatible, gauge_driver
                )
                audit = eigenspace_audit(
                    matrix,
                    expected_speeds(beta_s, alpha, compatible, gauge_driver),
                )
                if audit["basis_condition"] > worst_condition:
                    worst_condition = audit["basis_condition"]
                    worst_angle = angle
                minimum_geometric_dimension = min(
                    minimum_geometric_dimension,
                    int(audit["geometric_dimension"]),
                )
                maximum_imaginary = max(
                    maximum_imaginary,
                    audit["maximum_imaginary_eigenvalue"],
                )
            rows.append({
                "system": name,
                "radius": radius,
                "alpha": alpha,
                "beta": beta_magnitude,
                "chi_beta": chi_beta,
                "angle_sweep_worst_condition": worst_condition,
                "angle_sweep_worst_angle": worst_angle,
                "angle_sweep_min_geometric_dimension": minimum_geometric_dimension,
                "angle_sweep_max_imaginary": maximum_imaginary,
            })

            if compatible and chi_beta >= 1.0:
                angle = math.acos(alpha / beta_magnitude)
                normal = np.array([math.cos(angle), math.sin(angle), 0.0])
                beta_s = float(beta @ normal)
                matrix = assemble_symbol(
                    alpha, beta, normal, gamma2, compatible, gauge_driver
                )
                audit = eigenspace_audit(
                    matrix,
                    expected_speeds(beta_s, alpha, compatible, gauge_driver),
                )
                crossing_rows.append({
                    "system": name,
                    "radius": radius,
                    "chi_beta": chi_beta,
                    "crossing_angle": angle,
                    "dimension": audit["dimension"],
                    "geometric_dimension": audit["geometric_dimension"],
                    "basis_condition": audit["basis_condition"],
                    "maximum_imaginary_eigenvalue": audit[
                        "maximum_imaginary_eigenvalue"
                    ],
                })

    def write_tsv(path: pathlib.Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)

    write_tsv(args.output_dir / "principal_symbol_radial.tsv", rows)
    write_tsv(args.output_dir / "compatible_zero_speed_crossings.tsv", crossing_rows)
    summary = {
        "radii": list(RADII),
        "angle_count": len(angles),
        "compatible_crossing_count": len(crossing_rows),
        "compatible_crossings_defective": all(
            row["geometric_dimension"] < row["dimension"]
            for row in crossing_rows
        ),
        "standard_driver_maximum_angle_sweep_condition": max(
            row["angle_sweep_worst_condition"] for row in rows
            if row["system"] == "standard_gamma2_1_driver"
        ),
        "standard_driver_minimum_geometric_dimension": min(
            row["angle_sweep_min_geometric_dimension"] for row in rows
            if row["system"] == "standard_gamma2_1_driver"
        ),
        "maximum_imaginary_eigenvalue": max(
            row["angle_sweep_max_imaginary"] for row in rows
        ),
        "interpretation": (
            "Compatible ordering is defective for a direction satisfying "
            "beta_s=alpha wherever chi_beta>=1. Standard ordering, including "
            "the improved gauge principal coupling, remains diagonalizable."
        ),
    }
    (args.output_dir / "principal_symbol_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
