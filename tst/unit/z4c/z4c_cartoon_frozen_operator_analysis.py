#!/usr/bin/env python3
"""Representative frozen radial stability analysis for Cartoon Z4c closures.

This utility deliberately analyzes a documented six-field linear proxy, not the
full nonlinear Z4c principal symbol.  It compares only the radial discretization
choices at identical outer closure and continuum couplings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.linalg


FIELDS = ("g", "A", "Gamma", "Theta", "K", "alpha")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fd_weights(nodes: np.ndarray, target: float, derivative: int) -> np.ndarray:
    """Polynomial-exact finite-difference weights."""
    degree = len(nodes)
    matrix = np.vstack([(nodes - target) ** power for power in range(degree)])
    rhs = np.zeros(degree)
    rhs[derivative] = math.factorial(derivative)
    return np.linalg.solve(matrix, rhs)


def centered_half_plane(n: int, order: int, derivative: int,
                        parity: int, h: float) -> np.ndarray:
    radius = order // 2
    matrix = np.zeros((n, n))
    offsets = np.arange(-radius, radius + 1)
    centered = fd_weights(offsets.astype(float), 0.0, derivative) / h ** derivative
    for row in range(n):
        for offset, weight in zip(offsets, centered, strict=True):
            source = row + int(offset)
            sign = 1.0
            if source < 0:
                source = -source - 1
                sign = float(parity)
            elif source >= n:
                # One common homogeneous Dirichlet reflection at the remote
                # outer face.  It prevents an unrelated one-sided outer row
                # from dominating the axis-closure comparison.
                source = 2 * n - source - 1
                sign = -1.0
            matrix[row, source] += sign * weight
    return matrix


def regular_coefficient_row(order: int, target_layer: int, h: float) -> np.ndarray:
    nghost = order // 2 + 1
    nodes = ((np.arange(nghost) + 0.5) * h) ** 2
    target = ((target_layer + 0.5) * h) ** 2
    return fd_weights(nodes, target, 1)


def scalar_suppressed_operator(n: int, order: int, h: float,
                               policy: str) -> np.ndarray:
    """Discrete d_y^2 f = (d_rho f)/rho for an even scalar."""
    nghost = order // 2 + 1
    first = centered_half_plane(n, order, 1, +1, h)
    rho = (np.arange(n) + 0.5) * h
    result = first / rho[:, None]
    if policy == "parity_centered":
        return result
    fit_layers = nghost if policy == "legacy_fitted" else nghost - 1
    require(policy in ("legacy_fitted", "direct_regular"), f"unknown policy {policy}")
    for layer in range(fit_layers):
        result[layer, :] = 0.0
        result[layer, :nghost] = 2.0 * regular_coefficient_row(order, layer, h)
    return result


def coupled_operator(n: int, order: int, h: float, policy: str) -> np.ndarray:
    """Six-field radial proxy for metric/Ricci/A/Gamma/Theta/K/lapse feedback."""
    d_even = centered_half_plane(n, order, 1, +1, h)
    d_odd = centered_half_plane(n, order, 1, -1, h)
    d2_even = centered_half_plane(n, order, 2, +1, h)
    radial_laplacian = d2_even + scalar_suppressed_operator(n, order, h, policy)
    identity = np.eye(n)
    zero = np.zeros((n, n))

    blocks = [[zero.copy() for _ in FIELDS] for _ in FIELDS]
    # Three damped wave pairs with one-way derivative couplings mimic the
    # metric -> A -> Gamma/Theta -> K/lapse causal chain while retaining a
    # stable continuum proxy.  The triangular couplings make nonnormal growth
    # visible without changing continuum eigenvalues by construction.
    damping = 0.15 / h
    # In the first-order-in-time state used below, h*A, h*Theta, and h*alpha
    # are the dimensionless partner variables.  Scaling the triangular
    # derivative coupling as 1/h keeps h*A_h similar across resolutions and
    # makes the reported CFL quantities genuinely scale-free.
    coupling = 0.1 / h
    blocks[0][1] = identity
    blocks[1][0] = radial_laplacian
    blocks[1][1] = -damping * identity
    blocks[1][2] = coupling * d_odd
    blocks[2][3] = identity
    blocks[3][2] = radial_laplacian
    blocks[3][3] = -damping * identity
    blocks[3][4] = coupling * d_even
    blocks[4][5] = identity
    blocks[5][4] = radial_laplacian
    blocks[5][5] = -damping * identity
    physical = np.block(blocks)

    # Similarity-scale derivative-valued variables so h*A has a meaningful
    # CFL comparison against dimensionless g and alpha.
    component_scale = np.array((1.0, h, 1.0, h, 1.0, h))
    scale = np.repeat(component_scale, n)
    return scale[:, None] * physical / scale[None, :]


def rk4(matrix: np.ndarray, dt: float) -> np.ndarray:
    x = dt * matrix
    identity = np.eye(matrix.shape[0])
    return identity + x + x @ x / 2.0 + x @ x @ x / 6.0 + x @ x @ x @ x / 24.0


def mode_summary(vector: np.ndarray, n: int, radius: int) -> dict[str, object]:
    energy = np.abs(vector.reshape(len(FIELDS), n)) ** 2
    total = float(np.sum(energy))
    by_field = np.sum(energy, axis=1) / total
    by_cell = np.sum(energy, axis=0) / total
    return {
        "axis_layers_fraction": float(np.sum(by_cell[:radius])),
        "double_axis_layers_fraction": float(np.sum(by_cell[:2 * radius])),
        "peak_radial_layer": int(np.argmax(by_cell)),
        "dominant_field": FIELDS[int(np.argmax(by_field))],
        "field_fractions": {name: float(value) for name, value in zip(FIELDS, by_field)},
    }


def analyze_matrix(matrix: np.ndarray, n: int, order: int,
                   h: float) -> dict[str, object]:
    eigenvalues, eigenvectors = scipy.linalg.eig(matrix)
    hermitian = 0.5 * (matrix + matrix.T)
    numerical_abscissa = float(np.max(scipy.linalg.eigvalsh(hermitian)))
    condition = float(np.linalg.cond(eigenvectors))

    transient = []
    best_sigma = -1.0
    best_vector = None
    best_time = 0.0
    for scaled_time in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0):
        propagator = scipy.linalg.expm((scaled_time * h) * matrix)
        _, singular, vh = scipy.linalg.svd(propagator, full_matrices=False)
        sigma = float(singular[0])
        transient.append({"t_over_h": scaled_time, "norm2": sigma})
        if sigma > best_sigma:
            best_sigma = sigma
            best_time = scaled_time
            best_vector = vh[0]

    amplification = []
    for cfl in (0.025, 0.05, 0.1, 0.2, 0.25):
        update = rk4(matrix, cfl * h)
        singular = scipy.linalg.svdvals(update)
        amplification.append({
            "cfl": cfl,
            "spectral_radius": float(np.max(np.abs(scipy.linalg.eigvals(update)))),
            "norm2": float(singular[0]),
        })

    require(best_vector is not None, "transient analysis produced no mode")
    return {
        "max_real_eigenvalue_times_h": float(np.max(eigenvalues.real) * h),
        "max_abs_imag_eigenvalue_times_h": float(np.max(np.abs(eigenvalues.imag)) * h),
        "numerical_abscissa_times_h": numerical_abscissa * h,
        "eigenvector_condition": condition,
        "transient": transient,
        "rk4": amplification,
        "max_transient_t_over_h": best_time,
        "max_transient_mode": mode_summary(best_vector, n, order // 2 + 1),
    }


def verify_rows() -> None:
    for order in (2, 4, 6):
        nghost = order // 2 + 1
        for layer in range(nghost):
            weights = regular_coefficient_row(order, layer, 1.0)
            nodes = (np.arange(nghost) + 0.5) ** 2
            target = (layer + 0.5) ** 2
            for degree in range(nghost):
                expected = 0.0 if degree == 0 else degree * target ** (degree - 1)
                observed = float(weights @ (nodes ** degree))
                require(math.isclose(observed, expected, rel_tol=2e-12, abs_tol=2e-12),
                        f"O{order} regular row lost polynomial exactness")


def summarize_and_verify(cases: dict[str, object]) -> dict[str, object]:
    """Apply prospective, scale-free checks to the representative proxy."""
    verdict: dict[str, object] = {}
    for order in (2, 4, 6):
        order_result: dict[str, object] = {}
        for policy in ("legacy_fitted", "parity_centered", "direct_regular"):
            scaled_cases = [cases[f"o{order}_h{h:g}_{policy}"]
                            for h in (1.0, 0.5, 0.25)]
            reference = scaled_cases[0]
            for current in scaled_cases[1:]:
                for metric in ("max_real_eigenvalue_times_h",
                               "max_abs_imag_eigenvalue_times_h",
                               "numerical_abscissa_times_h"):
                    require(math.isclose(current[metric], reference[metric],
                                         rel_tol=2e-10, abs_tol=2e-10),
                            f"O{order} {policy} lost h-scaled invariance in {metric}")
                for reference_rk, current_rk in zip(reference["rk4"], current["rk4"],
                                                    strict=True):
                    require(reference_rk["cfl"] == current_rk["cfl"],
                            f"O{order} {policy} changed the CFL inventory")
                    require(math.isclose(current_rk["spectral_radius"],
                                         reference_rk["spectral_radius"],
                                         rel_tol=2e-10, abs_tol=2e-10),
                            f"O{order} {policy} lost h-scaled RK4 invariance")

            max_real = float(reference["max_real_eigenvalue_times_h"])
            max_transient = max(float(item["norm2"])
                                for item in reference["transient"])
            max_rk4_radius = max(float(item["spectral_radius"])
                                 for item in reference["rk4"])
            order_result[policy] = {
                "semidiscrete_nonpositive_real_part": max_real <= 2e-10,
                "sampled_rk4_spectral_radius_at_most_one": max_rk4_radius <= 1.0 + 2e-10,
                "max_sampled_transient_norm2": max_transient,
            }

        legacy = order_result["legacy_fitted"]
        for policy in ("parity_centered", "direct_regular"):
            current = order_result[policy]
            require(current["semidiscrete_nonpositive_real_part"],
                    f"O{order} {policy} proxy has a positive-real eigenvalue")
            require(current["sampled_rk4_spectral_radius_at_most_one"],
                    f"O{order} {policy} proxy violates the sampled RK4 gate")
            require(current["max_sampled_transient_norm2"] <
                    legacy["max_sampled_transient_norm2"],
                    f"O{order} {policy} proxy did not reduce sampled transient growth")
        if order >= 4:
            require(not legacy["semidiscrete_nonpositive_real_part"],
                    f"legacy O{order} proxy unexpectedly lost its unstable axis mode")
            require(not legacy["sampled_rk4_spectral_radius_at_most_one"],
                    f"legacy O{order} proxy unexpectedly passed the sampled RK4 gate")
        verdict[f"o{order}"] = order_result
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nradial", type=int, default=12)
    args = parser.parse_args()
    require(args.nradial >= 10, "nradial must resolve the O6 parity-centered bulk stencil")
    verify_rows()

    derivative_path = args.source_dir / "src/z4c/cartoon_derivatives.hpp"
    source = derivative_path.read_bytes()
    source_text = source.decode("utf-8")
    for token in ("return ActiveFirst(RhoDirection(), field) / rho_;",
                  "const Real radial_derivative = ActiveFirst(RhoDirection(), component, field);",
                  "const Real radial_derivative = ActiveFirst(RhoDirection(), a, b, field);"):
        require(token in source_text, f"production derivative source omitted {token!r}")
    for token in ("NearAxisCell", "RegularCoefficientDerivative",
                  "EvenCoefficientDerivative", "OddCoefficientDerivative",
                  "QuadraticCoefficientDerivative",
                  "QuadraticDifferenceCoefficientDerivative"):
        require(token not in source_text,
                f"production derivative source retains special closure {token!r}")

    cases: dict[str, object] = {}
    for order in (2, 4, 6):
        for h in (1.0, 0.5, 0.25):
            for policy in ("legacy_fitted", "parity_centered", "direct_regular"):
                key = f"o{order}_h{h:g}_{policy}"
                matrix = coupled_operator(args.nradial, order, h, policy)
                require(np.all(np.isfinite(matrix)), f"{key} matrix is nonfinite")
                cases[key] = analyze_matrix(matrix, args.nradial, order, h)

    verdict = summarize_and_verify(cases)

    result = {
        "schema": 1,
        "claim_scope": "representative_frozen_radial_linearization_not_nonlinear_proof",
        "source": {
            "path": "src/z4c/cartoon_derivatives.hpp",
            "sha256": hashlib.sha256(source).hexdigest(),
        },
        "fields": list(FIELDS),
        "nradial": args.nradial,
        "policies": {
            "legacy_fitted": "side-local even-in-rho polynomial fit through every closure row",
            "parity_centered": "centered parity extension with raw analytic rho quotient",
            "direct_regular": "historical parity extension plus fixed regular-coefficient rows",
            "production": "parity_centered",
        },
        "cases": cases,
        "prospective_proxy_verdict": verdict,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print("Frozen Cartoon radial operator analysis passed", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
