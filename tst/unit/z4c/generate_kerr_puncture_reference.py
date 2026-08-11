#!/usr/bin/env python3
"""Generate independent high-precision Kerr-puncture point references."""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import json
from pathlib import Path
import sys

getcontext().prec = 80
D = Decimal
ZERO = D(0)
ONE = D(1)
TWO = D(2)
THREE = D(3)
FOUR = D(4)


def cube_root(value: Decimal) -> Decimal:
    if value == 0:
        return ZERO
    estimate = D(str(float(value) ** (1.0 / 3.0)))
    for _ in range(40):
        estimate = (TWO * estimate + value / (estimate * estimate)) / THREE
    return +estimate


def determinant(matrix: list[list[Decimal]]) -> Decimal:
    return (
        matrix[0][0] * matrix[1][1] * matrix[2][2]
        + TWO * matrix[0][1] * matrix[0][2] * matrix[1][2]
        - matrix[0][0] * matrix[1][2] ** 2
        - matrix[1][1] * matrix[0][2] ** 2
        - matrix[2][2] * matrix[0][1] ** 2
    )


def inverse(matrix: list[list[Decimal]]) -> list[list[Decimal]]:
    det = determinant(matrix)
    return [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] ** 2) / det,
            (matrix[0][2] * matrix[1][2] - matrix[0][1] * matrix[2][2]) / det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / det,
        ],
        [
            (matrix[0][2] * matrix[1][2] - matrix[0][1] * matrix[2][2]) / det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] ** 2) / det,
            (matrix[0][1] * matrix[0][2] - matrix[0][0] * matrix[1][2]) / det,
        ],
        [
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / det,
            (matrix[0][1] * matrix[0][2] - matrix[0][0] * matrix[1][2]) / det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2) / det,
        ],
    ]


def packed(matrix: list[list[Decimal]]) -> list[Decimal]:
    return [matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][1], matrix[1][2], matrix[2][2]]


def permute_matrix(matrix: list[list[Decimal]], mapping: list[int]) \
        -> list[list[Decimal]]:
    return [[matrix[mapping[i]][mapping[j]] for j in range(3)] for i in range(3)]


def puncture_limit(horizon: Decimal, gauge: str) -> dict[str, object]:
    # From Eq. (11), r_BL ~ c^2/r with c=r_+/4.  Substitution in Eq. (13)
    # gives gamma_ij ~ (c^4/r^4) delta_ij, so det(gamma) has coefficient
    # c^12 and chi=det(gamma)^(-1/3) ~ r^4/c^4.  Thus the conformal metric
    # limit is obtained from the leading coefficients below, not inserted as
    # a special-case guess.  Equations (14),(15) give K_cartesian=O(r^-1),
    # while K=0 exactly, hence conformal A_ij=chi K_ij=O(r^3)->0.
    leading_metric = horizon ** 4
    leading_determinant = leading_metric ** 3
    leading_chi_denominator = cube_root(leading_determinant)
    conformal_diagonal = leading_metric / leading_chi_denominator
    identity_limit = [[conformal_diagonal, ZERO, ZERO],
                      [ZERO, conformal_diagonal, ZERO],
                      [ZERO, ZERO, conformal_diagonal]]
    zeros = [[ZERO, ZERO, ZERO], [ZERO, ZERO, ZERO], [ZERO, ZERO, ZERO]]
    return {
        "lapse": ONE if gauge == "stationary" else ZERO,
        "shift": [ZERO, ZERO, ZERO],
        "psi4": ZERO,
        "spatial_metric": packed(zeros),
        "extrinsic_curvature": packed(zeros),
        "conformal_chi": ZERO,
        "conformal_metric": packed(identity_limit),
        "trace_extrinsic_curvature": ZERO,
        "conformal_tracefree_curvature": packed(zeros),
    }


def evaluate(case: dict[str, object]) -> dict[str, object]:
    mass = D(str(case["M"]))
    chi = D(str(case["chi"]))
    center = D(str(case["z_h"]))
    code = [D(str(value)) for value in case["coordinates"]]
    cartoon = case["map"] == "cartoon"
    if cartoon:
        x, y, z = code[0], code[2], code[1] - center
        component_map = [0, 2, 1]
    else:
        x, y, z = code[0], code[1], code[2] - center
        component_map = [0, 1, 2]
    spin = chi * mass
    root = (mass * mass - spin * spin).sqrt()
    r_plus = mass + root
    r_minus = mass - root
    horizon = r_plus / FOUR
    radius = (x * x + y * y + z * z).sqrt()
    puncture = radius == 0
    if puncture:
        result = {
            "valid": True,
            "physical_adm_available": False,
            "at_puncture": True,
            "isotropic_radius": ZERO,
            "boyer_lindquist_radius": ZERO,
            "r_plus": r_plus,
            "r_minus": r_minus,
            "horizon_radius": horizon,
        }
        result.update(puncture_limit(horizon, str(case["gauge"])))
        return result

    rho = (x * x + y * y).sqrt()
    sin_theta = rho / radius
    cos_theta = z / radius
    c = horizon
    r_bl = radius + TWO * c + c * c / radius
    sigma = r_bl * r_bl + spin * spin * cos_theta * cos_theta
    delta = r_bl * r_bl - TWO * mass * r_bl + spin * spin
    r2a2 = r_bl * r_bl + spin * spin
    capital_a = r2a2 * r2a2 - delta * spin * spin * sin_theta * sin_theta

    # Independent direct Jacobian transformation of covariant spherical
    # components.  The production evaluator instead uses an orthonormal
    # spherical basis.
    dr = [x / radius, y / radius, z / radius]
    dtheta = [x * z / (rho * radius * radius),
              y * z / (rho * radius * radius),
              -rho / (radius * radius)]
    dphi = [-y / (rho * rho), x / (rho * rho), ZERO]
    g_rr = sigma * (radius + c) ** 2 / (radius ** 3 * (r_bl - r_minus))
    g_tt = sigma
    g_pp = capital_a * sin_theta * sin_theta / sigma
    physical_metric = [[
        g_rr * dr[i] * dr[j]
        + g_tt * dtheta[i] * dtheta[j]
        + g_pp * dphi[i] * dphi[j]
        for j in range(3)] for i in range(3)]

    sqrt_a_sigma = (capital_a * sigma).sqrt()
    polynomial = (
        THREE * r_bl ** 4 + TWO * spin * spin * r_bl * r_bl - spin ** 4
        - spin * spin * (r_bl * r_bl - spin * spin) * sin_theta * sin_theta
    )
    k_rp = (
        mass * spin * sin_theta * sin_theta * polynomial
        / (sigma * sqrt_a_sigma)
        * (ONE + c / radius)
        / (radius * (r_bl - r_minus)).sqrt()
    )
    k_tp = (
        -TWO * spin ** 3 * mass * r_bl * cos_theta * sin_theta ** 3
        / (sigma * sqrt_a_sigma)
        * (radius - c)
        * ((r_bl - r_minus) / radius).sqrt()
    )
    physical_curvature = [[
        k_rp * (dr[i] * dphi[j] + dphi[i] * dr[j])
        + k_tp * (dtheta[i] * dphi[j] + dphi[i] * dtheta[j])
        for j in range(3)] for i in range(3)]

    metric = permute_matrix(physical_metric, component_map)
    curvature = permute_matrix(physical_curvature, component_map)
    det = determinant(metric)
    conformal_chi = ONE / cube_root(det)
    conformal_metric = [[conformal_chi * metric[i][j] for j in range(3)]
                        for i in range(3)]
    inv_metric = inverse(metric)
    trace = sum(inv_metric[i][j] * curvature[i][j]
                for i in range(3) for j in range(3))
    conformal_a = [[
        conformal_chi * (curvature[i][j] - trace * metric[i][j] / THREE)
        for j in range(3)] for i in range(3)]

    physical_shift = [ZERO, ZERO, ZERO]
    if case["gauge"] == "stationary":
        lapse = (delta * sigma / capital_a).sqrt()
        beta_phi = -TWO * mass * spin * r_bl / capital_a
        physical_shift = [-beta_phi * y, beta_phi * x, ZERO]
    else:
        lapse = conformal_chi.sqrt()
    shift = [physical_shift[index] for index in component_map]
    return {
        "valid": True,
        "physical_adm_available": True,
        "at_puncture": False,
        "isotropic_radius": radius,
        "boyer_lindquist_radius": r_bl,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "horizon_radius": horizon,
        "lapse": lapse,
        "shift": shift,
        "psi4": ONE / conformal_chi,
        "spatial_metric": packed(metric),
        "extrinsic_curvature": packed(curvature),
        "conformal_chi": conformal_chi,
        "conformal_metric": packed(conformal_metric),
        "trace_extrinsic_curvature": trace,
        "conformal_tracefree_curvature": packed(conformal_a),
    }


def serializable(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, ".60g")
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def cases() -> list[dict[str, object]]:
    return [
        {"id": "schwarzschild", "M": "2", "chi": "0", "z_h": "0",
         "map": "cartesian", "gauge": "precollapsed",
         "coordinates": ["1.3", "-0.7", "0.4"]},
        {"id": "high_spin_plus", "M": "1", "chi": "0.99", "z_h": "0.15",
         "map": "cartesian", "gauge": "stationary",
         "coordinates": ["0.8", "-0.45", "1.2"]},
        {"id": "high_spin_minus", "M": "1", "chi": "-0.99", "z_h": "0.15",
         "map": "cartesian", "gauge": "stationary",
         "coordinates": ["0.8", "-0.45", "1.2"]},
        {"id": "high_spin_precollapsed", "M": "1", "chi": "0.99",
         "z_h": "0.15", "map": "cartesian", "gauge": "precollapsed",
         "coordinates": ["0.8", "-0.45", "1.2"]},
        {"id": "cartoon_positive_rho", "M": "1", "chi": "0.8", "z_h": "0.2",
         "map": "cartoon", "gauge": "stationary",
         "coordinates": ["0.75", "0.9", "0"]},
        {"id": "cartoon_negative_rho", "M": "1", "chi": "0.8", "z_h": "0.2",
         "map": "cartoon", "gauge": "stationary",
         "coordinates": ["-0.75", "0.9", "0"]},
        {"id": "component_map_cartesian", "M": "1", "chi": "0.8", "z_h": "0.2",
         "map": "cartesian", "gauge": "stationary",
         "coordinates": ["0.75", "0", "0.9"]},
        {"id": "horizon_equator", "M": "1", "chi": "0.99", "z_h": "0",
         "map": "cartesian", "gauge": "precollapsed",
         "coordinates": ["HORIZON", "0", "0"]},
        {"id": "horizon_stationary", "M": "1", "chi": "0.99", "z_h": "0",
         "map": "cartesian", "gauge": "stationary",
         "coordinates": ["HORIZON", "0", "0"]},
        {"id": "asymptotic", "M": "1.7", "chi": "0.6", "z_h": "-0.3",
         "map": "cartesian", "gauge": "precollapsed",
         "coordinates": ["100", "-80", "60"]},
        {"id": "near_puncture_limit", "M": "1", "chi": "0.99",
         "z_h": "0.15", "map": "cartesian", "gauge": "precollapsed",
         "coordinates": ["0.00008", "-0.00004", "0.15006"]},
        {"id": "puncture_precollapsed", "M": "1", "chi": "0.99", "z_h": "0.15",
         "map": "cartesian", "gauge": "precollapsed",
         "coordinates": ["0", "0", "0.15"]},
        {"id": "puncture_stationary", "M": "1", "chi": "0.99", "z_h": "0.15",
         "map": "cartesian", "gauge": "stationary",
         "coordinates": ["0", "0", "0.15"]},
    ]


def payload() -> dict[str, object]:
    records = []
    for specification in cases():
        specification = dict(specification)
        if specification["coordinates"][0] == "HORIZON":
            mass = D(str(specification["M"]))
            chi = D(str(specification["chi"]))
            horizon = mass * (ONE + (ONE - chi * chi).sqrt()) / FOUR
            specification["coordinates"] = [format(horizon, ".70g"), "0", "0"]
        records.append({"input": specification,
                        "expected": evaluate(specification)})
    return serializable({
        "schema": "athenak_kerr_puncture_reference_v1",
        "source": "arXiv:1001.4077 equations 6,7,11,13-15",
        "arithmetic": "python_decimal_80_digits_direct_coordinate_jacobian",
        "records": records,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("kerr_puncture_reference.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(payload(), indent=2, sort_keys=True,
                          allow_nan=False) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            print(f"stale Kerr puncture reference: {args.output}", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
