#!/usr/bin/env python3
"""Arbitrary-precision lower-order coefficient audit for the q=1 trumpet.

This is deliberately independent of the generated binary64 radial-q tables.
It imports only the implicit arbitrary-precision trumpet and coordinate-jet
construction used by the high-precision source oracle, then forms the linear
maps that occur in the fully subtracted gauge equations.  In particular, the
script measures coefficients with respect to the *stored* frame variables,
not raw coordinate-metric perturbations.

The reported powers are least-squares slopes of log(max coefficient) against
log(r) over the innermost requested radii.  A negative power is a divergent
coefficient; the calculation does not decide whether a weighted energy norm
controls the associated residual.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import mpmath as mp

REPOSITORY = Path(__file__).resolve().parents[2]
ORACLE_DIRECTORY = REPOSITORY / "tst" / "test_suite" / "ref_gh"
sys.path.insert(0, str(ORACLE_DIRECTORY))
ORACLE_SPEC = importlib.util.spec_from_file_location(
    "high_precision_trumpet_source_oracle",
    ORACLE_DIRECTORY / "high_precision_trumpet_source_oracle.py",
)
if ORACLE_SPEC is None or ORACLE_SPEC.loader is None:
    raise RuntimeError("could not load the high-precision trumpet oracle")
hp = importlib.util.module_from_spec(ORACLE_SPEC)
sys.modules[ORACLE_SPEC.name] = hp
ORACLE_SPEC.loader.exec_module(hp)


SYMMETRIC_PAIRS = [(a, b) for a in range(4) for b in range(a, 4)]
ETA = [-1, 1, 1, 1]


def maximum(value):
    if isinstance(value, (list, tuple)):
        return max([maximum(item) for item in value] + [mp.mpf(0)])
    return abs(value)


def zeros(*shape):
    if len(shape) == 1:
        return [mp.mpf(0) for _ in range(shape[0])]
    return [zeros(*shape[1:]) for _ in range(shape[0])]


def add_scaled(destination, source, coefficient):
    for a in range(4):
        for b in range(4):
            destination[a][b] += coefficient * source[a][b]


def determinant3(matrix):
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def physical_target(metric, d_metric, frame, upsilon, nu, eta_beta):
    """The unchanged production advective 1+log/Gamma-driver target."""
    inverse = hp.inverse_matrix(metric)
    spatial_metric = [[metric[i + 1][j + 1] for j in range(3)]
                      for i in range(3)]
    spatial_inverse = hp.inverse_matrix(spatial_metric)
    spatial_determinant = determinant3(spatial_metric)
    lapse = 1 / mp.sqrt(-inverse[0][0])
    shift = [lapse**2 * inverse[0][i + 1] for i in range(3)]
    d_inverse = zeros(3, 4, 4)
    d_lapse = zeros(3)
    d_shift = zeros(3, 3)
    for p in range(3):
        for a in range(4):
            for b in range(4):
                d_inverse[p][a][b] = -sum(
                    inverse[a][c] * inverse[b][d]
                    * d_metric[p + 1][c][d]
                    for c in range(4) for d in range(4)
                )
        d_lapse[p] = lapse**3 * d_inverse[p][0][0] / 2
        for i in range(3):
            d_shift[p][i] = (
                2 * lapse * d_lapse[p] * inverse[0][i + 1]
                + lapse**2 * d_inverse[p][0][i + 1]
            )

    first = zeros(4, 4, 4)
    christoffel = zeros(4, 4, 4)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                first[a][b][c] = (
                    d_metric[b][a][c] + d_metric[c][a][b]
                    - d_metric[a][b][c]
                ) / 2
    for a in range(4):
        for b in range(4):
            for c in range(4):
                christoffel[a][b][c] = sum(
                    inverse[a][d] * first[d][b][c] for d in range(4)
                )
    trace_k = -lapse * sum(
        spatial_inverse[i][j] * christoffel[0][i + 1][j + 1]
        for i in range(3) for j in range(3)
    )
    determinant_factor = spatial_determinant ** (mp.mpf(1) / 3)
    conformal_gamma = zeros(3)
    for i in range(3):
        conformal_gamma[i] = determinant_factor * sum(
            (spatial_inverse[i][k] * spatial_inverse[j][ell]
             - spatial_inverse[i][j] * spatial_inverse[k][ell] / 3)
            * d_metric[j + 1][k + 1][ell + 1]
            for j in range(3) for k in range(3) for ell in range(3)
        )
    desired_dt_shift = [
        nu * (conformal_gamma[i] - eta_beta * upsilon[i])
        - sum(shift[p] * d_shift[p][i] for p in range(3))
        for i in range(3)
    ]
    coordinate = zeros(4)
    for i in range(3):
        contracted_spatial_connection = sum(
            spatial_inverse[j][k]
            * (d_metric[j + 1][i + 1][k + 1]
               + d_metric[k + 1][i + 1][j + 1]
               - d_metric[i + 1][j + 1][k + 1]) / 2
            for j in range(3) for k in range(3)
        )
        coordinate[i + 1] = (
            d_lapse[i] / lapse - contracted_spatial_connection
            + sum(metric[i + 1][j + 1] * desired_dt_shift[j] / lapse**2
                  for j in range(3))
        )
    coordinate[0] = lapse * (2 / lapse - 1) * trace_k + sum(
        shift[i] * coordinate[i + 1] for i in range(3)
    )
    projected = [sum(frame[A][a] * coordinate[a] for a in range(4))
                 for A in range(4)]
    return projected, shift, conformal_gamma


def gauge_increment_from_j(state, j, d_j, gamma0=1):
    """Project -2 nabla_(a J_b)+gamma0 projector(J) to the frame."""
    output = zeros(4, 4)
    normal_upper = [1 / state["alpha"]] + [
        -value / state["alpha"] for value in state["shift"]
    ]
    normal_lower = [-state["alpha"], mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    for A in range(4):
        for B in range(4):
            for a in range(4):
                for b in range(4):
                    value = -d_j[a][b] - d_j[b][a]
                    value += 2 * sum(
                        state["christoffel"][c][a][b] * j[c]
                        for c in range(4)
                    )
                    for c in range(4):
                        projector = (
                            (normal_lower[b] if c == a else 0)
                            + (normal_lower[a] if c == b else 0)
                            - state["metric"][a][b] * normal_upper[c]
                        )
                        value += gamma0 * projector * j[c]
                    output[A][B] += (
                        state["frame"][A][a] * state["frame"][B][b] * value
                    )
    return output


def reference_state(radius, normalization):
    metric_jet, inverse_jet, frame_jet, coframe_jet, alpha, areal = (
        hp.reference_geometry(radius, normalization)
    )
    metric, inverse, d_metric, _, christoffel, d_christoffel = (
        hp.coordinate_geometry(metric_jet, inverse_jet)
    )
    frame = [[frame_jet[A][a].value for a in range(4)] for A in range(4)]
    coframe = [[coframe_jet[A][a].value for a in range(4)] for A in range(4)]
    d_frame = [[[frame_jet[A][a].first[p] for a in range(4)]
                for A in range(4)] for p in range(4)]
    d_coframe = [[[coframe_jet[A][a].first[p] for a in range(4)]
                  for A in range(4)] for p in range(4)]
    d_inverse = [[[inverse_jet[a][b].first[p] for b in range(4)]
                  for a in range(4)] for p in range(4)]

    h_upper = [-sum(inverse[b][c] * christoffel[a][b][c]
                    for b in range(4) for c in range(4))
               for a in range(4)]
    d_h_upper = [[-sum(
        d_inverse[p][b][c] * christoffel[a][b][c]
        + inverse[b][c] * d_christoffel[p][a][b][c]
        for b in range(4) for c in range(4))
        for a in range(4)] for p in range(4)]
    h_lower = [sum(metric[a][b] * h_upper[b] for b in range(4))
               for a in range(4)]
    d_h_lower = [[sum(
        d_metric[p][a][b] * h_upper[b] + metric[a][b] * d_h_upper[p][b]
        for b in range(4)) for a in range(4)] for p in range(4)]
    h_frame = [sum(frame[A][a] * h_lower[a] for a in range(4))
               for A in range(4)]
    d_h_frame = [[sum(
        d_frame[p][A][a] * h_lower[a] + frame[A][a] * d_h_lower[p][a]
        for a in range(4)) for A in range(4)] for p in range(4)]

    omega = zeros(4, 4, 4)
    d_omega = zeros(4, 4, 4, 4)
    for p in range(4):
        for A in range(4):
            for B in range(4):
                omega[p][A][B] = sum(
                    d_frame[p][A][a] * coframe[B][a] for a in range(4)
                )
                for q in range(4):
                    d_omega[q][p][A][B] = sum(
                        frame_jet[A][a].second[q][p] * coframe[B][a]
                        + d_frame[p][A][a] * d_coframe[q][B][a]
                        for a in range(4)
                    )
    reference_k = [[
        d_h_frame[i + 1][A]
        - sum(omega[i + 1][A][B] * h_frame[B] for B in range(4))
        for A in range(4)] for i in range(3)]
    direct_reference_k = [[sum(
        frame[A][a] * d_h_lower[i + 1][a] for a in range(4)
    ) for A in range(4)] for i in range(3)]
    shift = [alpha**2 * inverse[0][i + 1] for i in range(3)]
    beta_omega = [[sum(
        shift[i] * omega[i + 1][A][B] for i in range(3))
        for B in range(4)] for A in range(4)]

    state = {
        "radius": radius,
        "alpha": alpha,
        "areal": areal,
        "metric_jet": metric_jet,
        "inverse_jet": inverse_jet,
        "frame_jet": frame_jet,
        "coframe_jet": coframe_jet,
        "metric": metric,
        "inverse": inverse,
        "d_metric": d_metric,
        "d_inverse": d_inverse,
        "frame": frame,
        "coframe": coframe,
        "d_frame": d_frame,
        "d_coframe": d_coframe,
        "christoffel": christoffel,
        "d_christoffel": d_christoffel,
        "h_upper": h_upper,
        "d_h_upper": d_h_upper,
        "h_lower": h_lower,
        "d_h_lower": d_h_lower,
        "h_frame": h_frame,
        "d_h_frame": d_h_frame,
        "omega": omega,
        "d_omega": d_omega,
        "reference_k": reference_k,
        "reference_k_identity_error": max(
            abs(reference_k[i][A] - direct_reference_k[i][A])
            for i in range(3) for A in range(4)
        ),
        "shift": shift,
        "beta_omega": beta_omega,
    }

    source_h = []
    source_d_h = [[None for _ in range(4)] for _ in range(4)]
    for C in range(4):
        j = [coframe[C][a] for a in range(4)]
        d_j = [[d_coframe[p][C][a] for a in range(4)] for p in range(4)]
        source_h.append(gauge_increment_from_j(state, j, d_j))
        for p in range(4):
            j = zeros(4)
            d_j = zeros(4, 4)
            for a in range(4):
                d_j[p][a] = coframe[C][a]
            source_d_h[p][C] = gauge_increment_from_j(state, j, d_j)
    state["source_h"] = source_h
    state["source_d_h"] = source_d_h
    return state


def stored_metric_direction(state, pair, kind, spatial_frame=None):
    """Return delta g, delta dg, and d beta for one stored-variable basis."""
    delta_psi = zeros(4, 4)
    delta_d_psi = zeros(4, 4, 4)
    A, B = pair
    if kind == "psi":
        delta_psi[A][B] = 1
        delta_psi[B][A] = 1
    elif kind == "pi":
        delta_d_psi[0][A][B] = -state["alpha"]
        delta_d_psi[0][B][A] = -state["alpha"]
    elif kind == "phi":
        if spatial_frame is None:
            raise ValueError("phi direction requires a spatial frame index")
        for p in range(3):
            value = state["coframe"][spatial_frame + 1][p + 1]
            delta_d_psi[p + 1][A][B] = value
            delta_d_psi[p + 1][B][A] = value
            delta_d_psi[0][A][B] += state["shift"][p] * value
            delta_d_psi[0][B][A] += state["shift"][p] * value
    else:
        raise ValueError(kind)
    if A == B:
        if kind == "psi":
            delta_psi[A][B] = 1
        elif kind == "phi":
            for p in range(4):
                delta_d_psi[p][A][B] /= 2

    delta_metric = zeros(4, 4)
    delta_d_metric = zeros(4, 4, 4)
    for a in range(4):
        for b in range(4):
            delta_metric[a][b] = sum(
                state["coframe"][C][a] * state["coframe"][D][b]
                * delta_psi[C][D]
                for C in range(4) for D in range(4)
            )
            for p in range(4):
                delta_d_metric[p][a][b] = sum(
                    (state["d_coframe"][p][C][a]
                     * state["coframe"][D][b]
                     + state["coframe"][C][a]
                     * state["d_coframe"][p][D][b])
                    * delta_psi[C][D]
                    + state["coframe"][C][a]
                    * state["coframe"][D][b] * delta_d_psi[p][C][D]
                    for C in range(4) for D in range(4)
                )

    delta_inverse = zeros(4, 4)
    delta_d_inverse = zeros(4, 4, 4)
    for a in range(4):
        for b in range(4):
            delta_inverse[a][b] = -sum(
                state["inverse"][a][c] * delta_metric[c][d]
                * state["inverse"][d][b]
                for c in range(4) for d in range(4)
            )
            for p in range(4):
                delta_d_inverse[p][a][b] = -sum(
                    state["d_inverse"][p][a][c] * delta_metric[c][d]
                    * state["inverse"][d][b]
                    + state["inverse"][a][c] * delta_d_metric[p][c][d]
                    * state["inverse"][d][b]
                    + state["inverse"][a][c] * delta_metric[c][d]
                    * state["d_inverse"][p][d][b]
                    for c in range(4) for d in range(4)
                )

    delta_lapse = state["alpha"]**3 * delta_inverse[0][0] / 2
    delta_shift = [
        2 * state["alpha"] * delta_lapse * state["inverse"][0][i + 1]
        + state["alpha"]**2 * delta_inverse[0][i + 1]
        for i in range(3)
    ]
    return delta_metric, delta_d_metric, delta_inverse, delta_d_inverse, delta_shift


def residual_base_direction(state, direction):
    """Linearized Delta B_a and derivative, without full-source subtraction."""
    delta_metric, delta_d_metric, delta_inverse, delta_d_inverse, _ = direction
    delta_upper = [-sum(
        delta_inverse[b][c] * state["christoffel"][a][b][c]
        for b in range(4) for c in range(4)) for a in range(4)]
    d_delta_upper = [[-sum(
        delta_d_inverse[p][b][c] * state["christoffel"][a][b][c]
        + delta_inverse[b][c] * state["d_christoffel"][p][a][b][c]
        for b in range(4) for c in range(4))
        for a in range(4)] for p in range(4)]
    delta_lower = [sum(
        delta_metric[a][b] * state["h_upper"][b]
        + state["metric"][a][b] * delta_upper[b]
        for b in range(4)) for a in range(4)]
    d_delta_lower = [[sum(
        delta_d_metric[p][a][b] * state["h_upper"][b]
        + delta_metric[a][b] * state["d_h_upper"][p][b]
        + state["d_metric"][p][a][b] * delta_upper[b]
        + state["metric"][a][b] * d_delta_upper[p][b]
        for b in range(4)) for a in range(4)] for p in range(4)]
    return delta_lower, d_delta_lower


def target_direction(state, direction, nu, eta_beta, epsilon):
    delta_metric, delta_d_metric, _, _, _ = direction
    values = []
    for sign in (-1, 1):
        metric = [[state["metric"][a][b] + sign * epsilon * delta_metric[a][b]
                   for b in range(4)] for a in range(4)]
        d_metric = [[[
            state["d_metric"][p][a][b]
            + sign * epsilon * delta_d_metric[p][a][b]
            for b in range(4)] for a in range(4)] for p in range(4)]
        values.append(physical_target(
            metric, d_metric, state["frame"], zeros(3), nu, eta_beta
        )[0])
    return [(values[1][A] - values[0][A]) / (2 * epsilon)
            for A in range(4)]


def source_from_dt_h(state, dt_h):
    output = zeros(4, 4)
    for C in range(4):
        add_scaled(output, state["source_d_h"][0][C], dt_h[C])
    return output


def coefficient_row(radius, normalization, nu, eta_beta, epsilon):
    state = reference_state(radius, normalization)
    reference_target, target_shift, reference_conformal_gamma = physical_target(
        state["metric"], state["d_metric"], state["frame"], zeros(3),
        nu, eta_beta,
    )
    row = {
        "radius": radius,
        "check_target_minus_h": max(
            abs(reference_target[A] - state["h_frame"][A])
            for A in range(4)
        ),
        "check_target_shift": max(
            abs(target_shift[i] - state["shift"][i]) for i in range(3)
        ),
        "check_conformal_gamma": maximum(reference_conformal_gamma),
        "check_reference_k_identity": state["reference_k_identity_error"],
        "alpha": state["alpha"],
        "l": state["coframe"][1][1],
        "shift": maximum(state["shift"]),
        "reference_h": maximum(state["h_frame"]),
        "reference_k": maximum(state["reference_k"]),
        "omega_spatial": maximum(state["omega"][1:]),
        "d_omega_spatial": maximum([
            [state["d_omega"][q][p] for p in range(1, 4)]
            for q in range(1, 4)
        ]),
        "beta_omega": maximum(state["beta_omega"]),
        "einstein_h_direct": maximum(state["source_h"]),
        "einstein_dt_h": maximum(state["source_d_h"][0]),
        "einstein_di_h": maximum(state["source_d_h"][1:]),
    }

    # Same-stage h coefficient at fixed metric: -mu h-beta.Omega h, mu=1.
    stage_h = []
    for C in range(4):
        output = [[state["source_h"][C][A][B] for B in range(4)]
                  for A in range(4)]
        for D in range(4):
            coefficient = -(1 if C == D else 0) - state["beta_omega"][D][C]
            add_scaled(output, state["source_d_h"][0][D], coefficient)
        stage_h.append(output)
    row["einstein_h_same_stage"] = maximum(stage_h)

    coefficient_sets = {"psi": [], "pi": [], "phi": []}
    target_sets = {"psi": [], "pi": [], "phi": []}
    direct_sets = {"psi": [], "pi": [], "phi": []}
    beta_k_sets = []
    beta_jacobian = []
    theta_beta_k = []
    for pair in SYMMETRIC_PAIRS:
        for kind in ("psi", "pi"):
            direction = stored_metric_direction(state, pair, kind)
            target = target_direction(state, direction, nu, eta_beta, epsilon)
            delta_base, d_delta_base = residual_base_direction(state, direction)
            direct = gauge_increment_from_j(
                state, [-value for value in delta_base],
                [[-value for value in line] for line in d_delta_base]
            )
            dt_h = [target[A] for A in range(4)]  # mu=1
            if kind == "psi":
                beta_k = [sum(direction[4][i] * state["reference_k"][i][A]
                              for i in range(3)) for A in range(4)]
                dt_h = [dt_h[A] + beta_k[A] for A in range(4)]
                beta_k_sets.append(beta_k)
                beta_jacobian.append(direction[4])
                theta_beta_k.append([-value for value in beta_k])  # eta=1
            complete = [[direct[A][B] for B in range(4)] for A in range(4)]
            add_scaled(complete, source_from_dt_h(state, dt_h), 1)
            coefficient_sets[kind].append(complete)
            target_sets[kind].append(target)
            direct_sets[kind].append(direct)
        for spatial_frame in range(3):
            direction = stored_metric_direction(state, pair, "phi", spatial_frame)
            target = target_direction(state, direction, nu, eta_beta, epsilon)
            delta_base, d_delta_base = residual_base_direction(state, direction)
            direct = gauge_increment_from_j(
                state, [-value for value in delta_base],
                [[-value for value in line] for line in d_delta_base]
            )
            complete = [[direct[A][B] for B in range(4)] for A in range(4)]
            add_scaled(complete, source_from_dt_h(state, target), 1)
            coefficient_sets["phi"].append(complete)
            target_sets["phi"].append(target)
            direct_sets["phi"].append(direct)

    row["delta_beta_per_psi"] = maximum(beta_jacobian)
    row["delta_beta_k_per_psi"] = maximum(beta_k_sets)
    row["theta_delta_beta_k_per_psi"] = maximum(theta_beta_k)
    for kind in ("psi", "pi", "phi"):
        row[f"target_per_{kind}"] = maximum(target_sets[kind])
        row[f"einstein_base_per_{kind}"] = maximum(direct_sets[kind])
        row[f"einstein_complete_per_{kind}"] = maximum(coefficient_sets[kind])

    # Upsilon enters only the target; differentiate it analytically with a
    # centered high-precision direction to retain an independent target path.
    upsilon_target = []
    for i in range(3):
        minus = zeros(3)
        plus = zeros(3)
        minus[i] = -epsilon
        plus[i] = epsilon
        f_minus = physical_target(
            state["metric"], state["d_metric"], state["frame"], minus,
            nu, eta_beta)[0]
        f_plus = physical_target(
            state["metric"], state["d_metric"], state["frame"], plus,
            nu, eta_beta)[0]
        upsilon_target.append([
            (f_plus[A] - f_minus[A]) / (2 * epsilon) for A in range(4)
        ])
    row["target_per_upsilon"] = maximum(upsilon_target)
    row["einstein_complete_per_upsilon"] = maximum([
        source_from_dt_h(state, target) for target in upsilon_target
    ])
    return row


def fitted_power(rows, key, fit_count):
    selected = rows[-fit_count:]
    x = [mp.log(row["radius"]) for row in selected]
    y = [mp.log(row[key]) for row in selected]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum(
        (a - x_mean)**2 for a in x
    )


def serializable(value, digits):
    if isinstance(value, dict):
        return {key: serializable(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item, digits) for item in value]
    if isinstance(value, mp.mpf):
        return mp.nstr(value, digits)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dps", type=int, default=90)
    parser.add_argument("--fit-count", type=int, default=6)
    parser.add_argument(
        "--difference-exponent", type=int,
        help="use epsilon=10^(-N) for centered target Jacobians",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    if args.dps < 70:
        raise ValueError("at least 70 decimal digits are required")
    mp.mp.dps = args.dps
    hp.N = mp.mpf(2)
    hp.ALPHA_C, hp.RADIUS_C, hp.C_SQUARED = hp.constants()
    hp.radius_from_alpha.zero_radius = hp.bisect(
        lambda radius: hp.implicit(mp.mpf(0), radius),
        mp.mpf(0), hp.RADIUS_C, int(mp.mp.dps * 3.6) + 32,
    )
    normalization = hp.normalization_constant()
    zero_radius = hp.radius_from_alpha.zero_radius
    radius_alpha, _ = hp.radius_alpha_derivatives(mp.mpf(0), zero_radius)
    lapse_power = zero_radius / radius_alpha
    radii = [mp.mpf(value) for value in (
        "1e-2", "3e-3", "1e-3", "3e-4", "1e-4", "3e-5", "1e-5",
        "3e-6", "1e-6",
    )]
    difference_exponent = (
        args.difference_exponent
        if args.difference_exponent is not None else args.dps // 3
    )
    epsilon = mp.mpf(10) ** (-difference_exponent)
    check_tolerance = mp.mpf(10) ** (-(args.dps // 2))
    rows = []
    for radius in radii:
        row = coefficient_row(
            radius, normalization, mp.mpf("0.75"), mp.mpf(1), epsilon
        )
        for key, value in row.items():
            if key.startswith("check_") and value > check_tolerance:
                raise AssertionError(
                    f"{key} failed at r={radius}: {value} > {check_tolerance}"
                )
        rows.append(row)
        print(f"completed r={mp.nstr(radius, 8)}", flush=True)
    check_keys = [key for key in rows[0] if key.startswith("check_")]
    keys = [key for key in rows[0]
            if key != "radius" and not key.startswith("check_")]
    powers = {key: fitted_power(rows, key, args.fit_count) for key in keys}
    p = lapse_power
    expected_powers = {
        "alpha": p,
        "l": -1,
        "shift": 1,
        "reference_h": -2 * p,
        "reference_k": -(3 * p + 1),
        "omega_spatial": -(p + 1),
        "d_omega_spatial": -(p + 2),
        "beta_omega": -p,
        "delta_beta_per_psi": p + 1,
        "delta_beta_k_per_psi": -2 * p,
        "theta_delta_beta_k_per_psi": -2 * p,
        "einstein_h_direct": 0,
        "einstein_dt_h": -p,
        "einstein_di_h": 1 - p,
        "einstein_h_same_stage": -2 * p,
        "target_per_psi": -2 * p,
        "target_per_pi": -p,
        "target_per_phi": -(2 * p + 2),
        "target_per_upsilon": -(2 * p + 1),
        "einstein_base_per_psi": -3 * p,
        "einstein_base_per_pi": -2 * p,
        "einstein_base_per_phi": -3 * p,
        "einstein_complete_per_psi": -3 * p,
        "einstein_complete_per_pi": -2 * p,
        "einstein_complete_per_phi": -(3 * p + 2),
        "einstein_complete_per_upsilon": -(3 * p + 1),
    }
    power_errors = {
        key: powers[key] - expected for key, expected in expected_powers.items()
    }
    power_tolerance = mp.mpf("0.005")
    maximum_power_error = max(abs(error) for error in power_errors.values())
    if maximum_power_error > power_tolerance:
        raise AssertionError(
            f"asymptotic power gate failed: {maximum_power_error} "
            f"> {power_tolerance}"
        )
    payload = {
        "decimal_digits": args.dps,
        "directional_difference_epsilon": epsilon,
        "normalization_constant": normalization,
        "trumpet_areal_radius": zero_radius,
        "lapse_power_p": lapse_power,
        "fit_count": args.fit_count,
        "fit_radius_min": rows[-1]["radius"],
        "fit_radius_max": rows[-args.fit_count]["radius"],
        "identity_check_tolerance": check_tolerance,
        "identity_checks": {
            key: max(row[key] for row in rows) for key in check_keys
        },
        "expected_powers": expected_powers,
        "power_errors": power_errors,
        "power_tolerance": power_tolerance,
        "maximum_power_error": maximum_power_error,
        "powers": powers,
        "rows": rows,
    }
    encoded = json.dumps(serializable(payload, 40), indent=2,
                         sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        with args.table.open("w") as stream:
            stream.write("\t".join(["radius"] + keys) + "\n")
            for row in rows:
                stream.write("\t".join(
                    mp.nstr(row[key], 18) for key in ["radius"] + keys
                ) + "\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
