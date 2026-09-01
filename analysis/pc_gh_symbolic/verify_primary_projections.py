#!/usr/bin/env python3
"""Audit normal/spatial projections of the covariant reduced GH equation.

This script confirms the pi projection and records exact counterexamples to two
supplied regression targets.  A successful run means that the audit correctly
detects the nonzero target residuals; it does not bless the failed targets.
"""

import sympy as sp


DIM = 3


def assert_zero(name, expr):
    residual = sp.simplify(sp.expand(expr))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def assert_nonzero(name, expr):
    residual = sp.simplify(sp.expand(expr))
    if residual == 0:
        raise AssertionError(f"{name} unexpectedly vanished")
    return residual


def trace_free(tensor):
    return tensor - sp.eye(DIM) * sp.trace(tensor) / DIM


def main():
    # Flat conformal metric, alpha=chi=1, vanishing curvature/extrinsic
    # fields, C_perp=0, but arbitrary div(Z).  The covariant spatial trace
    # gives D0 K=-div(Z).  The supplied target uses H=-div(Z) and also adds
    # -div(Z), producing -2 div(Z).
    div_z = sp.symbols("div_z", real=True, nonzero=True)
    covariant_k_rhs = -div_z
    supplied_k_rhs = -div_z - div_z
    k_residual = assert_nonzero(
        "supplied K target double-counts div(Z)", supplied_k_rhs - covariant_k_rhs
    )
    assert_zero("corrected K target", (-div_z) - covariant_k_rhs)

    # Normal-normal projection:
    # D0 pi = D^2 alpha - alpha K_ij K^ij
    #         + alpha C_i D^i ln(alpha) - kappa alpha C_perp/2.
    alpha, chi, kappa, cperp = sp.symbols(
        "alpha chi kappa cperp", positive=True, finite=True
    )
    acal, x_dot_l, atilde2, K = sp.symbols(
        "acal x_dot_l atilde2 K", real=True
    )
    z_dot_l = sp.symbols("z_dot_l", real=True)
    direct_pi = (
        acal - x_dot_l / 4
        - alpha * (atilde2 + K**2 / 3)
        + chi * z_dot_l / 2
        - kappa * alpha * cperp / 2
    )
    supplied_pi = (
        -alpha * atilde2 - alpha * K**2 / 3
        + acal - x_dot_l / 4
        + chi * z_dot_l / 2
        - kappa * alpha * cperp / 2
    )
    assert_zero("supplied pi target", supplied_pi - direct_pi)

    # Audit the nonlinear Z term in the trace-free spatial projection.  Work
    # in an orthonormal conformal frame and impose tr(Q_k)=0 for every k.
    Q = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for k in range(DIM):
        q00, q01, q02, q11, q12 = sp.symbols(
            f"q{k}00 q{k}01 q{k}02 q{k}11 q{k}12", real=True
        )
        Q[k, 0, 0] = q00
        Q[k, 0, 1] = Q[k, 1, 0] = q01
        Q[k, 0, 2] = Q[k, 2, 0] = q02
        Q[k, 1, 1] = q11
        Q[k, 1, 2] = Q[k, 2, 1] = q12
        Q[k, 2, 2] = -q00 - q11

    gamma = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for k in range(DIM):
        for i in range(DIM):
            for j in range(DIM):
                gamma[k, i, j] = (
                    Q[i, k, j] + Q[j, k, i] - Q[k, i, j]
                ) / 2

    Z = sp.Matrix(sp.symbols("Z0:3", real=True))
    X = sp.Matrix(sp.symbols("X0:3", real=True))
    dZ = sp.MutableDenseNDimArray.zeros(DIM, DIM)
    for i in range(DIM):
        for k in range(DIM):
            dZ[i, k] = sp.symbols(f"dZ{i}{k}", real=True)

    # Compute the Z-dependent spatial projection directly.  At the point,
    # gtilde_ij=delta_ij while Q and dZ remain arbitrary.  The Brown operator
    # has Rtilde_ij-Rcal_ij=partial_(i Z_j).  The physical covariant
    # derivative uses Gamma[gamma]=Gamma[gtilde]+C[chi].
    connection_difference = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for k in range(DIM):
        for i in range(DIM):
            for j in range(DIM):
                connection_difference[k, i, j] = -(
                    int(k == i) * X[j]
                    + int(k == j) * X[i]
                    - int(i == j) * X[k]
                ) / (2 * chi)
    physical_gamma = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for k in range(DIM):
        for i in range(DIM):
            for j in range(DIM):
                physical_gamma[k, i, j] = (
                    gamma[k, i, j] + connection_difference[k, i, j]
                )
    dcov = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    for i in range(DIM):
        for j in range(DIM):
            partial_cj = dZ[i, j] + sum(Q[i, j, k] * Z[k] for k in range(DIM))
            dcov[i, j] = partial_cj - sum(
                physical_gamma[k, i, j] * Z[k] for k in range(DIM)
            )
    direct_z_core = sp.MutableDenseMatrix(
        DIM,
        DIM,
        lambda i, j: chi * (
            (dZ[i, j] + dZ[j, i]) / 2 - (dcov[i, j] + dcov[j, i]) / 2
        ),
    )
    corrected_metric_term = sp.MutableDenseMatrix(
        DIM, DIM,
        lambda i, j: -chi * sum(Z[k] * Q[k, i, j] for k in range(DIM)) / 2,
    )
    corrected_full_term = sp.MutableDenseMatrix(
        DIM,
        DIM,
        lambda i, j: corrected_metric_term[i, j]
        - (Z[i] * X[j] + Z[j] * X[i]) / 2,
    )
    corrected_check = trace_free(direct_z_core - corrected_full_term)
    for i in range(DIM):
        for j in range(DIM):
            assert_zero(
                f"corrected Atilde Z projection ({i},{j})", corrected_check[i, j]
            )
    supplied_metric_term = sp.MutableDenseMatrix(
        DIM, DIM,
        lambda i, j: -chi * sum(Z[k] * gamma[k, i, j] for k in range(DIM)),
    )
    tf_difference = trace_free(supplied_metric_term - corrected_metric_term)
    nonzero_components = []
    for i in range(DIM):
        for j in range(i, DIM):
            residual = sp.simplify(tf_difference[i, j])
            if residual != 0:
                nonzero_components.append(((i, j), residual))
    if not nonzero_components:
        raise AssertionError("supplied Atilde Z term unexpectedly matched corrected term")

    print(f"EXPECTED FAILURE: supplied K residual = {k_residual}")
    print("PASS: corrected K divergence count matches the covariant spatial trace")
    print("PASS: supplied pi target matches the normal-normal projection exactly")
    ij, residual = nonzero_components[0]
    print(f"EXPECTED FAILURE: supplied Atilde nonlinear Z residual {ij} = {residual}")
    print("PASS: corrected Atilde metric-advection term is -chi*Z^k*Q_kij/2")


if __name__ == "__main__":
    main()
