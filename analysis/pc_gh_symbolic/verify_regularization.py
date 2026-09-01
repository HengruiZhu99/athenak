#!/usr/bin/env python3
"""Exact symbolic checks for the puncture-regular conformal identities.

The identities checked here are tensorial.  We evaluate them in an orthonormal
frame for the conformal metric at one point while retaining arbitrary tensor
components and derivatives.  A tensor identity that holds for arbitrary frame
components in this frame holds in every frame.
"""

import sympy as sp


DIM = 3


def assert_zero(name, expr):
    residual = sp.simplify(sp.expand(expr))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def main():
    chi, alpha, K = sp.symbols("chi alpha K", positive=True, finite=True)
    X = sp.Matrix(sp.symbols("X0:3", real=True))
    Y = sp.Matrix(sp.symbols("Y0:3", real=True))
    W = X / sp.sqrt(chi)
    L = Y / alpha

    # Symmetric conformal covariant Hessians of chi and A=alpha**2.
    Xcov = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    Ycov = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    Rtilde = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    for i in range(DIM):
        for j in range(i, DIM):
            Xcov[i, j] = Xcov[j, i] = sp.symbols(f"Xcov{i}{j}", real=True)
            Ycov[i, j] = Ycov[j, i] = sp.symbols(f"Ycov{i}{j}", real=True)
            Rtilde[i, j] = Rtilde[j, i] = sp.symbols(f"Rt{i}{j}", real=True)

    # Acal = chi * tilde_D_i tilde_D_j alpha.  Differentiate
    # Y_j=2 alpha partial_j alpha before simplifying; no Y_i Y_j/A
    # temporary is introduced in the target form.
    tilde_hess_alpha = sp.MutableDenseMatrix(
        DIM,
        DIM,
        lambda i, j: Ycov[i, j] / (2 * alpha) - Y[i] * Y[j] / (4 * alpha**3),
    )
    Acal_direct = chi * tilde_hess_alpha
    Acal_regular = sp.MutableDenseMatrix(
        DIM,
        DIM,
        lambda i, j: chi / (2 * alpha) * (Ycov[i, j] - L[i] * L[j] / 2),
    )
    for i in range(DIM):
        for j in range(DIM):
            assert_zero(f"regular lapse Hessian ({i},{j})",
                        Acal_direct[i, j] - Acal_regular[i, j])

    # For gamma_ij=chi^{-1} gtilde_ij, the connection difference in an
    # orthonormal conformal frame is
    # C^k_ij=Gamma[gamma]^k_ij-Gamma[gtilde]^k_ij.
    C = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for k in range(DIM):
        for i in range(DIM):
            for j in range(DIM):
                C[k, i, j] = -(
                    int(k == i) * X[j]
                    + int(k == j) * X[i]
                    - int(i == j) * X[k]
                ) / (2 * chi)

    dalpha = Y / (2 * alpha)
    physical_hess = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    physical_hess_regular = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    LX = (L.dot(X))
    for i in range(DIM):
        for j in range(DIM):
            physical_hess[i, j] = tilde_hess_alpha[i, j] - sum(
                C[k, i, j] * dalpha[k] for k in range(DIM)
            )
            physical_hess_regular[i, j] = (
                Acal_regular[i, j]
                + (L[i] * X[j] + L[j] * X[i]) / 4
                - int(i == j) * LX / 4
            ) / chi
            assert_zero(f"physical lapse Hessian ({i},{j})",
                        physical_hess[i, j] - physical_hess_regular[i, j])

    # Direct conformal Ricci transformation, derived from the connection
    # difference.  This block verifies the regular scalar curvature and that
    # S_ij is the trace-free representative of
    # alpha*chi*R_ij-chi*D_iD_j alpha.
    divX = sp.trace(Xcov)
    X2 = X.dot(X)
    Rphysical = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    for i in range(DIM):
        for j in range(DIM):
            Rphysical[i, j] = (
                Rtilde[i, j]
                + (Xcov[i, j] + int(i == j) * divX) / (2 * chi)
                - (X[i] * X[j] + 3 * int(i == j) * X2) / (4 * chi**2)
            )

    scalar_direct = chi * sp.trace(Rphysical)
    scalar_regular = chi * sp.trace(Rtilde) + 2 * divX - sp.Rational(5, 2) * W.dot(W)
    assert_zero("regular physical scalar curvature", scalar_direct - scalar_regular)

    geom = alpha * chi * Rphysical - chi * physical_hess
    Sregular = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    for i in range(DIM):
        for j in range(DIM):
            Sregular[i, j] = (
                alpha * chi * Rtilde[i, j]
                + alpha * Xcov[i, j] / 2
                - alpha * W[i] * W[j] / 4
                - Acal_regular[i, j]
                - (L[i] * X[j] + L[j] * X[i]) / 4
            )
    delta = geom - Sregular
    delta_tf = delta - sp.eye(DIM) * sp.trace(delta) / DIM
    for i in range(DIM):
        for j in range(DIM):
            assert_zero(f"regular S trace-free identity ({i},{j})", delta_tf[i, j])

    # Extrinsic-curvature contraction.  Build a generic symmetric trace-free
    # conformal A in the orthonormal frame.
    A00, A01, A02, A11, A12 = sp.symbols("A00 A01 A02 A11 A12", real=True)
    A = sp.Matrix([
        [A00, A01, A02],
        [A01, A11, A12],
        [A02, A12, -A00 - A11],
    ])
    Kphysical = (A + sp.eye(DIM) * K / 3) / chi
    gamma_inv = chi * sp.eye(DIM)
    K2_direct = sp.trace(gamma_inv * Kphysical * gamma_inv * Kphysical)
    K2_regular = sp.trace(A * A) + K**2 / 3
    assert_zero("extrinsic-curvature contraction", K2_direct - K2_regular)

    hamiltonian_direct = scalar_direct + K**2 - K2_direct
    hamiltonian_regular = (
        sp.Rational(2, 3) * K**2
        - sp.trace(A * A)
        + chi * sp.trace(Rtilde)
        + 2 * divX
        - sp.Rational(5, 2) * W.dot(W)
    )
    assert_zero("regular Hamiltonian", hamiltonian_direct - hamiltonian_regular)

    # Difference between physical and conformal divergence of A^j_i.
    correction = sp.Matrix.zeros(DIM, 1)
    for i in range(DIM):
        correction[i] = sum(
            C[j, j, k] * A[k, i] - C[k, j, i] * A[j, k]
            for j in range(DIM) for k in range(DIM)
        )
        expected = -sp.Rational(3, 2) * sum(A[j, i] * X[j] for j in range(DIM)) / chi
        assert_zero(f"momentum connection correction {i}", correction[i] - expected)
        scaled_direct = sp.sqrt(chi) * correction[i]
        scaled_regular = -sp.Rational(3, 2) * sum(
            A[j, i] * W[j] for j in range(DIM)
        )
        assert_zero(f"scaled momentum correction {i}", scaled_direct - scaled_regular)

    print("PASS: exact regular Hessian identity")
    print("PASS: exact physical/conformal lapse-Hessian identity")
    print("PASS: exact regular scalar-curvature and Hamiltonian identities")
    print("PASS: exact trace-free curvature/lapse tensor identity")
    print("PASS: exact scaled momentum identity")


if __name__ == "__main__":
    main()
