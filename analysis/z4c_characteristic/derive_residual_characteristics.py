#!/usr/bin/env python3
"""Derive and verify AthenaK's frozen residual-Z4c normal symbol.

This script deliberately does not import characteristic formulae from the
literature.  The matrices below are transcribed term by term from
src/z4c/z4c_calcrhs.cpp after linearizing the full-minus-background equations
at zero residual and transforming the conformal background metric to an
orthonormal face frame.

Requires SymPy.  A successful run prints the characteristic polynomials and
finishes with "all residual-characteristic checks passed".
"""

import sympy as sp


C, L, G, N = sp.symbols("C L G N", positive=True, finite=True)
BF, BB = sp.symbols("BF BB", real=True, finite=True)


def scalar_symbol():
    """Return M_S for U_S=(k,theta,A_nn,Gamma_n,dchi,dg_nn,da,db_n)."""
    matrix = sp.zeros(8)
    matrix[0, 6] = -C
    matrix[1, 3] = sp.Rational(1, 2) * N * C
    matrix[1, 4] = N
    matrix[2, 3] = sp.Rational(2, 3) * N * C
    matrix[2, 4] = sp.Rational(1, 3) * N
    matrix[2, 5] = -sp.Rational(1, 2) * N * C
    matrix[2, 6] = -sp.Rational(2, 3) * C
    matrix[3, 0] = -sp.Rational(4, 3) * N
    matrix[3, 1] = -sp.Rational(2, 3) * N
    matrix[3, 7] = sp.Rational(4, 3)
    matrix[4, 0] = sp.Rational(2, 3) * C * N
    matrix[4, 1] = sp.Rational(4, 3) * C * N
    matrix[4, 7] = -sp.Rational(2, 3) * C
    matrix[5, 2] = -2 * N
    matrix[5, 7] = sp.Rational(4, 3)
    matrix[6, 0] = -L
    matrix[7, 3] = G
    return matrix


def vector_symbol():
    """Return M_V for U_V=(A_nA,Gamma_A,dg_nA,db_A)."""
    return sp.Matrix(
        [
            [0, sp.Rational(1, 2) * N * C,
             -sp.Rational(1, 2) * N * C, 0],
            [0, 0, 0, 1],
            [-2 * N, 0, 0, 1],
            [0, G, 0, 0],
        ]
    )


def tensor_symbol():
    """Return M_T for U_T=(A_AB^TF,dg_AB^TF)."""
    return sp.Matrix([[0, -sp.Rational(1, 2) * N * C], [-2 * N, 0]])


def finite_residual_symbols():
    """Return the exact finite-residual blocks with full/background advection.

    The geometric variables are advected by the reconstructed full shift BF,
    whereas the background-adapted lapse and shift residuals are advected by
    the analytic-background shift BB.  These matrices are an audit of the
    nonlinear scope.  The closed-form rows below diagonalize these matrices
    without importing formulae from the Z4c boundary-condition literature.
    """
    scalar = scalar_symbol()
    for index in range(8):
        scalar[index, index] = BF
    scalar[6, 6] = BB
    scalar[7, 7] = BB

    vector = vector_symbol()
    for index in range(4):
        vector[index, index] = BF
    vector[3, 3] = BB

    tensor = tensor_symbol() + BF * sp.eye(2)
    return scalar, vector, tensor


def hybrid_root(q, sign):
    """Root of (lambda-BF)(lambda-BB)=q."""
    return (
        BF + BB + sign * sp.sqrt((BF - BB) ** 2 + 4 * q)
    ) / 2


def finite_scalar_left_rows():
    """Exact finite-residual scalar rows for both coordinate-speed signs."""
    rows = []
    root_c = sp.sqrt(C)
    for sign, suffix in ((1, "+"), (-1, "-")):
        lapse = hybrid_root(C * L, sign)
        lapse_delta_bg = lapse - BB
        lapse_row = sp.Matrix(
            [[-lapse_delta_bg / C, 0, 0, 0, 0, 0, 1, 0]]
        )

        shift = hybrid_root(sp.Rational(4, 3) * G, sign)
        shift_mu = shift - BF
        shift_delta_bg = shift - BB
        lapse_separation = C * L - shift_mu * shift_delta_bg
        light_separation = C * N**2 - shift_mu**2
        shift_row = sp.Matrix(
            [[
                N * shift_delta_bg**2 / lapse_separation,
                N * shift_mu * shift_delta_bg /
                    (2 * light_separation),
                0,
                shift_delta_bg * (4 * C * N**2 - 3 * shift_mu**2) /
                    (4 * light_separation),
                N**2 * shift_delta_bg / (2 * light_separation),
                0,
                -C * N * shift_delta_bg / lapse_separation,
                1,
            ]]
        )

        light = BF + sign * N * root_c
        light_one = sp.Matrix(
            [[0, sign * root_c, 0, C / 2, 1, 0, 0, 0]]
        )
        light_two = sp.Matrix(
            [[
                sign * 4 / (3 * root_c),
                sign * 2 / (3 * root_c),
                -sign * 2 / root_c,
                -1, 0, 1, 0, 0,
            ]]
        )
        rows.extend(
            [
                ("lapse" + suffix, lapse, lapse_row),
                ("shift" + suffix, shift, shift_row),
                ("light1" + suffix, light, light_one),
                ("light2" + suffix, light, light_two),
            ]
        )
    return rows


def finite_vector_left_rows():
    rows = []
    root_c = sp.sqrt(C)
    for sign, suffix in ((1, "+"), (-1, "-")):
        shift = hybrid_root(G, sign)
        light = BF + sign * N * root_c
        rows.extend(
            [
                (
                    "shift" + suffix,
                    shift,
                    sp.Matrix([[0, shift - BB, 0, 1]]),
                ),
                (
                    "light" + suffix,
                    light,
                    sp.Matrix([[-sign * 2 / root_c, -1, 1, 0]]),
                ),
            ]
        )
    return rows


def finite_tensor_left_rows():
    root_c = sp.sqrt(C)
    return [
        (
            "light+",
            BF + N * root_c,
            sp.Matrix([[-2 / root_c, 1]]),
        ),
        (
            "light-",
            BF - N * root_c,
            sp.Matrix([[2 / root_c, 1]]),
        ),
    ]


def scalar_left_rows():
    """Closed-form rows used by the implementation, for both speed signs."""
    root_c = sp.sqrt(C)
    root_g = sp.sqrt(G)
    root_l_over_c = sp.sqrt(L) / root_c
    root_three_g = sp.sqrt(3) * root_g

    lapse_plus = sp.Matrix(
        [[-root_l_over_c, 0, 0, 0, 0, 0, 1, 0]]
    )
    lapse_minus = sp.Matrix(
        [[root_l_over_c, 0, 0, 0, 0, 0, 1, 0]]
    )
    shift_plus = sp.Matrix(
        [[
            4 * G * N / (3 * C * L - 4 * G),
            2 * G * N / (3 * C * N**2 - 4 * G),
            0,
            2 * root_three_g * (C * N**2 - G) /
                (3 * C * N**2 - 4 * G),
            root_three_g * N**2 / (3 * C * N**2 - 4 * G),
            0,
            -2 * root_three_g * C * N / (3 * C * L - 4 * G),
            1,
        ]]
    )
    shift_minus = sp.Matrix(
        [[
            4 * G * N / (3 * C * L - 4 * G),
            2 * G * N / (3 * C * N**2 - 4 * G),
            0,
            -2 * root_three_g * (C * N**2 - G) /
                (3 * C * N**2 - 4 * G),
            -root_three_g * N**2 / (3 * C * N**2 - 4 * G),
            0,
            2 * root_three_g * C * N / (3 * C * L - 4 * G),
            1,
        ]]
    )
    light_one_plus = sp.Matrix(
        [[0, root_c, 0, C / 2, 1, 0, 0, 0]]
    )
    light_one_minus = sp.Matrix(
        [[0, -root_c, 0, C / 2, 1, 0, 0, 0]]
    )
    light_two_plus = sp.Matrix(
        [[4 / (3 * root_c), 2 / (3 * root_c), -2 / root_c,
          -1, 0, 1, 0, 0]]
    )
    light_two_minus = sp.Matrix(
        [[-4 / (3 * root_c), -2 / (3 * root_c), 2 / root_c,
          -1, 0, 1, 0, 0]]
    )

    return [
        ("lapse+", sp.sqrt(C * L), lapse_plus),
        ("shift+", sp.sqrt(sp.Rational(4, 3) * G), shift_plus),
        ("light1+", N * root_c, light_one_plus),
        ("light2+", N * root_c, light_two_plus),
        ("lapse-", -sp.sqrt(C * L), lapse_minus),
        ("shift-", -sp.sqrt(sp.Rational(4, 3) * G), shift_minus),
        ("light1-", -N * root_c, light_one_minus),
        ("light2-", -N * root_c, light_two_minus),
    ]


def vector_left_rows():
    root_c = sp.sqrt(C)
    return [
        ("shift+", sp.sqrt(G), sp.Matrix([[0, sp.sqrt(G), 0, 1]])),
        ("light+", N * root_c,
         sp.Matrix([[-2 / root_c, -1, 1, 0]])),
        ("shift-", -sp.sqrt(G), sp.Matrix([[0, -sp.sqrt(G), 0, 1]])),
        ("light-", -N * root_c,
         sp.Matrix([[2 / root_c, -1, 1, 0]])),
    ]


def tensor_left_rows():
    root_c = sp.sqrt(C)
    return [
        ("light+", N * root_c, sp.Matrix([[-2 / root_c, 1]])),
        ("light-", -N * root_c, sp.Matrix([[2 / root_c, 1]])),
    ]


def assert_zero(matrix, label):
    simplified = matrix.applyfunc(lambda entry: sp.simplify(sp.factor(entry)))
    if any(entry != 0 for entry in simplified):
        raise AssertionError(f"{label} is nonzero:\n{simplified}")


def check_left_rows(matrix, rows, label):
    for name, speed, left in rows:
        assert_zero(left * matrix - speed * left, f"{label} {name}")


def check_projectors(
    matrix, rows, substitutions, incoming_count, label
):
    numeric_matrix = matrix.subs(substitutions)
    left = sp.Matrix.vstack(*(row.subs(substitutions) for _, _, row in rows))
    if left.det() == 0:
        raise AssertionError(f"{label}: singular left eigenbasis at {substitutions}")
    right = left.inv()
    assert_zero(left * right - sp.eye(matrix.rows), f"{label} LR-I")

    incoming = right[:, :incoming_count] * left[:incoming_count, :]
    assert_zero(incoming * incoming - incoming, f"{label} P_in^2-P_in")
    assert_zero(
        left[incoming_count:, :] * incoming,
        f"{label} incoming projector changes outgoing modes",
    )
    assert_zero(
        numeric_matrix * right - right * sp.diag(
            *(speed.subs(substitutions) for _, speed, _ in rows)
        ),
        f"{label} right eigenpairs",
    )


def check_incoming_scalar_rhs_map(substitutions):
    """Check that the four incoming rows determine the four corrected RHSs."""
    rows = scalar_left_rows()[:4]
    left_p = sp.Matrix.vstack(
        *(row[:, :4].subs(substitutions) for _, _, row in rows)
    )
    if left_p.det() == 0:
        raise AssertionError(
            "incoming scalar p-map is singular at "
            f"{substitutions}: det={left_p.det()}"
        )


def check_constraint_characteristic_identities():
    """Verify that the light-speed rows are incoming Z4 constraints.

    Rows act on normal derivatives of
      U_S=(k,theta,A_ss,Gamma_s,dchi,dg_ss,da,db_s)
    and U_V=(A_sA,Gamma_A,dg_sA,db_A).
    """
    root_c = sp.sqrt(C)
    hamiltonian = sp.Matrix(
        [[0, 0, 0, C, 2, 0, 0, 0]]
    )
    theta_gradient = sp.Matrix(
        [[0, 1, 0, 0, 0, 0, 0, 0]]
    )
    momentum_s = sp.Matrix(
        [[-sp.Rational(2, 3), -sp.Rational(4, 3), 1, 0, 0, 0, 0, 0]]
    )
    z_s_gradient = sp.Matrix(
        [[0, 0, 0, sp.Rational(1, 2), 0,
          -sp.Rational(1, 2), 0, 0]]
    )
    scalar_rows = scalar_left_rows()
    light_one = scalar_rows[2][2]
    light_two = scalar_rows[3][2]
    assert_zero(
        2 * light_one - hamiltonian - 2 * root_c * theta_gradient,
        "first scalar constraint identity",
    )
    assert_zero(
        light_two
        + (2 / root_c) * (momentum_s + theta_gradient)
        + 2 * z_s_gradient,
        "second scalar constraint identity",
    )

    momentum_a = sp.Matrix([[1, 0, 0, 0]])
    z_a_gradient = sp.Matrix(
        [[0, sp.Rational(1, 2), -sp.Rational(1, 2), 0]]
    )
    vector_light = vector_left_rows()[1][2]
    assert_zero(
        vector_light + (2 / root_c) * momentum_a + 2 * z_a_gradient,
        "vector constraint identity",
    )


def main():
    scalar = scalar_symbol()
    vector = vector_symbol()
    tensor = tensor_symbol()
    lam = sp.symbols("lambda")

    scalar_polynomial = sp.factor(scalar.charpoly(lam).as_expr())
    vector_polynomial = sp.factor(vector.charpoly(lam).as_expr())
    tensor_polynomial = sp.factor(tensor.charpoly(lam).as_expr())
    expected_scalar = (
        (lam**2 - C * L)
        * (lam**2 - sp.Rational(4, 3) * G)
        * (lam**2 - C * N**2) ** 2
    )
    expected_vector = (lam**2 - G) * (lam**2 - C * N**2)
    expected_tensor = lam**2 - C * N**2
    if sp.factor(scalar_polynomial - expected_scalar) != 0:
        raise AssertionError("wrong scalar characteristic polynomial")
    if sp.factor(vector_polynomial - expected_vector) != 0:
        raise AssertionError("wrong vector characteristic polynomial")
    if sp.factor(tensor_polynomial - expected_tensor) != 0:
        raise AssertionError("wrong tensor characteristic polynomial")

    finite_scalar, finite_vector, finite_tensor = finite_residual_symbols()
    finite_scalar_polynomial = sp.factor(
        finite_scalar.charpoly(lam).as_expr())
    finite_vector_polynomial = sp.factor(
        finite_vector.charpoly(lam).as_expr())
    finite_tensor_polynomial = sp.factor(
        finite_tensor.charpoly(lam).as_expr())
    expected_finite_scalar = (
        ((lam - BF) ** 2 - C * N**2) ** 2
        * ((lam - BF) * (lam - BB) - C * L)
        * ((lam - BF) * (lam - BB) - sp.Rational(4, 3) * G)
    )
    expected_finite_vector = (
        ((lam - BF) ** 2 - C * N**2)
        * ((lam - BF) * (lam - BB) - G)
    )
    expected_finite_tensor = (lam - BF) ** 2 - C * N**2
    if sp.factor(finite_scalar_polynomial - expected_finite_scalar) != 0:
        raise AssertionError("wrong finite-residual scalar polynomial")
    if sp.factor(finite_vector_polynomial - expected_finite_vector) != 0:
        raise AssertionError("wrong finite-residual vector polynomial")
    if sp.factor(finite_tensor_polynomial - expected_finite_tensor) != 0:
        raise AssertionError("wrong finite-residual tensor polynomial")

    scalar_rows = scalar_left_rows()
    vector_rows = vector_left_rows()
    tensor_rows = tensor_left_rows()
    check_left_rows(scalar, scalar_rows, "scalar")
    check_left_rows(vector, vector_rows, "vector")
    check_left_rows(tensor, tensor_rows, "tensor")
    finite_scalar_rows = finite_scalar_left_rows()
    finite_vector_rows = finite_vector_left_rows()
    finite_tensor_rows = finite_tensor_left_rows()
    check_left_rows(
        finite_scalar, finite_scalar_rows, "finite-residual scalar")
    check_left_rows(
        finite_vector, finite_vector_rows, "finite-residual vector")
    check_left_rows(
        finite_tensor, finite_tensor_rows, "finite-residual tensor")
    check_constraint_characteristic_identities()

    # The first point is the production asymptotic gauge.  The remaining exact
    # rational points sample nontrivial lapse/chi/driver values without
    # approaching a coincident characteristic cone.
    samples = [
        {C: sp.Integer(1), L: sp.Integer(2), G: sp.Integer(1), N: sp.Integer(1)},
        {C: sp.Rational(4, 5), L: sp.Rational(7, 5),
         G: sp.Rational(3, 5), N: sp.Rational(9, 10)},
        {C: sp.Rational(11, 10), L: sp.Rational(13, 8),
         G: sp.Rational(7, 10), N: sp.Rational(6, 5)},
        {C: sp.Rational(3, 4), L: sp.Rational(9, 5),
         G: sp.Rational(11, 20), N: sp.Rational(5, 4)},
    ]
    for sample_number, sample in enumerate(samples):
        check_projectors(
            scalar, scalar_rows, sample, 4, f"scalar sample {sample_number}"
        )
        check_projectors(
            vector, vector_rows, sample, 2, f"vector sample {sample_number}"
        )
        check_projectors(
            tensor, tensor_rows, sample, 1, f"tensor sample {sample_number}"
        )
        check_incoming_scalar_rhs_map(sample)

        # Exact finite-residual eigenpairs are proved symbolically above.
        # Projector inversions with nested radical number fields are
        # needlessly expensive in SymPy; the companion numerical audit checks
        # LR-I, idempotence, and outgoing preservation on randomized states.

    print("scalar polynomial:", scalar_polynomial)
    print("vector polynomial:", vector_polynomial)
    print("tensor polynomial:", tensor_polynomial)
    print("finite-residual scalar polynomial:", finite_scalar_polynomial)
    print("finite-residual vector polynomial:", finite_vector_polynomial)
    print("finite-residual tensor polynomial:", finite_tensor_polynomial)
    print("all residual-characteristic checks passed")


if __name__ == "__main__":
    main()
