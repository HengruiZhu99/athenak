#!/usr/bin/env python3
"""Independent exact component regression for the Brown conformal Ricci identity.

The script constructs a non-diagonal, coordinate-dependent, exactly unimodular
positive-definite conformal metric as L L^T with det(L)=1.  It compares the
coordinate-definition Ricci tensor with the proposed first-order Brown form after
setting Lambda^i to the contracted conformal Christoffel symbol.  Comparisons are
performed with exact rational arithmetic at several points.
"""

import sympy as sp


DIM = 3


def metric_family(coords):
    x, y, z = coords
    a = 1 + x / 7 + y / 11 + x * z / 37
    b = 1 + y / 5 + z / 13 + x * y / 41
    p = x / 17 - z / 19 + y * z / 43
    q = y / 23 + x / 29 - x * z / 47
    r = z / 31 - y / 53 + x * y / 59
    L = sp.Matrix([
        [a, 0, 0],
        [p, b, 0],
        [q, r, 1 / (a * b)],
    ])
    return sp.simplify(L * L.T)


def main():
    coords = sp.symbols("x y z", real=True)
    g = metric_family(coords)
    determinant = sp.factor(g.det())
    if determinant != 1:
        raise AssertionError(f"metric family is not unimodular: det={determinant}")
    gu = sp.simplify(g.inv())

    Q = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for k in range(DIM):
        for i in range(DIM):
            for j in range(DIM):
                Q[k, i, j] = sp.diff(g[i, j], coords[k])

    gamma_u = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    gamma_l = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
    for m in range(DIM):
        for i in range(DIM):
            for j in range(DIM):
                gamma_u[m, i, j] = sum(
                    gu[m, ell]
                    * (Q[i, ell, j] + Q[j, ell, i] - Q[ell, i, j])
                    / 2
                    for ell in range(DIM)
                )
    # First-index-lowered symbol Gamma_{ijk}=g_{in} Gamma^n_{jk}.
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                gamma_l[i, j, k] = sum(g[i, n] * gamma_u[n, j, k] for n in range(DIM))

    contracted = [
        sum(gu[k, ell] * gamma_u[m, k, ell]
            for k in range(DIM) for ell in range(DIM))
        for m in range(DIM)
    ]

    ricci_direct = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    ricci_brown = sp.MutableDenseMatrix(DIM, DIM, lambda i, j: 0)
    for i in range(DIM):
        for j in range(DIM):
            ricci_direct[i, j] = sum(
                sp.diff(gamma_u[k, i, j], coords[k])
                - sp.diff(gamma_u[k, i, k], coords[j])
                for k in range(DIM)
            ) + sum(
                gamma_u[k, i, j] * gamma_u[ell, k, ell]
                - gamma_u[ell, i, k] * gamma_u[k, j, ell]
                for k in range(DIM) for ell in range(DIM)
            )

            principal = -sum(
                gu[k, ell] * sp.diff(Q[ell, i, j], coords[k]) / 2
                for k in range(DIM) for ell in range(DIM)
            )
            quadratic = 0
            for k in range(DIM):
                for ell in range(DIM):
                    bracket = sum(
                        gamma_u[m, k, ell]
                        * (gamma_l[i, j, m] + gamma_l[j, i, m]) / 2
                        + gamma_u[m, k, i] * gamma_l[j, m, ell]
                        + gamma_u[m, k, j] * gamma_l[i, m, ell]
                        + gamma_u[m, i, k] * gamma_l[m, j, ell]
                        for m in range(DIM)
                    )
                    quadratic += gu[k, ell] * bracket
            lambda_derivative = sum(
                (
                    g[k, i] * sp.diff(contracted[k], coords[j])
                    + g[k, j] * sp.diff(contracted[k], coords[i])
                ) / 2
                for k in range(DIM)
            )
            ricci_brown[i, j] = principal + quadratic + lambda_derivative

    points = [
        {coords[0]: sp.Rational(0), coords[1]: sp.Rational(0), coords[2]: sp.Rational(0)},
        {coords[0]: sp.Rational(1, 5), coords[1]: sp.Rational(-1, 7),
         coords[2]: sp.Rational(2, 9)},
        {coords[0]: sp.Rational(-2, 11), coords[1]: sp.Rational(3, 13),
         coords[2]: sp.Rational(1, 17)},
    ]
    for pindex, point in enumerate(points):
        for i in range(DIM):
            for j in range(DIM):
                residual = sp.factor((ricci_brown[i, j] - ricci_direct[i, j]).subs(point))
                if residual != 0:
                    raise AssertionError(
                        f"Brown Ricci mismatch at point {pindex}, ({i},{j}): {residual}"
                    )

    print("PASS: det(gtilde)=1 exactly for the non-diagonal metric family")
    print("PASS: Brown first-order Ricci equals coordinate Ricci at 18 exact components")


if __name__ == "__main__":
    main()
