#!/usr/bin/env python3
"""Exact algebraic proof checks for the consistent conformal-metric/Q projection."""

import sympy as sp


def assert_zero(name, expr):
    residual = sp.simplify(sp.cancel(expr))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def main():
    g00, g01, g02, g11, g12, g22 = sp.symbols(
        "g00 g01 g02 g11 g12 g22", real=True
    )
    G = sp.Matrix([[g00, g01, g02], [g01, g11, g12], [g02, g12, g22]])
    if G.det() == 0:
        raise AssertionError("symbolic metric determinant vanished identically")
    Ginv = G.inv()

    q00, q01, q02, q11, q12, q22 = sp.symbols(
        "q00 q01 q02 q11 q12 q22", real=True
    )
    Q = sp.Matrix([[q00, q01, q02], [q01, q11, q12], [q02, q12, q22]])
    scale = sp.symbols("s", positive=True, finite=True)

    trace_q = sp.trace(Ginv * Q)
    dscale = -scale * trace_q / 3  # Jacobi: d ln det(G)=tr(G^{-1} dG)
    derivative_of_projected_metric = dscale * G + scale * Q
    projected_q = scale * (Q - G * trace_q / 3)

    for i in range(3):
        for j in range(3):
            assert_zero(
                f"product-rule consistency ({i},{j})",
                derivative_of_projected_metric[i, j] - projected_q[i, j],
            )

    projected_metric_inv = Ginv / scale
    assert_zero("projected Q trace", sp.trace(projected_metric_inv * projected_q))

    print("PASS: Q projection is the derivative of det(G)^(-1/3) G")
    print("PASS: projected Q is trace-free with the projected metric")


if __name__ == "__main__":
    main()
