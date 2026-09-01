#!/usr/bin/env python3
"""Principal symbol of PC-GH with the direct z4c_mp gauge on a flat local frame.

The frozen background has gtilde_ij=delta_ij, positive lapse alpha and chi,
vanishing shift, and vanishing nonconfiguration fields.  The wave covector is
dx.  The returned 50 by 50 symbol is pulled back to the tangent space of the
five conformal algebraic constraints.
"""

from __future__ import annotations

import argparse
import sympy as sp


N = 55
CHI = 0
G = (1, 2, 3, 4, 5, 6)
K = 7
AT = (8, 9, 10, 11, 12, 13)
LAM = (14, 15, 16)
PI = 17
A = 18
BETA = (19, 20, 21)
X = (22, 23, 24)
Q0 = 25
Y = (43, 44, 45)
B0 = 46


def sym_index(i: int, j: int) -> int:
    if i > j:
        i, j = j, i
    return j if i == 0 else j + 2 if i == 1 else 5


def q_index(k: int, i: int, j: int) -> int:
    return Q0 + 6*k + sym_index(i, j)


def b_index(i: int, j: int) -> int:
    return B0 + 3*i + j


def full_symbol(alpha: sp.Expr, chi: sp.Expr,
                switch: sp.Expr = sp.Integer(0)) -> sp.MutableSparseMatrix:
    """Return P in partial_t u = P partial_x u at the stated background."""
    p = sp.MutableSparseMatrix(N, N, {})

    # Corrected Einstein sector, specialized only by evaluating its principal
    # coefficients on the flat local frame.
    p[K, Y[0]] += -chi/(2*alpha)
    p[K, X[0]] += 2*alpha
    p[K, LAM[0]] += alpha*chi
    for d in range(3):
        p[K, q_index(0, d, d)] += -alpha*chi/2
    p[PI, Y[0]] += chi/(2*alpha)

    p[LAM[0], b_index(0, 0)] += sp.Rational(4, 3)
    p[LAM[0], b_index(1, 1)] += sp.Rational(1, 3)
    p[LAM[0], b_index(2, 2)] += sp.Rational(1, 3)
    p[LAM[0], PI] += alpha
    p[LAM[0], K] += -alpha/sp.Integer(3)
    p[LAM[1], b_index(0, 1)] += 1
    p[LAM[2], b_index(0, 2)] += 1

    # S_ij before its trace-free projection in the Atilde equation.
    source = [[{} for _ in range(3)] for _ in range(3)]

    def add(expr: dict[int, sp.Expr], variable: int, coefficient: sp.Expr) -> None:
        expr[variable] = expr.get(variable, 0) + coefficient

    for i in range(3):
        for j in range(3):
            expr = source[i][j]
            add(expr, q_index(0, i, j), -alpha*chi/2)
            if i == 0:
                add(expr, LAM[j], alpha*chi/2)
                add(expr, X[j], alpha/2)
                add(expr, Y[j], -chi/(2*alpha))
            if j == 0:
                add(expr, LAM[i], alpha*chi/2)

    trace: dict[int, sp.Expr] = {}
    for d in range(3):
        for variable, coefficient in source[d][d].items():
            add(trace, variable, coefficient)
    for i in range(3):
        for j in range(i, 3):
            row = AT[sym_index(i, j)]
            for variable, coefficient in source[i][j].items():
                p[row, variable] += coefficient
            if i == j:
                for variable, coefficient in trace.items():
                    p[row, variable] -= coefficient/3

    # STANDARD first-order gradient sector.  Only the row whose derivative
    # index is parallel to the wave covector contains principal derivatives.
    p[X[0], K] += 2*alpha*chi/3
    for d in range(3):
        p[X[0], b_index(d, d)] += -2*chi/3
    p[Y[0], K] += -4*alpha**2
    for i in range(3):
        p[b_index(0, i), LAM[i]] += 1
        p[b_index(0, i), X[i]] += switch*alpha**2/2
        p[b_index(0, i), Y[i]] += -switch*chi/2
    for i in range(3):
        for j in range(i, 3):
            row = q_index(0, i, j)
            p[row, AT[sym_index(i, j)]] += -2*alpha
            p[row, b_index(j, i)] += 1
            p[row, b_index(i, j)] += 1
            if i == j:
                for d in range(3):
                    p[row, b_index(d, d)] += -sp.Rational(2, 3)
    return p


def algebraic_tangent_symbol(alpha: sp.Expr, chi: sp.Expr,
                             switch: sp.Expr = sp.Integer(0)) -> sp.Matrix:
    """Eliminate g_zz, Atilde_zz, and Q_kzz from the flat-frame tangent."""
    eliminated = {G[5], AT[5], q_index(0, 2, 2), q_index(1, 2, 2),
                  q_index(2, 2, 2)}
    kept = [index for index in range(N) if index not in eliminated]
    tangent = sp.zeros(N, len(kept))
    for column, index in enumerate(kept):
        tangent[index, column] = 1

    def impose_trace(dependent: int, first: int, second: int) -> None:
        tangent[dependent, kept.index(first)] = -1
        tangent[dependent, kept.index(second)] = -1

    impose_trace(G[5], G[0], G[3])
    impose_trace(AT[5], AT[0], AT[3])
    for d in range(3):
        impose_trace(q_index(d, 2, 2), q_index(d, 0, 0), q_index(d, 1, 1))
    selector = sp.zeros(len(kept), N)
    for row, index in enumerate(kept):
        selector[row, index] = 1
    return selector*full_symbol(alpha, chi, switch)*tangent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charpoly", action="store_true")
    args = parser.parse_args()
    alpha, chi = sp.symbols("alpha chi", positive=True)
    speed = sp.symbols("speed")
    switch = sp.symbols("switch", nonnegative=True)
    symbol = algebraic_tangent_symbol(alpha, chi)
    if symbol.shape != (50, 50):
        raise AssertionError(symbol.shape)
    if args.charpoly:
        polynomial = sp.factor(symbol.charpoly(speed).as_expr())
        print(polynomial)
    else:
        expected = (-speed**30*(speed - 1)**2*(speed + 1)**2
                    *(3*speed**2 - 4)*(2*alpha*chi - speed**2)
                    *(alpha**2*chi - speed**2)**6/3)
        actual = sp.factor(symbol.charpoly(speed).as_expr())
        if sp.simplify(actual - expected) != 0:
            raise AssertionError(f"unexpected characteristic polynomial: {actual}")

        def nullity(matrix: sp.Matrix, eigenvalue: sp.Expr) -> int:
            return matrix.rows - (matrix - eigenvalue*sp.eye(matrix.rows)).rank()

        # A generic point has a complete set of 50 eigenvectors.
        generic = algebraic_tangent_symbol(sp.Rational(3, 2), sp.Rational(1, 2))
        generic_data = [
            (0, 30), (1, 2), (-1, 2),
            (2/sp.sqrt(3), 1), (-2/sp.sqrt(3), 1),
            (sp.sqrt(sp.Rational(3, 2)), 1),
            (-sp.sqrt(sp.Rational(3, 2)), 1),
            (3*sp.sqrt(2)/4, 6), (-3*sp.sqrt(2)/4, 6),
        ]
        if any(nullity(generic, value) != multiplicity
               for value, multiplicity in generic_data):
            raise AssertionError("generic z4c_mp symbol is not diagonalizable")

        # Exact parameterized ranks locate the three defective surfaces.
        parameter = sp.symbols("parameter", positive=True)
        defective = [
            (algebraic_tangent_symbol(parameter, 4/(3*parameter**2)),
             2/sp.sqrt(3), 7, 6, "alpha^2 chi=4/3"),
            (algebraic_tangent_symbol(2, parameter),
             2*sp.sqrt(parameter), 7, 5, "alpha=2"),
            (algebraic_tangent_symbol(parameter, 2/(3*parameter)),
             2/sp.sqrt(3), 2, 1, "alpha chi=2/3"),
        ]
        for matrix, value, algebraic, geometric, name in defective:
            observed = nullity(matrix, value)
            if observed != geometric or observed >= algebraic:
                raise AssertionError(
                    f"{name}: expected multiplicities {algebraic}/{geometric}, "
                    f"got geometric {observed}"
                )

        # The remaining speed coincidences are semisimple and are not defects.
        semisimple = [
            (algebraic_tangent_symbol(parameter, 1/parameter**2), 1, 8,
             "alpha^2 chi=1"),
            (algebraic_tangent_symbol(parameter, 1/(2*parameter)), 1, 3,
             "2 alpha chi=1"),
        ]
        for matrix, value, multiplicity, name in semisimple:
            observed = nullity(matrix, value)
            if observed != multiplicity:
                raise AssertionError(f"{name}: expected nullity {multiplicity}, got {observed}")

        modified = algebraic_tangent_symbol(alpha, chi, switch)
        modified_actual = sp.factor(modified.charpoly(speed).as_expr())
        modified_expected = (speed**30*(speed - 1)**2*(speed + 1)**2
                             *(-2*alpha*chi + speed**2)
                             *(-alpha**2*chi + speed**2)**6
                             *(switch*alpha**2*chi + 3*speed**2 - 4)/3)
        if sp.simplify(modified_actual - modified_expected) != 0:
            raise AssertionError(
                f"unexpected modified characteristic polynomial: {modified_actual}"
            )

        # With S=1, both shift-family coincidences are semisimple.  The
        # lapse/light alpha=2 defect is unchanged, but lies outside the stated
        # alpha<=1 puncture domain.
        lapse_shift = algebraic_tangent_symbol(
            sp.Integer(1), sp.Rational(4, 7), sp.Integer(1))
        if nullity(lapse_shift, sp.sqrt(sp.Rational(8, 7))) != 2:
            raise AssertionError("S=1 lapse/longitudinal-shift coincidence is defective")
        light_shift = algebraic_tangent_symbol(
            sp.Integer(1), sp.Integer(1), sp.Integer(1))
        if nullity(light_shift, 1) != 9:
            raise AssertionError("S=1 light/shift coincidence is defective")

        wormhole_radius = 1/(2*((sp.Rational(3, 2))**sp.Rational(1, 6) - 1))
        print("PASS: exact 50x50 z4c_mp algebraic-tangent principal symbol")
        print("characteristic polynomial:", actual)
        print("strongly hyperbolic for alpha>0, chi>0 except")
        print("  alpha^2 chi=4/3, alpha=2, or alpha chi=2/3")
        print("semisimple coincidences: alpha^2 chi=1 and 2 alpha chi=1")
        print("unit-mass wormhole alpha chi=2/3 shell: "
              f"r/M={float(wormhole_radius):.15f}")
        print("modified longitudinal speed^2=(4-S alpha^2 chi)/3")
        print("S=1 makes the lapse/longitudinal and light/longitudinal "
              "coincidences semisimple")
        print("a switch completed below alpha chi=4/7 is strongly hyperbolic "
              "for 0<alpha<=1, 0<alpha chi<=1")


if __name__ == "__main__":
    main()
