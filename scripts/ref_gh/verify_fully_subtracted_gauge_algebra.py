#!/usr/bin/env python3
"""Symbolically verify the fully subtracted Ref-GH gauge equations.

This is deliberately a small abstract-index oracle.  It does not assume the
stationary q=1 specialization while deriving the residual equations.  The
stationary matched-reference form is checked only as a corollary.
"""

from __future__ import annotations

import sympy as sp


NFRAME = 3
NSPACE = 2


def symbols(prefix: str, count: int) -> tuple[sp.Symbol, ...]:
    return sp.symbols(f"{prefix}0:{count}")


def matrix_symbols(prefix: str, rows: int, columns: int) -> list[list[sp.Symbol]]:
    return [list(symbols(f"{prefix}{row}_", columns)) for row in range(rows)]


def assert_zero(expressions: list[sp.Expr], label: str) -> None:
    failures = [sp.factor(expression) for expression in expressions
                if sp.factor(expression) != 0]
    if failures:
        raise AssertionError(f"{label}: {failures[0]}")


def verify_driver_residuals() -> None:
    mu, eta = sp.symbols("mu eta")
    beta_bar = symbols("beta_bar_", NSPACE)
    delta_beta = symbols("delta_beta_", NSPACE)
    beta = tuple(beta_bar[i] + delta_beta[i] for i in range(NSPACE))

    h_bar = symbols("Hbar_", NFRAME)
    delta_h = symbols("delta_H_", NFRAME)
    h = tuple(h_bar[a] + delta_h[a] for a in range(NFRAME))
    f_bar = symbols("Fbar_", NFRAME)
    delta_f = symbols("delta_F_", NFRAME)
    f = tuple(f_bar[a] + delta_f[a] for a in range(NFRAME))
    theta_bar = symbols("theta_bar_", NFRAME)
    delta_theta = symbols("delta_theta_", NFRAME)
    theta = tuple(theta_bar[a] + delta_theta[a] for a in range(NFRAME))
    dt_h_bar = symbols("dt_Hbar_", NFRAME)
    dt_theta_bar = symbols("dt_theta_bar_", NFRAME)

    d_h_bar = matrix_symbols("dHbar_", NSPACE, NFRAME)
    d_delta_h = matrix_symbols("ddeltaH_", NSPACE, NFRAME)
    omega_t = matrix_symbols("Omega_t_", NFRAME, NFRAME)
    omega_i = [matrix_symbols(f"Omega_{i}_", NFRAME, NFRAME)
               for i in range(NSPACE)]

    k_bar = [[
        d_h_bar[i][a]
        - sum(omega_i[i][a][b] * h_bar[b] for b in range(NFRAME))
        for a in range(NFRAME)] for i in range(NSPACE)]

    full_h_rhs: list[sp.Expr] = []
    residual_h_rhs: list[sp.Expr] = []
    full_theta_rhs: list[sp.Expr] = []
    residual_theta_rhs: list[sp.Expr] = []
    source_h: list[sp.Expr] = []
    source_theta: list[sp.Expr] = []
    for a in range(NFRAME):
        full_h = sum(beta[i] * (d_h_bar[i][a] + d_delta_h[i][a])
                     for i in range(NSPACE)) - mu * (h[a] - f[a]) + theta[a]
        full_h += sum(omega_t[a][b] * h[b] for b in range(NFRAME))
        full_h -= sum(beta[i] * omega_i[i][a][b] * h[b]
                      for i in range(NSPACE) for b in range(NFRAME))
        full_h_rhs.append(full_h)

        s_h = (sum(beta_bar[i] * k_bar[i][a] for i in range(NSPACE))
               - mu * (h_bar[a] - f_bar[a]) + theta_bar[a]
               + sum(omega_t[a][b] * h_bar[b] for b in range(NFRAME))
               - dt_h_bar[a])
        source_h.append(s_h)
        residual_h = sum(beta[i] * d_delta_h[i][a] for i in range(NSPACE))
        residual_h -= mu * (delta_h[a] - delta_f[a])
        residual_h += delta_theta[a]
        residual_h += sum(omega_t[a][b] * delta_h[b]
                          for b in range(NFRAME))
        residual_h -= sum(beta[i] * omega_i[i][a][b] * delta_h[b]
                          for i in range(NSPACE) for b in range(NFRAME))
        residual_h += s_h
        residual_h += sum(delta_beta[i] * k_bar[i][a]
                          for i in range(NSPACE))
        residual_h_rhs.append(residual_h)

        full_theta = -eta * theta[a]
        full_theta -= eta * sum(
            beta[i] * (d_h_bar[i][a] + d_delta_h[i][a])
            for i in range(NSPACE))
        full_theta += sum(omega_t[a][b] * theta[b]
                          for b in range(NFRAME))
        full_theta += eta * sum(
            beta[i] * omega_i[i][a][b] * h[b]
            for i in range(NSPACE) for b in range(NFRAME))
        full_theta_rhs.append(full_theta)

        s_theta = (-eta * theta_bar[a]
                   + sum(omega_t[a][b] * theta_bar[b]
                         for b in range(NFRAME))
                   - eta * sum(beta_bar[i] * k_bar[i][a]
                               for i in range(NSPACE))
                   - dt_theta_bar[a])
        source_theta.append(s_theta)
        residual_theta = -eta * delta_theta[a]
        residual_theta -= eta * sum(
            beta[i] * d_delta_h[i][a] for i in range(NSPACE))
        residual_theta += sum(omega_t[a][b] * delta_theta[b]
                              for b in range(NFRAME))
        residual_theta += eta * sum(
            beta[i] * omega_i[i][a][b] * delta_h[b]
            for i in range(NSPACE) for b in range(NFRAME))
        residual_theta += s_theta
        residual_theta -= eta * sum(
            delta_beta[i] * k_bar[i][a] for i in range(NSPACE))
        residual_theta_rhs.append(residual_theta)

    assert_zero([
        full_h_rhs[a] - dt_h_bar[a] - residual_h_rhs[a]
        for a in range(NFRAME)], "general delta-H equation")
    assert_zero([
        full_theta_rhs[a] - dt_theta_bar[a] - residual_theta_rhs[a]
        for a in range(NFRAME)], "general delta-theta equation")

    # If the reference itself obeys the two driver equations with beta_bar,
    # both explicit reference forcing terms vanish.  This substitution is
    # intentionally independent of stationarity.
    reference_driver_substitution = {
        dt_h_bar[a]: (sum(beta_bar[i] * k_bar[i][a]
                          for i in range(NSPACE))
                      - mu * (h_bar[a] - f_bar[a]) + theta_bar[a]
                      + sum(omega_t[a][b] * h_bar[b]
                            for b in range(NFRAME)))
        for a in range(NFRAME)
    }
    reference_driver_substitution.update({
        dt_theta_bar[a]: (-eta * theta_bar[a]
                          + sum(omega_t[a][b] * theta_bar[b]
                                for b in range(NFRAME))
                          - eta * sum(beta_bar[i] * k_bar[i][a]
                                      for i in range(NSPACE)))
        for a in range(NFRAME)
    })
    assert_zero([
        expression.subs(reference_driver_substitution)
        for expression in source_h + source_theta
    ], "reference forcing cancellation")

    # Static matched q=1: Fbar=Hbar, Omega_t=0, and
    # theta_bar=-beta_bar^i Kbar_i.  These are sufficient to make both
    # reference source terms exactly zero.
    stationary = {dt_h_bar[a]: 0 for a in range(NFRAME)}
    stationary.update({dt_theta_bar[a]: 0 for a in range(NFRAME)})
    stationary.update({f_bar[a]: h_bar[a] for a in range(NFRAME)})
    stationary.update({omega_t[a][b]: 0
                       for a in range(NFRAME) for b in range(NFRAME)})
    stationary.update({
        theta_bar[a]: -sum(beta_bar[i] * k_bar[i][a]
                           for i in range(NSPACE))
        for a in range(NFRAME)
    })
    assert_zero([
        expression.subs(stationary)
        for expression in source_h + source_theta
    ], "stationary matched-reference forcing")


def verify_einstein_residual() -> None:
    # Two coordinate/frame components are enough to verify every indexed
    # product rule and the symmetric source assembly.
    n = 2
    coframe = matrix_symbols("coframe_", n, n)
    d_coframe = [matrix_symbols(f"dcoframe_{p}_", n, n) for p in range(n)]
    h_bar = symbols("HbarE_", n)
    delta_h = symbols("deltaHE_", n)
    d_h_bar = matrix_symbols("dHbarE_", n, n)
    d_delta_h = matrix_symbols("ddeltaHE_", n, n)
    base = symbols("base_", n)
    d_base = matrix_symbols("dbase_", n, n)
    connection = [[[sp.Symbol(f"Gamma_{c}_{a}_{b}") for b in range(n)]
                   for a in range(n)] for c in range(n)]
    projector = [[[sp.Symbol(f"P_{a}_{b}_{c}") for c in range(n)]
                  for b in range(n)] for a in range(n)]
    gamma0 = sp.Symbol("gamma0")

    href_coordinate = [sum(coframe[a][capital] * h_bar[capital]
                            for capital in range(n)) for a in range(n)]
    h_coordinate = [sum(coframe[a][capital]
                        * (h_bar[capital] + delta_h[capital])
                        for capital in range(n)) for a in range(n)]
    delta_base = [base[a] - href_coordinate[a] for a in range(n)]
    full_j = [h_coordinate[a] - base[a] for a in range(n)]
    residual_j = [sum(coframe[a][capital] * delta_h[capital]
                      for capital in range(n)) - delta_base[a]
                  for a in range(n)]
    assert_zero([full_j[a] - residual_j[a] for a in range(n)],
                "Einstein J residual")

    d_href_coordinate = [[
        sum(d_coframe[p][a][capital] * h_bar[capital]
            + coframe[a][capital] * d_h_bar[p][capital]
            for capital in range(n))
        for a in range(n)] for p in range(n)]
    d_h_coordinate = [[
        sum(d_coframe[p][a][capital]
                * (h_bar[capital] + delta_h[capital])
            + coframe[a][capital]
                * (d_h_bar[p][capital] + d_delta_h[p][capital])
            for capital in range(n))
        for a in range(n)] for p in range(n)]
    d_delta_base = [[d_base[p][a] - d_href_coordinate[p][a]
                     for a in range(n)] for p in range(n)]
    d_full_j = [[d_h_coordinate[p][a] - d_base[p][a]
                 for a in range(n)] for p in range(n)]
    d_residual_j = [[
        sum(d_coframe[p][a][capital] * delta_h[capital]
            + coframe[a][capital] * d_delta_h[p][capital]
            for capital in range(n)) - d_delta_base[p][a]
        for a in range(n)] for p in range(n)]
    assert_zero([d_full_j[p][a] - d_residual_j[p][a]
                 for p in range(n) for a in range(n)],
                "Einstein dJ residual")

    def source(j: list[sp.Expr], dj: list[list[sp.Expr]]) -> list[sp.Expr]:
        result = []
        for a in range(n):
            for b in range(n):
                nabla_ab = dj[a][b] - sum(
                    connection[c][a][b] * j[c] for c in range(n))
                nabla_ba = dj[b][a] - sum(
                    connection[c][b][a] * j[c] for c in range(n))
                result.append(-nabla_ab - nabla_ba + gamma0 * sum(
                    projector[a][b][c] * j[c] for c in range(n)))
        return result

    assert_zero([left - right for left, right in zip(
        source(full_j, d_full_j), source(residual_j, d_residual_j))],
        "Einstein ordinary-gauge increment")


def main() -> None:
    verify_driver_residuals()
    verify_einstein_residual()
    print("fully_subtracted_driver_general_equivalence=PASS")
    print("fully_subtracted_driver_stationary_q1_reduction=PASS")
    print("fully_subtracted_einstein_gauge_increment=PASS")


if __name__ == "__main__":
    main()
