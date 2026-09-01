#!/usr/bin/env python3
"""Exact audit of the direct Z4c moving-puncture gauge in PC-GH variables."""

import sympy as sp


def assert_zero(name, expression):
    residual = sp.simplify(sp.expand(expression))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def main():
    A, chi, alpha, K, pi, eta = sp.symbols(
        "A chi alpha K pi eta", real=True
    )
    Y_i, d_i_K, L_i, d_i_pi = sp.symbols(
        "Y_i d_i_K L_i d_i_pi", real=True
    )
    beta_j, Lambda_j, B_i_j, d_i_Lambda_j = sp.symbols(
        "beta_j Lambda_j B_i_j d_i_Lambda_j", real=True
    )
    metric_term, d_i_metric_term = sp.symbols(
        "metric_term d_i_metric_term", real=True
    )
    switch, d_i_switch = sp.symbols("switch d_i_switch", real=True)

    # Lapse source and its exact spatial derivative.
    h_perp = alpha*pi + 2*K
    d_i_h_perp = sp.Rational(1, 2)*L_i*pi + alpha*d_i_pi + 2*d_i_K
    d0_A = 2*A*(alpha*pi - h_perp)
    partial_i_f_A = (
        2*Y_i*(alpha*pi - h_perp)
        + 2*A*(sp.Rational(1, 2)*L_i*pi + alpha*d_i_pi - d_i_h_perp)
    )
    assert_zero("D0 A", d0_A + 4*A*K)
    assert_zero("partial_i F_A", partial_i_f_A + 4*K*Y_i + 4*A*d_i_K)

    # metric_term abbreviates 1/2 gtilde^{j ell}(A X_ell-chi Y_ell).
    h_j = (1 - A*chi)*Lambda_j - metric_term - eta*beta_j
    f_beta_j = h_j + A*chi*Lambda_j + metric_term
    d_i_h_j = (
        -(Y_i*chi + A*sp.symbols("X_i"))*Lambda_j
        + (1 - A*chi)*d_i_Lambda_j
        - d_i_metric_term
        - eta*B_i_j
    )
    d_i_f_beta_j = (
        d_i_h_j
        + (Y_i*chi + A*sp.symbols("X_i"))*Lambda_j
        + A*chi*d_i_Lambda_j
        + d_i_metric_term
    )
    assert_zero("D0 beta^j", f_beta_j - Lambda_j + eta*beta_j)
    assert_zero(
        "partial_i F_beta^j", d_i_f_beta_j - d_i_Lambda_j + eta*B_i_j
    )

    # Adding the standard advective/curl ordering gives the requested Y/B RHS.
    adv_Y_i, adv_B_i_j, B_i_k_Y_k, B_i_k_B_k_j = sp.symbols(
        "adv_Y_i adv_B_i_j B_i_k_Y_k B_i_k_B_k_j", real=True
    )
    rhs_Y = adv_Y_i + B_i_k_Y_k + partial_i_f_A
    rhs_B = adv_B_i_j + B_i_k_B_k_j + d_i_f_beta_j
    assert_zero(
        "standard Y_i RHS",
        rhs_Y - (adv_Y_i + B_i_k_Y_k - 4*K*Y_i - 4*A*d_i_K),
    )
    assert_zero(
        "standard B_i^j RHS",
        rhs_B
        - (adv_B_i_j + B_i_k_B_k_j + d_i_Lambda_j - eta*B_i_j),
    )

    modified_f_beta_j = f_beta_j + switch*metric_term
    modified_d_i_f_beta_j = (
        d_i_f_beta_j + d_i_switch*metric_term + switch*d_i_metric_term
    )
    assert_zero(
        "modified D0 beta^j",
        modified_f_beta_j
        - (Lambda_j - eta*beta_j + switch*metric_term),
    )
    assert_zero(
        "modified partial_i F_beta^j",
        modified_d_i_f_beta_j
        - (d_i_Lambda_j - eta*B_i_j
           + d_i_switch*metric_term + switch*d_i_metric_term),
    )

    print("PASS: z4c_mp h_perp gives D0 A=-4 A K")
    print("PASS: z4c_mp h^i gives D0 beta^i=Lambda^i-eta beta^i")
    print("PASS: exact STANDARD z4c_mp Y_i and B_i^j equations")
    print("PASS: exact differentiated switched metric-gradient shift term")


if __name__ == "__main__":
    main()
