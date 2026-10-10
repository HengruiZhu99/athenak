"""Independently structured enclosures: polynomial Hermites, dimensional series.

The trusted interval primitives are shared and separately unit-gated; Gaussian
construction and angular minimization do not call producer_bounds.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial


@lru_cache(maxsize=128)
def hermite_polynomial(n):
    # Explicit coefficient formula, independent of the producer recurrence.
    coeff = [0] * (n + 1)
    for k in range(n // 2 + 1):
        coeff[n - 2 * k] = ((-1) ** k) * factorial(n) // ((1 << k) * factorial(k) * factorial(n - 2 * k))
    return tuple(coeff)


def derivative_family(ctx, orders, argument, sigma):
    x = argument / sigma
    x2, exponential = x.square(), ctx.exp_neg(x.square() / 2)
    result = {}
    for order in orders:
        # Parity Horner in x^2; one common exponential per argument.
        coeffs = hermite_polynomial(order)[order % 2::2]
        value = ctx.point(0)
        for coeff in reversed(coeffs):
            value = value * x2 + coeff
        if order % 2:
            value = value * x
        result[order] = ((-1) ** order) * sigma ** (4 - order) * value * exponential
    return result


def derivative_bound(order, sigma):
    if order % 2:
        m = (order - 1) // 2
        factor = (1 << (m + 1)) * factorial(m)
    else:
        factor = factorial(order) // ((1 << (order // 2)) * factorial(order // 2))
    return Q(factor) * sigma ** (4 - order)


def regular(ctx, R, T, exact_Rhi, sigma, K):
    rho = R.square()
    C, CT, Crho = ctx.point(0), ctx.point(0), ctx.point(0)
    derivatives = derivative_family(ctx, range(5, 2 * K + 7), T, sigma)
    # Descending Horner order, ordinary dimensional derivatives.
    for j in range(K, -1, -1):
        b = Q(1, (2 * j + 1) * (2 * j + 3) * (2 * j + 5) * factorial(2 * j)) * 2
        C = C * rho - b * derivatives[2 * j + 5]
        CT = CT * rho - b * derivatives[2 * j + 6]
    for j in range(K, 0, -1):
        b = Q(1, (2 * j + 1) * (2 * j + 3) * (2 * j + 5) * factorial(2 * j - 1))
        Crho = Crho * rho - b * derivatives[2 * j + 5]
    den = (2 * K + 3) * (2 * K + 5) * (2 * K + 7)
    e0 = 2 * derivative_bound(2 * K + 7, sigma) * exact_Rhi ** (2 * K + 2) / (den * factorial(2 * K + 2))
    e1 = 2 * derivative_bound(2 * K + 8, sigma) * exact_Rhi ** (2 * K + 2) / (den * factorial(2 * K + 2))
    e2 = derivative_bound(2 * K + 7, sigma) * exact_Rhi ** (2 * K) / (den * factorial(2 * K + 1))
    return C + ctx.enclose(-e0, e0), CT + ctx.enclose(-e1, e1), Crho + ctx.enclose(-e2, e2)


def separated(ctx, R, T, sigma):
    u, v = T - R, T + R
    a = derivative_family(ctx, range(4), u, sigma)
    b = derivative_family(ctx, range(4), v, sigma)
    # Common denominator grouping differs from producer inverse-power sums.
    C = ((a[2] - b[2]) * R.square() + 3 * (a[1] + b[1]) * R + 3 * (a[0] - b[0])) / R.power(5)
    CT = ((a[3] - b[3]) * R.square() + 3 * (a[2] + b[2]) * R + 3 * (a[1] - b[1])) / R.power(5)
    CR = (-(a[3] + b[3]) * R.power(3) - 6 * (a[2] - b[2]) * R.square()
          - 15 * (a[1] + b[1]) * R - 15 * (a[0] - b[0])) / R.power(6)
    return C, CT, CR / (2 * R)


def lower_bound(ctx, box, sigma, K):
    rlo, rhi, tlo, thi = box
    R, T = ctx.enclose(rlo, rhi), ctx.enclose(tlo, thi)
    if rhi <= 2 * sigma:
        C, CT, Crho = regular(ctx, R, T, rhi, sigma, K)
    elif rlo >= 2 * sigma:
        C, CT, Crho = separated(ctx, R, T, sigma)
    else:
        raise ValueError("replay cell crosses series interface")
    r2, e = R.square(), Q(3, 4)
    h_over_r = 1 / ctx.sqrt(4 + r2)
    constant = 4 / (4 + r2) - e * e * r2 * C.square()
    linear = 2 * e * r2 * (CT + 2 * h_over_r * C + 2 * h_over_r * r2 * Crho)
    quadratic = e * e * r2.square() * CT.square()
    quadratic = quadratic - 4 * e * e * r2.power(3) * Crho.square()
    quadratic = quadratic - 8 * e * e * r2.square() * C * Crho
    q, L = quadratic.lo, max(abs(linear.lo), abs(linear.hi))
    candidates = [Q(0), q / 4 - L / 2]
    if q > 0:
        vertex = L / (2 * q)
        if vertex <= Q(1, 2):
            candidates.append(q * vertex * vertex - L * vertex)
    return constant.lo + min(candidates)
