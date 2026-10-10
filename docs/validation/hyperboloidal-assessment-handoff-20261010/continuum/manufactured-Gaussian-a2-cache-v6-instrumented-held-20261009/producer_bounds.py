"""Scaled Gaussian formulas and exact angular minimum, no import-time run."""
from fractions import Fraction as Q
from math import factorial


def moment(n):
    if n < 0:
        raise ValueError("nonnegative order required")
    if n % 2:
        m = (n - 1) // 2
        return Q((1 << (m + 1)) * factorial(m))
    m, value = n // 2, 1
    for j in range(1, m + 1):
        value *= 2 * j - 1
    return Q(value)


def remainder_constants(K):
    return [2 * moment(2 * K + 7) / (15 * factorial(2 * K + 2)),
            2 * moment(2 * K + 8) / (15 * factorial(2 * K + 2)),
            moment(2 * K + 7) / (15 * factorial(2 * K + 1))]


def hermites(x, maximum):
    out = [x.ctx.point(1), x]
    for n in range(1, maximum):
        out.append(x * out[-1] - n * out[-2])
    return out[:maximum + 1]


def gaussian_derivatives(ctx, arg, sigma, maximum):
    x = arg / sigma
    exponential = ctx.exp_neg(x.square() / 2)
    he = hermites(x, maximum)
    return [((-1) ** n) * (sigma ** (4 - n)) * he[n] * exponential
            for n in range(maximum + 1)]


def regular(ctx, R, T, Rhi, sigma, K):
    """C,CT,C_rho; exact domain Rhi controls remainders, not rounded R.hi."""
    x, rho = T / sigma, (R / sigma).square()
    exponential = ctx.exp_neg(x.square() / 2)
    he = hermites(x, 2 * K + 6)
    scaled = [((-1) ** n) * he[n] * exponential for n in range(2 * K + 7)]
    c, ct, crho = ctx.point(0), ctx.point(0), ctx.point(0)
    power, prior = ctx.point(1), None
    for j in range(K + 1):
        b = Q(8 * (j + 1) * (j + 2), factorial(2 * j + 5))
        c = c - b * scaled[2 * j + 5] * power
        ct = ct - b * scaled[2 * j + 6] * power
        if j:
            crho = crho - j * b * scaled[2 * j + 5] * prior
        prior, power = power, power * rho
    r = Rhi / sigma
    den = (2 * K + 3) * (2 * K + 5) * (2 * K + 7)
    ec = 2 * moment(2 * K + 7) * r ** (2 * K + 2) / (den * factorial(2 * K + 2))
    ect = 2 * moment(2 * K + 8) * r ** (2 * K + 2) / (den * factorial(2 * K + 2))
    erho = moment(2 * K + 7) * r ** (2 * K) / (den * factorial(2 * K + 1))
    return [(c + ctx.enclose(-ec, ec)) / sigma,
            (ct + ctx.enclose(-ect, ect)) / (sigma * sigma),
            (crho + ctx.enclose(-erho, erho)) / (sigma ** 3)]


def separated(ctx, R, T, sigma):
    fu = gaussian_derivatives(ctx, T - R, sigma, 3)
    fv = gaussian_derivatives(ctx, T + R, sigma, 3)
    inv = 1 / R
    c = (fu[2] - fv[2]) * inv.power(3) + 3 * (fu[1] + fv[1]) * inv.power(4)
    c = c + 3 * (fu[0] - fv[0]) * inv.power(5)
    ct = (fu[3] - fv[3]) * inv.power(3) + 3 * (fu[2] + fv[2]) * inv.power(4)
    ct = ct + 3 * (fu[1] - fv[1]) * inv.power(5)
    cr = -(fu[3] + fv[3]) * inv.power(3) - 6 * (fu[2] - fv[2]) * inv.power(4)
    cr = cr - 15 * (fu[1] + fv[1]) * inv.power(5) - 15 * (fu[0] - fv[0]) * inv.power(6)
    return [c, ct, cr / (2 * R)]


def quad_lower(q, ell):
    if ell < 0:
        raise ValueError("negative absolute linear bound")
    if q <= 0 or ell > q:
        return q / 4 - ell / 2
    return -(ell * ell) / (4 * q)


def coefficients(ctx, box, sigma, K):
    rlo, rhi, tlo, thi = box
    R, T = ctx.enclose(rlo, rhi), ctx.enclose(tlo, thi)
    if rhi <= sigma:
        c, ct, crho = regular(ctx, R, T, rhi, sigma, K)
        method = "regular"
    elif rlo >= sigma:
        c, ct, crho = separated(ctx, R, T, sigma)
        method = "separated"
    else:
        raise ValueError("cell crosses fixed series boundary")
    rho, E = R.square(), Q(3, 4)
    m = 4 / (4 + rho) - E * E * rho * c.square()
    ell = 2 * E * rho * (ct + 2 * (c + rho * crho) / ctx.sqrt(4 + rho))
    q = E * E * rho.square() * (ct.square() - 4 * rho * crho.square() - 8 * c * crho)
    L = max(abs(ell.lo), abs(ell.hi))
    return {"mlo": m.lo, "qlo": q.lo, "Lhi": L,
            "lower": m.lo + quad_lower(q.lo, L), "method": method}
