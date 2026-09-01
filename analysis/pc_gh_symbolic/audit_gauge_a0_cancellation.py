#!/usr/bin/env python3
"""Audit Gauge-A0 production RHS temporaries in three arithmetic precisions.

The audit differentiates the same cubic-Hermite interpolant used by the C++ target
initializer.  It evaluates the conformally-flat stationary trumpet on the positive
x axis, retaining all tensor components and the additive grouping of the production
RHS.  Spherical symmetry makes this axis representative while preserving radial and
tangential tensor sectors.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

import mpmath as mp
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE.parents[1] / "inputs/pc_gh/gauge_a0_m1.dat"
FIELDS = ("A", "chi", "beta_r", "K", "At_radial")


class Arithmetic:
    def __init__(self, name: str, cast: Callable, sqrt: Callable, log: Callable):
        self.name = name
        self.cast = cast
        self.sqrt = sqrt
        self.log = log


ARITHMETICS = (
    Arithmetic("binary64", np.float64, np.sqrt, np.log),
    Arithmetic("long_double", np.longdouble, np.sqrt, np.log),
    Arithmetic("mp100", mp.mpf, mp.sqrt, mp.log),
)


def load_table(path: Path, arithmetic: Arithmetic):
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            rows.append([arithmetic.cast(token) for token in line.split()])
    if len(rows) < 2 or any(len(row) != 18 for row in rows):
        raise ValueError("Gauge A0 table must contain at least two 18-column rows")
    return rows


def interpolate(table, field: int, log_r, cast: Callable):
    """Return the value and first three log-radius derivatives of one polynomial."""
    n = len(table)
    spacing = (table[-1][0] - table[0][0])/cast(n - 1)
    location = (log_r - table[0][0])/spacing
    interval = int(mp.floor(location)) if isinstance(location, mp.mpf) \
        else int(np.floor(location))
    if interval < 0 or interval >= n - 1:
        raise ValueError(f"sample outside open table domain: log(r)={log_r}")
    t = location - cast(interval)
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    y0, m0, a0 = table[interval][1 + field:4 + field]
    y1, m1, a1 = table[interval + 1][1 + field:4 + field]
    c0 = y0
    c1 = spacing*m0
    c2 = spacing*spacing*a0/cast(2)
    dy = y1 - c0 - c1 - c2
    dm = spacing*m1 - c1 - cast(2)*c2
    da = spacing*spacing*a1 - cast(2)*c2
    c3 = cast(10)*dy - cast(4)*dm + da/cast(2)
    c4 = -cast(15)*dy + cast(7)*dm - da
    c5 = cast(6)*dy - cast(3)*dm + da/cast(2)
    value = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t4*t
    dx = (c1 + cast(2)*c2*t + cast(3)*c3*t2 + cast(4)*c4*t3
          + cast(5)*c5*t4)/spacing
    dxx = (cast(2)*c2 + cast(6)*c3*t + cast(12)*c4*t2
           + cast(20)*c5*t3)/(spacing*spacing)
    dxxx = (cast(6)*c3 + cast(24)*c4*t + cast(60)*c5*t2) \
        /(spacing*spacing*spacing)
    return value, dx, dxx, dxxx


def interpolate_cubic(table, field: int, log_r, cast: Callable):
    n = len(table)
    spacing = (table[-1][0] - table[0][0])/cast(n - 1)
    location = (log_r - table[0][0])/spacing
    interval = int(mp.floor(location)) if isinstance(location, mp.mpf) \
        else int(np.floor(location))
    if interval < 0 or interval >= n - 1:
        raise ValueError(f"sample outside open table domain: log(r)={log_r}")
    t = location - cast(interval)
    t2, t3 = t*t, t*t*t
    y0, m0 = table[interval][1 + field:3 + field]
    y1, m1 = table[interval + 1][1 + field:3 + field]
    value = (cast(2)*t3 - cast(3)*t2 + cast(1))*y0 \
        + (t3 - cast(2)*t2 + t)*spacing*m0 \
        + (-cast(2)*t3 + cast(3)*t2)*y1 + (t3 - t2)*spacing*m1
    dx = ((cast(6)*t2 - cast(6)*t)*y0
          + (-cast(6)*t2 + cast(6)*t)*y1)/spacing \
        + (cast(3)*t2 - cast(4)*t + cast(1))*m0 \
        + (cast(3)*t2 - cast(2)*t)*m1
    return value, dx


def tensor(shape, zero):
    if len(shape) == 1:
        return [zero for _ in range(shape[0])]
    return [tensor(shape[1:], zero) for _ in range(shape[0])]


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def evaluate(radius_text: str, table, arithmetic: Arithmetic):
    c = arithmetic.cast
    zero, one = c(0), c(1)
    r = c(radius_text)
    log_r = arithmetic.log(r)
    a, ax, axx, _ = interpolate(table, 0, log_r, c)
    chi, chix, chixx, _ = interpolate(table, 3, log_r, c)
    beta_r, betax, betaxx, _ = interpolate(table, 6, log_r, c)
    trace_k, kx = interpolate_cubic(table, 9, log_r, c)
    at_radial, atx = interpolate_cubic(table, 11, log_r, c)
    h_perp, hpx = interpolate_cubic(table, 13, log_r, c)
    h_radial, hrx = interpolate_cubic(table, 15, log_r, c)
    alpha = arithmetic.sqrt(a)
    alpha_x = alpha*ax/(c(2)*a)
    tangential_b = beta_r/r
    tracefree_b = alpha*at_radial
    radial_b = tangential_b + tracefree_b
    tangential_b_x = (betax - beta_r)/r
    tracefree_b_x = alpha_x*at_radial + alpha*atx
    radial_b_x = tangential_b_x + tracefree_b_x
    sqrt_chi = arithmetic.sqrt(chi)

    g = tensor((3, 3), zero)
    gu = tensor((3, 3), zero)
    at = tensor((3, 3), zero)
    aa = tensor((3, 3), zero)
    at_uu = tensor((3, 3), zero)
    q = tensor((3, 3, 3), zero)
    b = tensor((3, 3), zero)
    beta = [beta_r, zero, zero]
    normal = [one, zero, zero]
    x = [chix/r, zero, zero]
    y = [ax/r, zero, zero]
    w = [item/sqrt_chi for item in x]
    ell_lapse = [item/alpha for item in y]
    lam = [zero, zero, zero]
    pi = -trace_k
    c_perp = pi + trace_k
    for i in range(3):
        g[i][i] = gu[i][i] = one
        for j in range(3):
            delta = one if i == j else zero
            at[i][j] = at_radial*(normal[i]*normal[j] - delta/c(3))
            at_uu[i][j] = at[i][j]
    for i in range(3):
        for j in range(3):
            aa[i][j] = sum(at[i][p]*at[p][j] for p in range(3))

    b_delta = tracefree_b
    for i in range(3):
        for j in range(3):
            delta = one if i == j else zero
            b[i][j] = tangential_b*delta + b_delta*normal[i]*normal[j]

    d_k = [kx/r, zero, zero]
    d_pi = [-item for item in d_k]
    d_lam = tensor((3, 3), zero)
    d_at = tensor((3, 3, 3), zero)
    d_x = tensor((3, 3), zero)
    d_y = tensor((3, 3), zero)
    d_q = tensor((3, 3, 3, 3), zero)
    d_b = tensor((3, 3, 3), zero)

    chi_r = chix/r
    chi_rr = (chixx - chix)/(r*r)
    a_r = ax/r
    a_rr = (axx - ax)/(r*r)
    at_r = atx/r
    for d in range(3):
        for i in range(3):
            delta_di = one if d == i else zero
            d_x[d][i] = chi_rr*normal[d]*normal[i] \
                + chi_r*(delta_di - normal[d]*normal[i])/r
            d_y[d][i] = a_rr*normal[d]*normal[i] \
                + a_r*(delta_di - normal[d]*normal[i])/r
            for j in range(3):
                delta_ij = one if i == j else zero
                delta_di = one if d == i else zero
                delta_dj = one if d == j else zero
                d_at[d][i][j] = at_r*normal[d]*(normal[i]*normal[j]
                    - delta_ij/c(3)) + at_radial*(
                        delta_di*normal[j] + normal[i]*delta_dj
                        - c(2)*normal[i]*normal[j]*normal[d])/r

    t_r = tangential_b_x/r
    v_r = tracefree_b_x/r
    for d in range(3):
        for i in range(3):
            for j in range(3):
                delta_ij = one if i == j else zero
                delta_di = one if d == i else zero
                delta_dj = one if d == j else zero
                d_b[d][i][j] = t_r*normal[d]*delta_ij \
                    + v_r*normal[d]*normal[i]*normal[j] \
                    + b_delta*(delta_di*normal[j] + normal[i]*delta_dj
                               - c(2)*normal[i]*normal[j]*normal[d])/r

    h_source = [h_radial, zero, zero]
    d_h_perp = [hpx/r, zero, zero]
    d_h_source = tensor((3, 3), zero)
    for d in range(3):
        for i in range(3):
            tangent = (one if d == i else zero) - normal[d]*normal[i]
            d_h_source[d][i] = (hrx*normal[d]*normal[i] + h_radial*tangent)/r

    values = {}
    kinds = {}

    def record(name, value, kind="temporary"):
        magnitude = max((abs(item) for item in flatten(value)), default=zero)
        values[name] = magnitude
        kinds[name] = kind

    def rhs(component, terms):
        total = sum(terms.values(), zero)
        for name, value in terms.items():
            record(f"rhs/{component}/{name}", value, "rhs_term")
        record(f"rhs/{component}/sum", total, "rhs_sum")
        record(f"rhs/{component}/sumabs", sum(abs(v) for v in terms.values()),
               "rhs_sumabs")

    r_minus = chi/alpha
    r_plus = alpha/sqrt_chi
    record("state/A", a)
    record("state/chi", chi)
    record("state/K", trace_k)
    record("state/At", at)
    record("state/beta", beta)
    record("state/X", x)
    record("state/Y", y)
    record("state/B", b)
    record("derived/r_minus", r_minus)
    record("derived/r_plus", r_plus)
    record("derived/W", w)
    record("derived/L", ell_lapse)
    record("derivative/dK", d_k)
    record("derivative/dAt", d_at)
    record("derivative/dX", d_x)
    record("derivative/dY", d_y)
    record("derivative/dB", d_b)
    record("gauge/h_perp", h_perp)
    record("gauge/h_source", h_source)
    record("gauge/d_h_perp", d_h_perp)
    record("gauge/d_h_source", d_h_source)

    trace_b = sum(b[i][i] for i in range(3))
    at_sq = sum(at[i][j]*at_uu[i][j] for i in range(3) for j in range(3))
    w_sq = sum(gu[i][j]*w[i]*w[j] for i in range(3) for j in range(3))
    x_dot_l = sum(gu[i][j]*x[i]*ell_lapse[j]
                  for i in range(3) for j in range(3))
    cal_x = d_x
    cal_y = d_y
    cal_a = tensor((3, 3), zero)
    s_tensor = tensor((3, 3), zero)
    for i in range(3):
        for j in range(3):
            cal_a[i][j] = c("0.5")*r_minus*(cal_y[i][j]
                                             - c("0.5")*ell_lapse[i]*ell_lapse[j])
            s_tensor[i][j] = c("0.5")*alpha*cal_x[i][j] \
                - c("0.25")*alpha*w[i]*w[j] - cal_a[i][j] \
                - c("0.25")*(ell_lapse[i]*x[j] + ell_lapse[j]*x[i])
    trace_cal_a = sum(cal_a[i][i] for i in range(3))
    trace_cal_x = sum(cal_x[i][i] for i in range(3))
    trace_s = sum(s_tensor[i][i] for i in range(3))
    hamiltonian = c(2)*trace_k*trace_k/c(3) - at_sq \
        + c(2)*trace_cal_x - c("2.5")*w_sq
    record("composite/trace_B", trace_b)
    record("composite/At_sq", at_sq)
    record("composite/W_sq", w_sq)
    record("composite/X_dot_L", x_dot_l)
    record("composite/cal_X", cal_x)
    record("composite/cal_Y", cal_y)
    record("composite/cal_A", cal_a)
    record("composite/S", s_tensor)
    record("composite/Hamiltonian", hamiltonian)

    adv_chi = sum(beta[d]*x[d] for d in range(3))
    adv_a = sum(beta[d]*y[d] for d in range(3))
    adv_k = sum(beta[d]*d_k[d] for d in range(3))
    adv_pi = sum(beta[d]*d_pi[d] for d in range(3))
    rhs("chi", {"advection": adv_chi,
                "configuration": c(2)*chi*(alpha*trace_k - trace_b)/c(3)})
    rhs("A", {"advection": adv_a,
              "configuration": c(2)*a*(alpha*pi - h_perp)})
    rhs("K", {"advection": adv_k, "alpha_At2": alpha*at_sq,
              "alpha_K2": alpha*trace_k*trace_k/c(3),
              "minus_trace_calA": -trace_cal_a,
              "X_dot_L": c("0.25")*x_dot_l,
              "hamiltonian": alpha*(hamiltonian - trace_k*c_perp)})
    rhs("pi", {"advection": adv_pi, "minus_alpha_At2": -alpha*at_sq,
               "minus_alpha_K2": -alpha*trace_k*trace_k/c(3),
               "trace_calA": trace_cal_a, "minus_X_dot_L": -c("0.25")*x_dot_l})

    for i in range(3):
        adv_beta = sum(beta[d]*b[d][i] for d in range(3))
        metric_gradient = c("0.5")*sum(gu[i][j]*(a*x[j] - chi*y[j])
                                        for j in range(3))
        rhs(f"beta{i}", {"advection": adv_beta, "gauge": h_source[i],
                         "Lambda": a*chi*lam[i], "metric_gradient": metric_gradient})

        adv_lam = sum(beta[d]*d_lam[d][i] for d in range(3))
        second_shift = zero
        shift_trace_gradient = zero
        minus_at_l = zero
        minus_at_w = zero
        d_k_term = zero
        d_constraint = zero
        for j in range(3):
            d_trace_b = sum(d_b[j][p][p] for p in range(3))
            second_shift += sum(gu[j][p]*d_b[j][p][i] for p in range(3))
            shift_trace_gradient += gu[i][j]*d_trace_b/c(3)
            minus_at_l -= at_uu[i][j]*ell_lapse[j]
            minus_at_w -= c(3)*r_plus*at_uu[i][j]*w[j]
            d_k_term -= c(4)*alpha*gu[i][j]*d_k[j]/c(3)
            d_constraint += alpha*gu[i][j]*(d_pi[j] + d_k[j])
        rhs(f"Lambda{i}", {"advection": adv_lam, "second_shift": second_shift,
                            "shift_trace_gradient": shift_trace_gradient,
                            "minus_At_L": minus_at_l, "minus_At_W": minus_at_w,
                            "dK": d_k_term, "dCperp": d_constraint})

    sym_names = ((0, 0, "xx"), (0, 1, "xy"), (0, 2, "xz"),
                 (1, 1, "yy"), (1, 2, "yz"), (2, 2, "zz"))
    for i, j, suffix in sym_names:
        adv_g = sum(beta[d]*q[d][i][j] for d in range(3))
        adv_at = sum(beta[d]*d_at[d][i][j] for d in range(3))
        shift_g = sum(g[d][i]*b[j][d] + g[d][j]*b[i][d] for d in range(3))
        shift_at = sum(at[d][i]*b[j][d] + at[d][j]*b[i][d] for d in range(3))
        rhs(f"g{suffix}", {"advection": adv_g, "extrinsic": -c(2)*alpha*at[i][j],
                           "shift": shift_g, "trace_shift": -c(2)*g[i][j]*trace_b/c(3)})
        s_tf = s_tensor[i][j] - g[i][j]*trace_s/c(3)
        rhs(f"At{suffix}", {"advection": adv_at, "S_TF": s_tf,
                            "shift": shift_at,
                            "trace_shift": -c(2)*at[i][j]*trace_b/c(3),
                            "At_squared": -c(2)*alpha*aa[i][j],
                            "K_At": alpha*trace_k*at[i][j]})

    d_f_chi = tensor((3,), zero)
    d_f_a = tensor((3,), zero)
    d_f_beta = tensor((3, 3), zero)
    d_f_g = tensor((3, 3, 3), zero)
    for ell in range(3):
        d_trace_b = sum(d_b[ell][p][p] for p in range(3))
        d_f_chi[ell] = c(2)*(x[ell]*(alpha*trace_k - trace_b)
            + chi*(c("0.5")*ell_lapse[ell]*trace_k
                   + alpha*d_k[ell] - d_trace_b))/c(3)
        d_f_a[ell] = c(2)*y[ell]*(alpha*pi - h_perp) \
            + c(2)*a*(c("0.5")*ell_lapse[ell]*pi
                      + alpha*d_pi[ell] - d_h_perp[ell])
        for i in range(3):
            d_f_beta[ell][i] = d_h_source[ell][i] \
                + (y[ell]*chi + a*x[ell])*lam[i] + a*chi*d_lam[ell][i]
            for j in range(3):
                d_f_beta[ell][i] += c("0.5")*gu[i][j]*(
                    y[ell]*x[j] + a*d_x[ell][j]
                    - x[ell]*y[j] - chi*d_y[ell][j])
        for i in range(3):
            for j in range(3):
                d_f_g[ell][i][j] = -ell_lapse[ell]*at[i][j] \
                    - c(2)*alpha*d_at[ell][i][j]
                for p in range(3):
                    d_f_g[ell][i][j] += g[p][i]*d_b[ell][j][p] \
                        + g[p][j]*d_b[ell][i][p]
                d_f_g[ell][i][j] -= c(2)*g[i][j]*d_trace_b/c(3)
    record("gradient_source/dF_chi", d_f_chi)
    record("gradient_source/dF_A", d_f_a)
    record("gradient_source/dF_beta", d_f_beta)
    record("gradient_source/dF_g", d_f_g)

    for ell in range(3):
        rhs(f"X{ell}", {"dF": d_f_chi[ell],
                         "advected_gradient": sum(beta[d]*d_x[d][ell] for d in range(3)),
                         "index_advection": sum(b[ell][d]*x[d] for d in range(3))})
        rhs(f"Y{ell}", {"dF": d_f_a[ell],
                         "advected_gradient": sum(beta[d]*d_y[d][ell] for d in range(3)),
                         "index_advection": sum(b[ell][d]*y[d] for d in range(3))})
        for i in range(3):
            rhs(f"B{ell}{i}", {"dF": d_f_beta[ell][i],
                                "advected_gradient": sum(beta[d]*d_b[d][ell][i]
                                                         for d in range(3)),
                                "index_advection": sum(b[ell][d]*b[d][i]
                                                       for d in range(3))})
            for j in range(i, 3):
                d_trace_b = sum(d_b[ell][p][p] for p in range(3))
                source_lapse = -ell_lapse[ell]*at[i][j] \
                    - c(2)*alpha*d_at[ell][i][j]
                source_shift = sum(g[p][i]*d_b[ell][j][p]
                                   + g[p][j]*d_b[ell][i][p] for p in range(3))
                source_trace = -c(2)*g[i][j]*d_trace_b/c(3)
                rhs(f"Q{ell}{i}{j}", {"source_lapse": source_lapse,
                    "source_shift": source_shift, "source_trace": source_trace,
                    "advected_gradient": sum(beta[d]*d_q[d][ell][i][j]
                                             for d in range(3)),
                    "index_advection": sum(b[ell][d]*q[d][i][j]
                                           for d in range(3))})
    return values, kinds


def fit_power(radii, magnitudes):
    points = [(math.log(float(r)), math.log(float(v)))
              for r, v in zip(radii, magnitudes) if float(v) > 1.0e-280]
    if len(points) < 4:
        return math.nan
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path,
                        help="optional long-form CSV containing every logged magnitude")
    parser.add_argument("--samples", type=int, default=73)
    args = parser.parse_args()
    if args.samples < 25:
        raise ValueError("need at least 25 radial samples")
    mp.mp.dps = 100
    radius_texts = [f"{10.0**x:.17e}"
                    for x in np.linspace(math.log10(1.1e-8), 2.0, args.samples)]
    all_results = {}
    kinds = None
    for arithmetic in ARITHMETICS:
        table = load_table(args.table, arithmetic)
        results = []
        for radius in radius_texts:
            values, this_kinds = evaluate(radius, table, arithmetic)
            results.append(values)
            if kinds is None:
                kinds = this_kinds
            elif this_kinds != kinds:
                raise AssertionError("precision backends logged different temporaries")
        all_results[arithmetic.name] = results

    if args.output:
        with args.output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("precision", "r_over_M", "kind", "name", "magnitude"))
            for precision, results in all_results.items():
                for radius, values in zip(radius_texts, results):
                    for name in sorted(values):
                        writer.writerow((precision, radius, kinds[name], name,
                                         mp.nstr(values[name], 25)))

    inner_count = max(12, args.samples//3)
    inner_r = radius_texts[:inner_count]
    mp_results = all_results["mp100"]
    divergent_rhs = []
    divergent_temporaries = []
    for name in sorted(kinds):
        magnitudes = [row[name] for row in mp_results[:inner_count]]
        power = fit_power(inner_r, magnitudes)
        if math.isfinite(power) and power < -0.25:
            entry = (power, name)
            if kinds[name] == "rhs_term":
                divergent_rhs.append(entry)
            elif kinds[name] == "temporary":
                divergent_temporaries.append(entry)

    if divergent_rhs:
        listing = ", ".join(f"{name} (p={power:.3f})"
                            for power, name in divergent_rhs[:12])
        raise AssertionError(f"divergent additive RHS terms: {listing}")

    sum_names = [name for name, kind in kinds.items() if kind == "rhs_sum"]
    worst_mp_sum = max((float(abs(row[name])), name, radius_texts[i])
                       for i, row in enumerate(mp_results) for name in sum_names)
    if worst_mp_sum[0] > 1.0e-4:
        raise AssertionError("high-precision table-domain RHS residual exceeds 1e-4: "
                             f"{worst_mp_sum}")
    worst_float = (0.0, "")
    worst_long = (0.0, "")
    worst_float_abs = (0.0, "")
    worst_long_abs = (0.0, "")
    for name in sum_names:
        sumabs_name = name[:-3] + "sumabs"
        for i, reference_row in enumerate(mp_results):
            reference = reference_row[name]
            scale = max(reference_row[sumabs_name], mp.mpf("1e-70"))
            absolute64 = abs(mp.mpf(str(all_results["binary64"][i][name])) - reference)
            absoluteld = abs(mp.mpf(str(all_results["long_double"][i][name])) - reference)
            error64 = absolute64/scale
            errorld = absoluteld/scale
            if float(error64) > worst_float[0]:
                worst_float = (float(error64), f"{name}@r={radius_texts[i]}")
            if float(errorld) > worst_long[0]:
                worst_long = (float(errorld), f"{name}@r={radius_texts[i]}")
            if float(absolute64) > worst_float_abs[0]:
                worst_float_abs = (float(absolute64), f"{name}@r={radius_texts[i]}")
            if float(absoluteld) > worst_long_abs[0]:
                worst_long_abs = (float(absoluteld), f"{name}@r={radius_texts[i]}")

    print(f"PASS: Gauge A0 cancellation audit logged {len(kinds)} quantities at "
          f"{args.samples} radii in binary64, long double, and 100-digit arithmetic")
    print("PASS: no additive production RHS term has fitted inner power below -0.25")
    print(f"PASS: maximum 100-digit RHS sum on the open table domain is "
          f"{worst_mp_sum[0]:.3e} at {worst_mp_sum[1]}@r={worst_mp_sum[2]}")
    if divergent_temporaries:
        print("AUDIT: divergent raw derivatives (not additive RHS terms):")
        for power, name in divergent_temporaries:
            print(f"  {name}: fitted power {power:.6f}")
    else:
        print("PASS: no logged raw production temporary has fitted inner power below -0.25")
    print(f"AUDIT: worst term-scale-normalized binary64 RHS-sum discrepancy "
          f"{worst_float[0]:.3e} "
          f"at {worst_float[1]}")
    print(f"AUDIT: worst term-scale-normalized long-double RHS-sum discrepancy "
          f"{worst_long[0]:.3e} "
          f"at {worst_long[1]}")
    print(f"AUDIT: worst absolute binary64 RHS-sum discrepancy {worst_float_abs[0]:.3e} "
          f"at {worst_float_abs[1]}")
    print(f"AUDIT: worst absolute long-double RHS-sum discrepancy {worst_long_abs[0]:.3e} "
          f"at {worst_long_abs[1]}")


if __name__ == "__main__":
    main()
