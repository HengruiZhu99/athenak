#!/usr/bin/env python3
"""Three-precision puncture audit for conformally flat Bowen-York leading fields.

The time-symmetric case is exact Schwarzschild wormhole data.  Momentum and spin cases
use the standard analytic Bowen-York conformal extrinsic curvature with psi=1+M/(2r).
Those latter fields are a near-puncture regularity model, not a Hamiltonian-constraint
solution: the regular TwoPunctures correction is deliberately not invented here.
"""

from __future__ import annotations

import math
from typing import Callable

import mpmath as mp
import numpy as np


class Arithmetic:
    def __init__(self, name: str, cast: Callable, sqrt: Callable):
        self.name = name
        self.cast = cast
        self.sqrt = sqrt


ARITHMETICS = (
    Arithmetic("binary64", np.float64, np.sqrt),
    Arithmetic("long_double", np.longdouble, np.sqrt),
    Arithmetic("mp100", mp.mpf, mp.sqrt),
)
CASES = {
    "time_symmetric": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "momentum": ((0.2, -0.1, 0.05), (0.0, 0.0, 0.0)),
    "spin": ((0.0, 0.0, 0.0), (0.03, 0.04, -0.02)),
    "combined": ((0.2, -0.1, 0.05), (0.03, 0.04, -0.02)),
}


def zeros(*shape, zero):
    if len(shape) == 1:
        return [zero for _ in range(shape[0])]
    return [zeros(*shape[1:], zero=zero) for _ in range(shape[0])]


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def levi_civita(i: int, j: int, k: int) -> int:
    if len({i, j, k}) < 3:
        return 0
    return 1 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def evaluate(radius_text: str, arithmetic: Arithmetic,
             momentum_values, spin_values):
    c = arithmetic.cast
    zero, one = c(0), c(1)
    r = c(radius_text)
    mass = one
    a = mass/c(2)
    raw_direction = [c(1), c(2), c(3)]
    direction_norm = arithmetic.sqrt(sum(x*x for x in raw_direction))
    n = [x/direction_norm for x in raw_direction]
    momentum = [c(x) for x in momentum_values]
    spin = [c(x) for x in spin_values]
    delta = [[one if i == j else zero for j in range(3)] for i in range(3)]

    psi = one + a/r
    chi = one/(psi**4)
    alpha = arithmetic.sqrt(chi)
    lapse_sq = chi
    sqrt_chi = alpha
    chi_r = c(4)*a*r**3/(r + a)**5
    chi_rr = c(4)*a*r*r*(c(3)*a - c(2)*r)/(r + a)**6
    x = [chi_r*n[i] for i in range(3)]
    y = list(x)
    w = [value/sqrt_chi for value in x]
    ell_lapse = list(w)
    d_x = zeros(3, 3, zero=zero)
    for d in range(3):
        for i in range(3):
            d_x[d][i] = chi_rr*n[d]*n[i] \
                + chi_r*(delta[d][i] - n[d]*n[i])/r
    d_y = [[value for value in row] for row in d_x]

    p_dot_n = sum(momentum[i]*n[i] for i in range(3))
    pbar = zeros(3, 3, zero=zero)
    sbar = zeros(3, 3, zero=zero)
    d_pbar = zeros(3, 3, 3, zero=zero)
    d_sbar = zeros(3, 3, 3, zero=zero)
    spin_cross_n = [sum(c(levi_civita(i, k, ell))*spin[k]*n[ell]
                            for k in range(3) for ell in range(3)) for i in range(3)]
    for i in range(3):
        for j in range(3):
            p_tensor = momentum[i]*n[j] + momentum[j]*n[i] \
                - (delta[i][j] - n[i]*n[j])*p_dot_n
            pbar[i][j] = c("1.5")*p_tensor/(r*r)
            spin_tensor = spin_cross_n[i]*n[j] + spin_cross_n[j]*n[i]
            sbar[i][j] = c(3)*spin_tensor/(r**3)
            for d in range(3):
                dn = [(delta[d][q] - n[d]*n[q])/r for q in range(3)]
                d_p_dot_n = sum(momentum[q]*dn[q] for q in range(3))
                d_p_tensor = momentum[i]*dn[j] + momentum[j]*dn[i] \
                    + (dn[i]*n[j] + n[i]*dn[j])*p_dot_n \
                    - (delta[i][j] - n[i]*n[j])*d_p_dot_n
                d_pbar[d][i][j] = c("1.5")*(
                    -c(2)*n[d]*p_tensor/(r**3) + d_p_tensor/(r*r))
                d_cross = [sum(c(levi_civita(q, k, ell))*spin[k]*dn[ell]
                                   for k in range(3) for ell in range(3))
                           for q in range(3)]
                d_spin_tensor = d_cross[i]*n[j] + spin_cross_n[i]*dn[j] \
                    + d_cross[j]*n[i] + spin_cross_n[j]*dn[i]
                d_sbar[d][i][j] = c(3)*(
                    -c(3)*n[d]*spin_tensor/(r**4) + d_spin_tensor/(r**3))

    conformal_factor = one/(psi**6)
    conformal_factor_r = c(6)*a/(r*r*psi**7)
    at = zeros(3, 3, zero=zero)
    d_at = zeros(3, 3, 3, zero=zero)
    for i in range(3):
        for j in range(3):
            bar = pbar[i][j] + sbar[i][j]
            at[i][j] = conformal_factor*bar
            for d in range(3):
                d_at[d][i][j] = conformal_factor_r*n[d]*bar \
                    + conformal_factor*(d_pbar[d][i][j] + d_sbar[d][i][j])

    values = {}
    kinds = {}

    def record(name, value, kind="temporary"):
        values[name] = max((abs(item) for item in flatten(value)), default=zero)
        kinds[name] = kind

    def rhs(component, terms):
        total = sum(terms.values(), zero)
        for name, value in terms.items():
            record(f"rhs/{component}/{name}", value, "rhs_term")
        record(f"rhs/{component}/sum", total, "rhs_sum")
        record(f"rhs/{component}/sumabs", sum(abs(v) for v in terms.values()),
               "rhs_sumabs")

    trace_at = sum(at[i][i] for i in range(3))
    at_sq = sum(at[i][j]*at[i][j] for i in range(3) for j in range(3))
    aa = [[sum(at[i][p]*at[p][j] for p in range(3)) for j in range(3)]
          for i in range(3)]
    w_sq = sum(value*value for value in w)
    x_dot_l = sum(x[i]*ell_lapse[i] for i in range(3))
    trace_cal_x = sum(d_x[i][i] for i in range(3))
    cal_a = zeros(3, 3, zero=zero)
    s_tensor = zeros(3, 3, zero=zero)
    for i in range(3):
        for j in range(3):
            cal_a[i][j] = alpha/c(2)*(d_y[i][j]
                                             - ell_lapse[i]*ell_lapse[j]/c(2))
            s_tensor[i][j] = alpha*d_x[i][j]/c(2) \
                - alpha*w[i]*w[j]/c(4) - cal_a[i][j] \
                - (ell_lapse[i]*x[j] + ell_lapse[j]*x[i])/c(4)
    trace_cal_a = sum(cal_a[i][i] for i in range(3))
    trace_s = sum(s_tensor[i][i] for i in range(3))
    hamiltonian = -at_sq + c(2)*trace_cal_x - c("2.5")*w_sq

    record("state/chi", chi, "state")
    record("state/A", lapse_sq, "state")
    record("state/Atilde", at, "state")
    record("state/X", x, "state")
    record("state/Y", y, "state")
    record("state/gtilde", delta, "state")
    record("derived/W", w)
    record("derived/L", ell_lapse)
    record("derived/r_minus", alpha)
    record("derived/r_plus", one)
    record("derivative/dX", d_x)
    record("derivative/dY", d_y)
    record("derivative/dAtilde", d_at)
    record("composite/Atilde_trace", trace_at)
    record("composite/Atilde_sq", at_sq)
    record("composite/AA", aa)
    record("composite/cal_A", cal_a)
    record("composite/S", s_tensor)
    record("composite/Hamiltonian", hamiltonian)
    record("constraint/H_plus_At_sq", hamiltonian + at_sq, "identity")

    rhs("chi", {"configuration": zero})
    rhs("A", {"configuration": zero})
    rhs("K", {"alpha_At2": alpha*at_sq,
               "minus_trace_calA": -trace_cal_a,
               "X_dot_L": x_dot_l/c(4),
               "hamiltonian": alpha*hamiltonian})
    rhs("pi", {"minus_alpha_At2": -alpha*at_sq,
                "trace_calA": trace_cal_a,
                "minus_X_dot_L": -x_dot_l/c(4)})
    for i in range(3):
        rhs(f"beta{i}", {"metric_gradient": (lapse_sq*x[i] - chi*y[i])/c(2)})
        rhs(f"Lambda{i}", {
            "minus_At_L": -sum(at[i][j]*ell_lapse[j] for j in range(3)),
            "minus_3At_W": -c(3)*sum(at[i][j]*w[j] for j in range(3)),
        })
    for i in range(3):
        for j in range(i, 3):
            suffix = f"{i}{j}"
            rhs(f"g{suffix}", {"extrinsic": -c(2)*alpha*at[i][j]})
            rhs(f"At{suffix}", {"S_TF": s_tensor[i][j] - delta[i][j]*trace_s/c(3),
                                  "At_squared": -c(2)*alpha*aa[i][j]})
            for ell in range(3):
                rhs(f"Q{ell}{suffix}", {
                    "source_lapse_gradient": -ell_lapse[ell]*at[i][j],
                    "source_Atilde_gradient": -c(2)*alpha*d_at[ell][i][j],
                })
    for ell in range(3):
        rhs(f"X{ell}", {"dF": zero})
        rhs(f"Y{ell}", {"dF": zero})
        for i in range(3):
            rhs(f"B{ell}{i}", {"dF": zero})
    return values, kinds


def fitted_power(radii, values):
    points = [(math.log(float(r)), math.log(float(v)))
              for r, v in zip(radii, values) if float(v) > 1.0e-280]
    if len(points) < 5:
        return math.nan
    return float(np.polyfit([x for x, _ in points], [y for _, y in points], 1)[0])


def main() -> None:
    mp.mp.dps = 100
    radii = [f"{10.0**power:.17e}" for power in np.linspace(-8.0, 2.0, 81)]
    for case, (momentum, spin) in CASES.items():
        all_results = {}
        kinds = None
        for arithmetic in ARITHMETICS:
            rows = []
            for radius in radii:
                values, these_kinds = evaluate(radius, arithmetic, momentum, spin)
                rows.append(values)
                if kinds is None:
                    kinds = these_kinds
                elif kinds != these_kinds:
                    raise AssertionError("precision backends logged different quantities")
            all_results[arithmetic.name] = rows

        inner = 27
        divergent = []
        for name, kind in kinds.items():
            power = fitted_power(radii[:inner],
                                 [row[name] for row in all_results["mp100"][:inner]])
            if math.isfinite(power) and power < -0.25 and kind in {"state", "temporary", "rhs_term"}:
                divergent.append((power, name))
        if divergent:
            raise AssertionError(f"{case}: divergent production quantities {divergent[:8]}")

        identity_max = max(float(row["constraint/H_plus_At_sq"])
                           for row in all_results["mp100"])
        if identity_max > 1.0e-80:
            raise AssertionError(f"{case}: conformal Hamiltonian identity residual {identity_max}")
        sum_names = [name for name, kind in kinds.items() if kind == "rhs_sum"]
        worst64 = (0.0, "")
        worstld = (0.0, "")
        for index, reference_row in enumerate(all_results["mp100"]):
            for name in sum_names:
                scale = max(reference_row[name[:-3] + "sumabs"], mp.mpf("1e-80"))
                error64 = abs(mp.mpf(str(all_results["binary64"][index][name]))
                              - reference_row[name])/scale
                errorld = abs(mp.mpf(str(all_results["long_double"][index][name]))
                              - reference_row[name])/scale
                worst64 = max(worst64, (float(error64), f"{name}@r={radii[index]}"))
                worstld = max(worstld, (float(errorld), f"{name}@r={radii[index]}"))
        print(f"PASS: {case}: {len(kinds)} quantities at 81 radii; no stored field, "
              "temporary, or additive RHS term has fitted inner power below -0.25")
        print(f"  max |H+Atilde^2| (mp100)={identity_max:.3e}")
        print(f"  worst normalized binary64 RHS discrepancy={worst64[0]:.3e} at {worst64[1]}")
        print(f"  worst normalized long-double RHS discrepancy={worstld[0]:.3e} at {worstld[1]}")
    print("PASS: Bowen-York puncture regularity/source-cancellation audit")


if __name__ == "__main__":
    main()
