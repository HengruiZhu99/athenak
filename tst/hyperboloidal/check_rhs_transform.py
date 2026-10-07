#!/usr/bin/env python3
"""Exact chain-rule audit of the C_Z4c=0 tensor-system variable transformation.

Starting expressions are the untransformed trace, Theta, A and Lambda terms
of arXiv:1412.3827 Appendix B. Gauge time derivatives are independent symbols:
their cancellation must hold without assuming a stationary solution.
"""
import sympy as s


def zero(expr):
    result = s.factor(expr)
    if result != 0:
        raise AssertionError(result)


def main():
    o, alpha = s.symbols("Omega alpha", positive=True)
    p, t, w, aa, ricci, divz = s.symbols("P T w A2 Rbar chi_divZ")
    la, lo, cross, grad2 = s.symbols("lap_alpha lap_Omega grad_alpha_Omega grad_Omega2")
    k1, k2, aperp, osecond = s.symbols("kappa1 kappa2 alpha_perp Omega_perpperp")
    k = (p-3*w)/o
    theta = t/o
    # Perpendicular derivatives, i.e. time derivative minus shift Lie derivative.
    kdot = alpha*(aa+(k+2*theta)**2/3+k1*(1-k2)*theta/o)-la
    kdot += 3*alpha*(w*w-grad2)/o**2+3*cross/o+alpha*lo/o
    kdot += (k+4*theta)*alpha*w/o+3*aperp*w/(o*alpha)-3*osecond/(o*alpha)
    wdot = osecond/alpha-w*aperp/alpha
    pdot = o*kdot+alpha*w*k+3*wdot
    expected_p = o*(alpha*aa-la)+3*cross+alpha*lo
    expected_p += alpha*((p+2*t)**2/3-3*grad2+k1*(1-k2)*t)/o
    zero(pdot-expected_p)
    assert not s.factor(pdot).has(aperp, osecond)
    print("PASS transformed trace: gauge time derivatives and double poles cancel")

    thdot = alpha*(ricci+2*divz-aa+s.Rational(2, 3)*(k+2*theta)**2
                   - 2*k1*(2+k2)*theta/o)/2
    thdot += 2*alpha*lo/o+3*alpha*(w*w-grad2)/o**2 + 2*k*alpha*w/o
    tdot = o*thdot+alpha*w*theta
    expected_t = o*alpha*(ricci-aa+2*divz)/2+2*alpha*lo
    expected_t += alpha*((p+2*t)**2/3-3*grad2-(3*w+k1*(2+k2))*t)/o
    zero(tdot-expected_t)
    print("PASS transformed Theta, including the off-constraint -3*w*T/Omega term")

    # One raised spatial component; all derivatives are independent variables.
    dp, dt, dw, do = s.symbols("dP dT dw dOmega")
    dk = (dp-3*dw)/o-(p-3*w)*do/o**2
    dtheta = dt/o-t*do/o**2
    gamma_terms = -alpha*(4*dk+2*dtheta)/3
    gamma_terms -= 2*alpha*(2*k+theta)*do/(3*o)+4*alpha*dw/o
    zero(gamma_terms+2*alpha*(2*dp+dt)/(3*o))
    z = s.symbols("Z_raised_tilde")
    gamma_z = -4*alpha*z*(k+2*theta)/3-2*alpha*k1*z/o-4*alpha*z*w/o
    zero(gamma_z+alpha*(s.Rational(4, 3)*(p+2*t)+2*k1)*z/o)
    # Trace-free extrinsic-curvature coefficient.
    zero(alpha*(k+2*theta)+2*alpha*w/o-alpha*(p+2*t-w)/o)
    print("PASS transformed Lambda and trace-free curvature pole coefficients")


if __name__ == "__main__":
    main()
