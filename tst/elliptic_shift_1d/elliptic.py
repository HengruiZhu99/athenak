"""Exact nonlinear spherical connection-freezing/damping shift, regular center.

Notation and uniqueness proof: docs/elliptic-shift-1d/nonlinear-formulation.md.
Lambda is deliberately absent: the gauge uses metric-derived V, not evolved Lambda.
"""
import numpy as np


def connection(r, a, da):
    # sqrt(a)-1/a = (a**1.5-1)/a; expm1 prevents loss near a=1.
    return da/a**2 + 2*np.expm1(1.5*np.log(a))/(a*r)


def solve(ops, a, da, dda, alpha, dalpha, arr, darr, outer_beta, eta=0.):
    r = ops.r
    if eta < 0 or not np.isfinite(outer_beta) or np.any(a <= 0):
        raise ValueError('Invalid elliptic coefficients/boundary')
    V = connection(r, a, da)
    F = ops.integral(np.expm1(1.5*np.log(a))/r)
    mu = r**3*np.exp(F)/a**2
    Q = -eta*ops.integral(r**3*np.exp(F)*V)/mu
    P = -2*da/a+(a**1.5+2)/r
    dQ = -P*Q-eta*a*a*V
    # nu beta = a**(3/4) beta/r. Integrate from outer boundary inward.
    integrand = .75*a**(-.25)*(Q+2*alpha*arr)/r
    if getattr(ops,'elliptic_quadrature','prefix')=='inward':tail=ops.tail_integral(integrand)
    else:
        integral=ops.integral(integrand);tail=integral[-1]-integral
    b = r*a**(-.75)*(a[-1]**.75*outer_beta/r[-1]-tail)
    b[-1] = outer_beta  # Impose the stated Dirichlet gauge datum exactly.
    B = .75*da/a-1/r
    source = .75*(Q+2*alpha*arr)/a
    db = source-B*b
    dB = .75*(dda/a-(da/a)**2)+1/r**2
    dsource = .75*((dQ+2*dalpha*arr+2*alpha*darr)/a
                   -(Q+2*alpha*arr)*da/a**2)
    ddb = dsource-dB*b-B*db
    if not all(np.all(np.isfinite(x)) for x in (b, db, ddb, Q)):
        raise FloatingPointError('Nonfinite elliptic solution')
    return b, db, ddb, Q


def residual(r, a, da, dda, alpha, dalpha, arr, darr, beta, dbeta, ddbeta, eta):
    """Expanded second-order operator, independent of quadrature implementation."""
    P = -2*da/a+(a**1.5+2)/r
    A = 4*a/3
    B = da-4*a/(3*r)
    dA = 4*da/3
    dB = dda-4*da/(3*r)+4*a/(3*r*r)
    return (A*ddbeta+(dA+B+P*A)*dbeta+(dB+P*B)*beta
            -2*(dalpha*arr+alpha*darr)-2*P*alpha*arr
            +eta*a*a*connection(r, a, da))
