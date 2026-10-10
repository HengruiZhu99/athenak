"""HELD exact Gaussian wave and bounded outer profiles, complete Taylor jets."""
import mpmath as mp
from taylor3 import Jet, indices


def fderivative(t, sigma, degree):
    x = t/sigma
    exponential = (-x*x/2).exp() if isinstance(x, Jet) else mp.exp(-x*x/2)
    h0, h1 = 1, x
    if degree == 0:
        hermite = h0
    elif degree == 1:
        hermite = h1
    else:
        for k in range(1, degree):
            h0, h1 = h1, x*h1-k*h0
        hermite = h1
    return sigma**(4-degree)*(-1)**degree*hermite*exponential


def wave(T, Q, sigma):
    """F=partial_Q1 partial_Q2 [(f(T-R)-f(T+R))/R].

    At the exact origin this polynomial is exact through order three, including
    F_12 and F_T12. Nonzero radii use the full advanced/retarded expression.
    No finite series, deleted advanced tail or radial floor is used.
    """
    radius_squared = sum(q*q for q in Q)
    value = radius_squared.v if isinstance(radius_squared, Jet) else radius_squared
    if value == 0:
        return -mp.mpf(2)/15*Q[0]*Q[1]*fderivative(T, sigma, 5)
    R = radius_squared**mp.mpf(".5")
    u, v = T-R, T+R
    C = ((fderivative(u, sigma, 2)-fderivative(v, sigma, 2))/R**3
         +3*(fderivative(u, sigma, 1)+fderivative(v, sigma, 1))/R**4
         +3*(fderivative(u, sigma, 0)-fderivative(v, sigma, 0))/R**5)
    return Q[0]*Q[1]*C


def outer_profiles(u, z, sigma):
    v = u+2/z
    f = [fderivative(u, sigma, k) for k in range(4)]
    g = [fderivative(v, sigma, k) for k in range(4)]
    Phi = f[2]-g[2]+3*z*(f[1]+g[1])+3*z*z*(f[0]-g[0])
    plus = -f[2]-6*z*f[1]-9*z*z*f[0]-2*g[3]/z+7*g[2]-12*z*g[1]+9*z*z*g[0]
    minus = 2*f[3]+7*z*f[2]+12*z*z*f[1]+9*z**3*f[0]-z*g[2]+6*z*z*g[1]-9*z**3*g[0]
    return Phi, plus, minus


def implicit_jet(root, function, derivative_at_root, order=3):
    """Formal homogeneous implicit solve at a separately gated scalar root.

    Coefficients of degree d depend linearly on the unknown degree-d block,
    with scalar coefficient G_c. Each lower degree is retained unchanged.
    This is complete multivariate implicit differentiation, without finite
    differences or differentiating a scalar root-finding routine.
    """
    if not (mp.isfinite(derivative_at_root) and derivative_at_root > 0):
        raise ArithmeticError("invalid implicit scalar Jacobian")
    answer = Jet(root, order)
    for degree in range(1, order+1):
        residual = function(answer)
        for m in indices(order):
            if sum(m) == degree:
                answer.c[m] -= residual.c[m]/derivative_at_root
    return answer, function(answer)


def numerical_root(function, center, bound, lower_j, settings):
    """Fixed numerical sign/width certificate; not an interval enclosure."""
    tolerance = mp.mpf(settings["absolute"])
    width = mp.mpf(settings["width"])
    lo, hi = center-bound, center+bound
    calls = 0
    def evaluate(x):
        nonlocal calls
        calls += 1
        g, j = function(x)
        if not (mp.isfinite(g) and mp.isfinite(j) and j > 0):
            raise ArithmeticError("invalid inverse value or Jacobian")
        return g, j
    gl, _ = evaluate(lo)
    gh, _ = evaluate(hi)
    initial = [lo, hi, gl, gh]
    if not (bound >= 0 and gl <= 0 <= gh):
        raise ArithmeticError("inverse numerical endpoint signs fail")
    if bound == 0:
        if abs(gl) > tolerance:
            raise ArithmeticError("exact branch scalar residual fails")
        return center, dict(initial=initial, final=initial, residual=abs(gl),
                            width=mp.mpf(0), calls=calls, iterations=0,
                            formal_interval_enclosure=False)
    x = center
    for count in range(settings["newton"]+settings["bisection"]):
        g, j = evaluate(x)
        if g < 0:
            lo, gl = x, g
        elif g > 0:
            hi, gh = x, g
        if abs(g) <= lower_j*width/8:
            l, h = max(lo, x-width/4), min(hi, x+width/4)
            fl, _ = evaluate(l)
            fh, _ = evaluate(h)
            if fl <= 0 <= fh:
                lo, hi, gl, gh = l, h, fl, fh
        if hi-lo <= width:
            answer = (hi+lo)/2
            residual, _ = evaluate(answer)
            if abs(residual) <= tolerance:
                return answer, dict(initial=initial, final=[lo, hi, gl, gh],
                    residual=abs(residual), width=hi-lo, calls=calls,
                    iterations=count+1, formal_interval_enclosure=False)
        proposal = x-g/j
        x = proposal if count < settings["newton"] and lo < proposal < hi else (hi+lo)/2
    raise ArithmeticError("fixed inverse iterations, signs, width or residual fail")
