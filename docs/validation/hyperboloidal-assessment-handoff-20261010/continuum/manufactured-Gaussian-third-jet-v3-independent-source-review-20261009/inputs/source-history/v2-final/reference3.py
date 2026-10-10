"""HELD analytic reference jets. Height quadrature supplies its scalar only."""
import mpmath as mp
from taylor3 import Jet, compose_radial


def radial_coefficients(radius, a, r0, r1):
    """Return complete univariate order-three Omega,b,h and order-two L.

    Axis zero in this local object denotes radial differentiation, not time.
    The endpoint formulas have exactly zero cutoff jets. Inside the transition
    both logistic tails are formed directly from the smaller exponential.
    """
    r = Jet.variable(radius, 0)
    outer = (1-r)*(1+r)/(2*a)
    if radius <= r0:
        w, wc = Jet(0), Jet(1)
        omega, b = Jet(1), Jet(0)
        branch = "exact_Cauchy"
    elif radius >= r1:
        w, wc = Jet(1), Jet(0)
        omega, b = outer, r/a
        branch = "exact_CMC"
    else:
        s, t = (r-r0)/(r1-r0), (r1-r)/(r1-r0)
        logit = -1/s+1/t
        if logit.v <= 0:
            e = logit.exp()
            w, wc = e/(1+e), 1/(1+e)
        else:
            e = (-logit).exp()
            w, wc = 1/(1+e), e/(1+e)
        omega, b = wc+w*outer, r*w/a
        branch = "analytic_transition"
    h = (omega*omega+b*b)**mp.mpf(".5")
    L = omega.truncate(2)-r.truncate(2)*omega.diff(0)
    if not (omega.v > 0 and h.v > 0 and L.v > 0):
        raise ArithmeticError("invalid stationary reference coefficients")
    return dict(r=r, omega=omega, b=b, h=h, L=L, w=w, wc=wc, branch=branch)


def lift_radial(coefficient, radius):
    if coefficient.order != 3:
        raise ValueError("only complete third radial jets may be lifted")
    return compose_radial(coefficient.v, coefficient.derivative(0),
                          coefficient.derivative(0, 0),
                          coefficient.derivative(0, 0, 0), radius)


def make_reference(event, layer):
    """Return native Cartesian Omega3, embedding3, analytic H3 and ADM2."""
    variables = [Jet.variable(value, axis) for axis, value in enumerate(event)]
    t, x = variables[0], variables[1:]
    rvalue = mp.sqrt(mp.fsum(value*value for value in event[1:]))
    radial = radial_coefficients(rvalue, layer.a, layer.r0, layer.r1)
    if rvalue <= layer.r0:
        # H=0 and Omega=1 on an open neighborhood; avoid |x| at x=0.
        omega, H = Jet(1), Jet(0)
        radius = None  # the radius is never differentiated at the origin
        if rvalue != 0:
            radius = sum(v*v for v in x)**mp.mpf(".5")
        Q = x
        b, h, L = Jet(0), Jet(1), Jet(1, 2)
    else:
        radius = sum(v*v for v in x)**mp.mpf(".5")
        omega = lift_radial(radial["omega"], radius)
        b, h = lift_radial(radial["b"], radius), lift_radial(radial["h"], radius)
        # H_r=b L/(h Omega^2); its order-two jet gives H through three.
        Hr = radial["b"].truncate(2)*radial["L"]/(radial["h"].truncate(2)*radial["omega"].truncate(2)**2)
        H = compose_radial(layer.height(rvalue), Hr.v, Hr.derivative(0),
                           Hr.derivative(0, 0), radius)
        Q = [v/omega for v in x]
        L = omega.truncate(2)-sum(x[i].truncate(2)*omega.diff(i+1) for i in range(3))
    B = t+H
    return dict(event=event, variables=variables, t=t, x=x, radius=radius,
                rvalue=rvalue, radial=radial, omega=omega, b=b, h=h, L=L,
                H=H, Q=Q, B=B, embedding=[B]+Q)


def height_physical_derivatives(reference):
    """H_R,H_RR,H_RRR, with d/dR=(Omega^2/L)d/dr."""
    radial = reference["radial"]
    if reference["rvalue"] <= mp.mpf(".05"):
        return mp.mpf(0), mp.mpf(0), mp.mpf(0)
    hr = radial["b"]/radial["h"]
    factor = radial["omega"].truncate(2)**2/radial["L"]
    hrr = factor*hr.diff(0)
    hrrr = factor.truncate(1)*hrr.diff(0)
    return hr.v, hrr.v, hrrr.v


def reference_adm(reference):
    """Independent stationary ADM2 formulas, without embedding differentiation.

    bar(gamma)=I+(L^2/h^2-1)nn, alpha=h, beta=-(bh/L)n.
    The exact core branch avoids all divisions by the Cartesian radius.
    """
    if reference['rvalue'] <= mp.mpf('.05'):
        return Jet(1,2), [Jet(0,2) for _ in range(3)], [
            [Jet(int(i==j),2) for j in range(3)] for i in range(3)]
    n = [q/reference['radius'] for q in reference['x']]
    h,b,L = reference['h'].truncate(2),reference['b'].truncate(2),reference['L']
    gamma = [[int(i==j)+(L*L/(h*h)-1)*n[i]*n[j] for j in range(3)] for i in range(3)]
    beta = [-b*h/L*n[i] for i in range(3)]
    return h,beta,gamma
