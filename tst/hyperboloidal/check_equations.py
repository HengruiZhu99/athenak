#!/usr/bin/env python3
"""Independent exact checks of the CMC conformal Einstein sources (SymPy).

This does not prove regularity of a perturbed Z4c system. Spherical coordinates
are used only for symbolic computation; the exported reference is Cartesian.
"""
import sympy as s


def zero(expression):
    result = s.factor(s.trigsimp(expression))
    if result != 0:
        raise AssertionError(result)


def main():
    t, r, theta, phi = s.symbols("t r theta phi", real=True)
    a, radius = s.symbols("a S", positive=True)
    coords = (t, r, theta, phi)
    omega = (radius**2-r**2)/(2*a*radius)
    alpha = (radius**2+r**2)/(2*a*radius)
    # Built directly from ds_phys^2=-dT^2+dR^2+R^2 dOmega_2^2,
    # R=r/omega, T=t+sqrt(R^2+a^2).
    metric = s.Matrix([[-omega**2, -r/a, 0, 0], [-r/a, 1, 0, 0],
                       [0, 0, r*r, 0], [0, 0, 0, r*r*s.sin(theta)**2]])
    inverse = metric.inv().applyfunc(s.factor)
    connection = [[[s.factor(sum(inverse[k, ell]*(s.diff(metric[ell, j], coords[i])
                    + s.diff(metric[ell, i], coords[j])-s.diff(metric[i, j], coords[ell]))
                    for ell in range(4))/2) for j in range(4)]
                   for i in range(4)] for k in range(4)]
    ricci = s.zeros(4)
    for i in range(4):
        for j in range(4):
            ricci[i, j] = s.factor(s.trigsimp(sum(
                s.diff(connection[k][i][j], coords[k])
                - s.diff(connection[k][i][k], coords[j])
                + sum(connection[k][i][j]*connection[ell][k][ell]
                      - connection[ell][i][k]*connection[k][j][ell] for ell in range(4))
                for k in range(4))))
    scalar = s.factor(sum(inverse[i, j]*ricci[i, j]
                          for i in range(4) for j in range(4)))
    einstein = (ricci-metric*scalar/2).applyfunc(s.factor)
    gradient = s.Matrix([s.diff(omega, x) for x in coords])
    hessian = s.Matrix(4, 4, lambda i, j: (
        s.diff(omega, coords[i], coords[j])
        - sum(connection[k][i][j]*gradient[k] for k in range(4))))
    box = s.factor(sum(inverse[i, j]*hessian[i, j]
                       for i in range(4) for j in range(4)))
    norm = s.factor((gradient.T*inverse*gradient)[0])
    source = -2*(hessian-metric*box)/omega-3*metric*norm/omega**2
    for i in range(4):
        for j in range(i, 4):
            zero(einstein[i, j]-source[i, j])
    print("PASS all ten conformal Einstein components")

    normal = s.Matrix([1/alpha, r/(a*alpha), 0, 0])
    energy = 3/(a*a*alpha*alpha)
    momentum = -2*s.diff(alpha, r)/(a*alpha*alpha)
    pressure = -energy+2/(a*radius*alpha)+2*r*r/(a**3*radius*alpha**3)
    zero((normal.T*einstein*normal)[0]-energy)
    zero(-(normal.T*einstein)[1]-momentum)
    zero(einstein[1, 1]-pressure)
    zero(einstein[2, 2]/r**2-pressure)
    zero(einstein[3, 3]/(r*s.sin(theta))**2-pressure)
    for expr in (energy, momentum, pressure):
        limit = s.simplify(s.limit(expr, r, radius, dir="-"))
        assert not limit.has(s.oo, -s.oo, s.zoo, s.nan)
    print("PASS regular reference projections and scri limits")

    # Stable analytic phase used by the radial evolution test.
    retarded = t+a*(radius-r)/(radius+r)
    speed = (radius+r)**2/(2*a*radius)
    zero(s.diff(retarded, t)+speed*s.diff(retarded, r))
    # Physical curvature maps and vacuum Hamiltonian; conformal vacuum would fail.
    kbar = -3/(a*alpha)
    kphys = omega*kbar-3*(-r/a)*s.diff(omega, r)/alpha
    zero(kphys+3/a)
    zero(-6/a**2+s.Rational(2, 3)*kphys**2)
    zero(s.Rational(2, 3)*kbar**2-2*energy)
    assert s.simplify(s.Rational(2, 3)*kbar**2) != 0
    print("PASS characteristic phase, curvature map and missing-source counterexample")


if __name__ == "__main__":
    main()
