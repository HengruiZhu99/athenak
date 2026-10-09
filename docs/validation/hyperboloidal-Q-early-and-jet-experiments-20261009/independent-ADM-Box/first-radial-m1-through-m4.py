"""Independent ADM/4D-Box identity; no actual-kernel source or artifact edits."""
import json
import sympy as s

r, a, sigma = s.symbols('r a sigma', positive=True)
omega = (1-r*r)/(2*a)
h = (1+r*r)/(2*a)
beta = -r/a
op = s.diff(omega, r)
Bh = beta*op
g4rr = 1-beta*beta/(h*h)


def div(vector):
    return s.diff(r*r*vector, r)/(r*r)


boxref = div(h*g4rr*op)/h


def rate(f, b):
    gamma_nn = 2*s.diff(b, r)+2*(f/a-b*op)/omega
    gamma_trace = 2*s.diff(b, r)+4*b/r+6*(f/a-b*op)/omega
    delta_g4rr = -2*beta*b/(h*h)+2*beta*beta*f/(h*h*h)
    delta_N = delta_g4rr*op*op
    delta_spdiv = f*div(h*g4rr*op)+h*div((f*g4rr+h*delta_g4rr)*op)
    delta_box = sigma*delta_N/omega
    ndot = (-op*op*gamma_nn-2*Bh*delta_box
            -4*Bh*f*boxref/h+(Bh*Bh/(h*h))*gamma_trace
            +2*Bh*delta_spdiv/(h*h))
    return s.factor(delta_N), s.factor(ndot)


rows = []
for m in (1, 2, 3, 4):
    for name, f, b in (
        ('radial_shift', s.Integer(0), omega**m),
        ('lapse', omega**m, s.Integer(0)),
        ('matched_radial', omega**m, -omega**m),
        ('exact_proportional', omega**m, beta*omega**m/h),
    ):
        dn, dt = rate(f, b)
        leading = s.simplify(s.limit(dt/omega**(m-1), r, 1, dir='-'))
        n0 = s.simplify(s.limit(dt, r, 1, dir='-'))
        n1 = s.simplify(s.limit((dt-n0)/omega, r, 1, dir='-'))
        if name == 'radial_shift':
            assert s.simplify(leading-4*(m+1-sigma)/a**3) == 0
        rows.append(dict(case=name, m=m, deltaN=str(dn), deltaNdot=str(dt),
                         leading=str(leading), N0dot=str(n0), N1dot=str(n1)))
print(json.dumps({'scope':'linear Einstein gauge-only initial data; exact ADM metric rate and prescribed preferred Box source; positiveOmega radial identity',
                  'rows':rows}, indent=2, allow_nan=False))
