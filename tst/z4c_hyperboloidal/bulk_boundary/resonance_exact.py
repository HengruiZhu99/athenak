#!/usr/bin/env python3
"""Exact static spherical first recurrence; see README for scope and normalization."""
import argparse
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--sympy-path',type=Path);args=p.parse_args()
if args.sympy_path:sys.path.insert(0,str(args.sympy_path))
import sympy as S
z=S.symbols('z',positive=True);r=1-z;O=z-z*z/2;al=1-z+z*z/2;k=S.Rational(1,4);R=4
C1,G1,A1,P1,l1,b1,L=S.symbols('C1 G1 A1 P1 l1 b1 L')
c=2*z**3+(C1+L*S.log(z)/2)*z**4
g=z**3+G1*z**4
a=S.Rational(21,2)*z**2+(A1+L*S.Rational(10,3)*S.log(z))*z**3
p=(P1+L*S.Rational(5,4)*S.log(z))*z**4
l=3*z**3+(l1+L*S.log(z))*z**4
b=-S.Rational(9,2)*z**3+(b1-L*S.Rational(5,4)*S.log(z))*z**4
d=lambda f:-S.diff(f,z)
D=d(g)+3*g/r-z**3
F=al*al+4*O*O*al
rhs_chi=-r*d(c)-S.Rational(2,3)*(d(b)+2*b/r)+S.Rational(2,3)*(al*p-3*l-3*r*b)/O
rhs_g=-2*al*a-r*d(g)+S.Rational(4,3)*(d(b)-b/r)
H=O*O*(d(d(g))+5*d(g)/r+3*g/r**2+2*d(d(c))+4*d(c)/r)-4*p+4*O*(-3*c+r*d(g)+3*g+r*d(c)/2)-6*r*r*(c-g)
M=O*(d(a)+3*a/r)-S.Rational(2,3)*d(p)+2*r*a
rhs_b=-r*d(b)-b+S.Rational(3,2)*F*D-R*b/O
rhs_l=-r*d(l)+r*b+(-F*p-2*k*al*l+16*O**4*r*r/al*l+r*al*b-r*r*l)/O
es=[]
for f,n in [(rhs_chi,3),(rhs_g,3),(M,3),(H,4),(rhs_b,3),(rhs_l,3)]:
 e=S.expand(S.series(f,z,0,n+1).removeO()).coeff(z,n);es.append(S.simplify(e))
solution=next(iter(S.linsolve(es,[C1,G1,A1,P1,l1,b1,L])))
assert solution[-1]==S.Rational(120,13)
assert all(S.simplify(e.subs(dict(zip([C1,G1,A1,P1,l1,b1,L],solution))))==0 for e in es)
# The auxiliary relaxation gives D-Lambda=-rho^3 at this order.
assert S.expand(r*d(-z**3)+D).coeff(z,2)==0
result=dict(equations=[str(e) for e in es],variables=['C1','G1','A1','P1','l1','b1','L'],solution=[str(v) for v in solution],log_coefficient=str(solution[-1]),scope='Static first radial recurrence about the flat reference, unit leading s=3 radial metric amplitude. Not a global nonlinear solution.')
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
