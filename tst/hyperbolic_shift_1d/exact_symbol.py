"""Exact scalar factorization and endpoint eigenspaces; needs SymPy."""
import json,sys
from pathlib import Path
import sympy as s
a,c,g,F,E,w,k,t,m=s.symbols('a c g F E w k t m',nonzero=True);z=s.Symbol('z')
P=s.zeros(10);P[0,3]=2*a*c/3;P[0,4]=4*a*c/3;P[0,7]=-2*c/3;P[1,2]=-2*a;P[1,7]=4*g/3
for j,v in [(0,a/3),(1,-a*c/(2*g)),(5,2*a*c*g/3),(6,-2*c/3)]:P[2,j]=v
P[3,6]=-c/g;P[4,0]=a/g;P[4,5]=a*c/2
for j,v in [(3,-4*a/(3*g)),(4,-2*a/(3*g)),(7,4/(3*g))]:P[5,j]=v
P[6,3]=-F;P[6,8]=E*c*w;P[7,9]=m;P[8,6]=k*w/t;P[9,:]=P[5,:]
poly=s.factor(P.charpoly(z).as_expr())
expected=z*z*(z*z-a*a*c/g)**2*(z*z-c*F/g-E*c*k*w*w/t)*(z*z-4*m/(3*g))
assert s.simplify(poly-expected)==0
corners={}
for ratio in [s.Rational(1,2),s.Integer(1),s.Integer(2)]:
 Q=P.subs({a:1,c:1,g:1,F:1,E:1,w:0,k:s.Rational(1,200),t:s.Rational(1,50),m:3*ratio/4})
 corners[str(ratio)]=[dict(value=str(e),algebraic=int(mult),geometric=len(vec)) for e,mult,vec in Q.eigenvects()]
collapsed=P.subs({a:0,c:0,g:1,F:0,E:0,w:1,k:s.Rational(1,200),t:s.Rational(1,50),m:0})
assert collapsed**2==s.zeros(10) and collapsed.rank()>0
record=dict(matrix=str(P),characteristic_polynomial=str(poly),identity_verified=True,
 exact_scri_eigenspaces=corners,exact_collapsed_limit=dict(rank=collapsed.rank(),square_zero=True,
 geometric_zero_multiplicity=10-collapsed.rank(),algebraic_zero_multiplicity=10),
 scope='Exact scalar frozen principal symbol; no uniform puncture or endpoint well-posedness claim')
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(poly)
