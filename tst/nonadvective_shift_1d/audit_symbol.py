"""Exact scalar factorization and incoming boundary-vector checks."""
import argparse,json,sys
from pathlib import Path
import sympy as s
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a0,c,g,F,E,w,k,t,m,b=s.symbols('a c g F E w k t m b',nonzero=True);z=s.Symbol('z')
L=s.zeros(9)
L[0,3]=2*a0*c/3;L[0,4]=4*a0*c/3;L[0,7]=-2*c/3
L[1,2]=-2*a0;L[1,7]=4*g/3
for j,v in [(0,a0/3),(1,-a0*c/(2*g)),(5,2*a0*c*g/3),(6,-2*c/3)]:L[2,j]=v
L[3,6]=-c/g;L[4,0]=a0/g;L[4,5]=a0*c/2
for j,v in [(3,-4*a0/(3*g)),(4,-2*a0/(3*g)),(7,4/(3*g))]:L[5,j]=v
L[6,3]=-F;L[6,8]=E*c*w;L[8,6]=k*w/t
L+=b*s.eye(9);L[7,:]=s.zeros(1,9);L[7,5]=m
poly=s.factor(L.charpoly(z).as_expr())
expected=(z-b)*((z-b)**2-a0*a0*c/g)**2*((z-b)**2-c*F/g-E*c*k*w*w/t)*(z*z-b*z-4*m/(3*g))
assert s.simplify(poly-expected)==0
# Gauge vector is exact for either nonzero shift eigenvalue.
lam=s.symbols('lambda',nonzero=True);v=s.zeros(9,1);v[0]=-.5*c*g;v[1]=g*g;v[5]=1;v[7]=m/lam
v=v.applyfunc(s.nsimplify)
res=(L-lam*s.eye(9))*v
for entry in res:assert s.simplify(entry.subs(m,3*g*lam*(lam-b)/4))==0
assert s.simplify(v[5]-v[1]/g**2)==0
corners={}
for mu in [s.Rational(3,2),s.Rational(9,4)]:
 Q=L.subs({a0:1,c:1,g:1,F:1,E:1,w:0,k:s.Rational(1,200),t:s.Rational(1,50),m:mu,b:-1})
 rows=[dict(value=str(e),algebraic=int(mult),geometric=len(vec)) for e,mult,vec in Q.eigenvects()]
 corners[str(mu)]=dict(eigenspaces=rows,complete=all(x['algebraic']==x['geometric'] for x in rows))
assert not corners['3/2']['complete'] and corners['9/4']['complete']
# Outer margin / g = cL^2 + beta^2 - |beta| cL > 0 for cL>0.
x,y=s.symbols('x y',positive=True)
assert s.expand(x*x+y*y-x*y-((x-y/2)**2+3*y*y/4))==0
record=dict(characteristic_polynomial=str(poly),factorization_exact=True,incoming_vector_exact=True,
 preserves_connection_constraint_trace=True,scri_corners=corners,
 additional_reference_mode='b has coordinate speed zero; b_r reduction adds one tangent scalar mode',
 separation='4 mu/(3g) > cL^2 + abs(beta)*cL; outer choice gives cL^2+beta^2-abs(beta)*cL > 0',
 scope='Scalar frozen symbol only; no full 3D or continuum boundary well-posedness claim.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
