"""Exact scalar corner eigenspaces for an unsuppressed Gamma driver."""
import json,sys
from pathlib import Path
import sympy as s
a,c,g,F,E,w,k,t,m=s.symbols('a c g F E w k t m',nonzero=True);z=s.Symbol('z')
P=s.zeros(10)
P[0,3]=2*a*c/3;P[0,4]=4*a*c/3;P[0,7]=-2*c/3
P[1,2]=-2*a;P[1,7]=4*g/3
for j,v in [(0,a/3),(1,-a*c/(2*g)),(5,2*a*c*g/3),(6,-2*c/3)]:P[2,j]=v
P[3,6]=-c/g;P[4,0]=a/g;P[4,5]=a*c/2
for j,v in [(3,-4*a/(3*g)),(4,-2*a/(3*g)),(7,4/(3*g))]:P[5,j]=v
P[6,3]=-F;P[6,8]=E*c*w;P[7,9]=m;P[8,6]=k*w/t;P[9,:]=P[5,:]
poly=s.factor(P.charpoly(z).as_expr())
assert s.simplify(poly-z*z*(z*z-a*a*c/g)**2*(z*z-c*F/g-E*c*k*w*w/t)*(z*z-4*m/(3*g)))==0
records={}
for label,values in [
 ('Minkowski075',{a:s.Rational(1,2),c:1,g:1,F:s.Rational(3,4),E:1,w:1,m:s.Rational(3,4)}),
 ('Minkowski1',{a:s.Rational(1,2),c:1,g:1,F:s.Rational(3,4),E:1,w:1,m:1}),
 ('collapsed',{a:0,c:0,g:1,F:0,E:0,w:1,m:1})]:
 Q=P.subs({**values,k:s.Rational(1,200),t:s.Rational(1,50)})
 records[label]=[dict(value=str(v),algebraic=int(mult),geometric=len(vec)) for v,mult,vec in Q.eigenvects()]
assert any(x['algebraic']!=x['geometric'] for x in records['Minkowski075'])
assert all(x['algebraic']==x['geometric'] for x in records['Minkowski1'])
record=dict(characteristic_polynomial=str(poly),polynomial_identity_verified=True,corners=records,
 scope='Exact selected scalar corners; singular physical reduction and uniform collapsed-puncture well-posedness remain unresolved.')
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
