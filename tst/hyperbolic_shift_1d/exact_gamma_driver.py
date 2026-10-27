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
 ('Minkowski2',{a:s.Rational(1,2),c:1,g:1,F:s.Rational(3,4),E:1,w:1,m:2}),
 ('Minkowski4',{a:s.Rational(1,2),c:1,g:1,F:s.Rational(3,4),E:1,w:1,m:4}),
 ('collapsed',{a:0,c:0,g:1,F:0,E:0,w:1,m:1})]:
 Q=P.subs({**values,k:s.Rational(1,200),t:s.Rational(1,50)})
 records[label]=[dict(value=str(v),algebraic=int(mult),geometric=len(vec)) for v,mult,vec in Q.eigenvects()]
assert any(x['algebraic']!=x['geometric'] for x in records['Minkowski075'])
assert all(x['algebraic']==x['geometric'] for name in ['Minkowski1','Minkowski2','Minkowski4'] for x in records[name])
# At a lapse/shift coincidence, remove row p/Omega and column chi_r.
# This nonzero 9x9 minor proves rank nine for positive interior lapse data.
collision_values={m:3*g*z*z/4,F:g*z*z/c-E*g*k*w*w/t}
Q=P.subs(collision_values)-z*s.eye(10)
minor=s.factor(Q.minor_submatrix(3,0).det())
expected=2*a*c*c*F*z*z*(g*z*z-a*a*c)**2/(3*g**3)
assert s.simplify(minor-expected.subs(collision_values))==0
assert s.simplify(poly.subs(collision_values))==0
record=dict(characteristic_polynomial=str(poly),polynomial_identity_verified=True,corners=records,
 collision_minor=str(minor),collision_minor_positive_form=str(expected),
 collision_scope='On lapse/shift coincidence, positive alpha,chi,g,F and lapse speed strictly above light speed imply a nonzero 9x9 minor; each repeated nonzero speed has geometric multiplicity one. This is an interior scalar-symbol statement.',
 scope='Exact selected scalar corners; singular physical reduction and uniform collapsed-puncture well-posedness remain unresolved.')
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
