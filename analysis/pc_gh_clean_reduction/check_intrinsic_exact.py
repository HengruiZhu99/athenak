#!/usr/bin/env python3
"""Exact chart inverse and tangent-map identities, separate from compiled tests."""
import argparse
import json
from pathlib import Path
import sympy as s
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
a,c,b,d,e=s.symbols('a c b d e',real=True);q=(a,c,b,d,e)
t=s.Matrix([[s.exp(a),0,0],[b,s.exp(c),0],[d,e,s.exp(-a-c)]])
it=s.Matrix([[s.exp(-a),0,0],[-b*s.exp(-a-c),s.exp(-c),0],[(b*e*s.exp(-c)-d)*s.exp(c),-e*s.exp(a),s.exp(a+c)]])
zero=lambda x: all(s.simplify(v)==0 for v in x)
assert zero(it*t-s.eye(3)) and zero(t*it-s.eye(3))
g=t*t.T;inv=it.T*it
assert zero(g*inv-s.eye(3)) and s.simplify(g.det()-1)==0
v=s.symbols('v:5');x=s.zeros(3)
for qi,vi in zip(q,v):x+=g.diff(qi)*vi
w=s.simplify(it*x*it.T);ell=s.zeros(3)
for i in range(3):
 for j in range(i+1):ell[i,j]=w[i,j]/(2 if i==j else 1)
td=t*ell
recovered=s.Matrix([ell[0,0],ell[1,1],td[1,0],td[2,0],td[2,1]])
assert zero(recovered-s.Matrix(v))
ah=s.symbols('A:5');curv=s.Matrix([[ah[0],ah[1],ah[2]],[ah[1],ah[3],ah[4]],[ah[2],ah[4],-ah[0]-ah[3]]])
assert s.simplify(s.trace(inv*t*curv*t.T))==0
assert s.simplify(s.trace(inv*x))==0
out={'status':'PASS','exact':['two-sided triangular inverse','metric inverse','unit determinant','arbitrary curvature trace','arbitrary independent gradient trace','five-component tangent inverse'],'scope':'exact real chart identities; not a bounded-condition or puncture theorem'}
args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
