#!/usr/bin/env python3
"""Exact full 50-field Fourier polynomial via an invariant reduction manifold."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import sympy as s
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();sys.path.insert(0,str(a.reference_dir))
from exact_flat_symbol import matrices, rotations
from check_flat_fourier import source_matrix
k,rate,eta,kap=s.symbols('k lambda eta kappa',real=True);z=s.Symbol('s')
J=source_matrix(rate,eta,kap);Ps=matrices(rate,eta)
for R,T in rotations():
 assert s.simplify(T*J-J*T)==s.zeros(50)
 for i in range(3):assert s.simplify(T*Ps[i]-Ps[i]*T-sum((R[j,i]*Ps[j] for j in range(3)),s.zeros(50)))==s.zeros(50)
eK=s.eye(50)[:,10];chain=J*eK
assert chain[0]==s.Rational(1,3) and chain[1]==-s.Rational(7,3)
assert J*chain==s.zeros(50,1)
A=J+s.I*k*Ps[0]
C=s.zeros(30,50);C[:,20:]=s.eye(30);C[0,0]=-s.I*k;C[3,0]=C[3,1]=-s.I*k
for v in range(5):C[6+v,2+v]=-s.I*k
for v in range(3):C[21+v,7+v]=-s.I*k
assert s.simplify(C*A+rate*C)==s.zeros(30,50)
embed=s.zeros(50,20);embed[:20,:]=s.eye(20);embed[20:,:]=-C[:,:20]
assert C*embed==s.zeros(30,20)
F=s.simplify((A*embed)[:20,:]);assert s.simplify(A*embed-embed*F)==s.zeros(50,20)
actual=s.factor(F.charpoly(z).as_expr())
expected=(z*z+k*k)**2*(z*z+2*k*k)*(z*z+eta*z+k*k)**3*(z*z+kap*z+k*k)**3*(z*z+2*kap*z+k*k)
assert s.expand(actual-expected)==0
# [primary variables, reductions] is an invertible triangular state map.
transform=s.eye(50);transform[20:,:]=C
assert transform.det()==1
finite_jordan=[]
for pars in [(1,2,1),(0,2,0),(1,0,0)]:
 rr,ee,kk=map(s.Integer,pars)
 AA=source_matrix(rr,ee,kk)+s.I*matrices(rr,ee)[0]
 for ev in [s.I,s.sqrt(2)*s.I]:
  M=AA-ev*s.eye(50)
  finite_jordan.append(dict(parameters=pars,k=1,eigenvalue=str(ev),nullities=[50-M.rank(),50-(M*M).rank()]))
out=dict(status='PASS',finite_frequency_neutral_nullities=finite_jordan,exact_reduction_closure=True,exact_invariant_embedding=True,state_transform_determinant=1,exact_rotational_covariance=True,homogeneous_K_Jordan_chain=['e_K','(e_w-7*e_rho)/3','0'],constrained_polynomial=str(actual),full_polynomial=str((z+rate)**30*expected),spectral_conclusion='For nonnegative lambda, eta and kappa and real k, every root has nonpositive real part; zero modes and Jordan/transient growth are not excluded.',reference_sha256={name:hashlib.sha256((a.reference_dir/name).read_bytes()).hexdigest() for name in ['exact_flat_symbol.py','check_flat_fourier.py']},scope='Minkowski alpha=w=1, sigma=1, beta=0; arbitrary real wave number and direction by exact rotational covariance')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
