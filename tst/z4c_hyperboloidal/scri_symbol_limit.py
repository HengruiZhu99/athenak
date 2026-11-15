#!/usr/bin/env python3
"""Exact radial scalar limit, checked against the full bulk-kernel symbol."""
import sys,json,subprocess,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'build-scri-experiment/python-libs'))
import sympy as sp
root=Path(__file__).resolve().parents[2];out=root/'docs/z4c_hyperboloidal/evidence/scri-investigation/symbol'
e,x,y=sp.symbols('epsilon lambda y');R=sp.Rational
M=sp.Matrix([
 [-1,0,0,R(2,3),R(4,3),0,0,0,-R(2,3),0],
 [0,-1,-2,0,0,0,0,0,R(4,3),0],
 [R(1,3),-R(1,2),-1,0,0,R(2,3),-R(2,3),0,0,0],
 [0,0,0,-1,0,0,-1,0,0,0],
 [1,0,0,0,-1,R(1,2),0,0,0,0],
 [0,0,0,-R(4,3),-R(2,3),-1,0,0,R(4,3),0],
 [R(7,4)*e,-e/4,0,-1,0,0,-1,0,0,-e/2],
 [0,0,0,0,0,0,0,-1,0,0],
 [0,0,0,0,0,0,0,0,-1,R(3,2)],
 [0,0,0,-R(4,3),-R(2,3),0,0,0,R(4,3),-1]])
poly=sp.factor(M.charpoly(x).as_expr());quartic=y**4-3*y**2+R(13,6)*e*y+2
assert sp.expand(poly-x**2*(x+1)**2*(x+2)**2*quartic.subs(y,x+1))==0
u=(3+sp.sqrt(33))/6;critical=12*sp.sqrt(u)*(3-2*u)/13
assert sp.simplify(quartic.subs({y:sp.sqrt(u),e:critical}))==0
assert sp.simplify(sp.diff(quartic,y).subs({y:sp.sqrt(u),e:critical}))==0
exe=root/'build-scri-experiment/local/tst/z4c_hyperboloidal/hyperboloidal_scri_symbol'
checks=[]
for eps in [0,.5,1]:
 for r in [.99,.999,.9999,.99999]:
  obj=json.loads(subprocess.check_output([str(exe),str(r),str(eps),'0','0','flat','.001','.98'],text=True))
  a=np.array(obj['matrix']);fields=obj['fields'];idx=[fields.index(f) for f in [0,1,7,13,14,15,18,19,22,25]]
  T=np.zeros((26,10))
  for j,i in enumerate(idx):T[i,j]=1
  T[fields.index(4),1]=-.5;T[fields.index(10),2]=-.5
  b=np.einsum('ij,jk->ik',a,T)[idx]
  leakage=float(np.max(abs(np.einsum('ij,jk->ik',a,T)-np.einsum('ij,jk->ik',T,b))))
  # The probe uses collar end .99, so W=1 for all these samples.
  error=float(np.max(abs(b-np.array(M.subs(e,eps),dtype=float))))
  assert leakage<1e-8 and error<5*(1-r)
  checks.append(dict(epsilon=eps,radius=r,scalar_subspace_leakage=leakage,limit_error=error))
summary={'scalar_basis':['d_n chi','d_n g_xx','A_xx','delta p/Omega','delta Theta/Omega','Lambda_x','d_n alpha','B_x','d_n beta_x','D_x'],'axisymmetry':'g_yy=g_zz=-g_xx/2 and A_yy=A_zz=-A_xx/2 at the linearized flat reference','matrix':str(M),'characteristic_polynomial':str(poly),'shifted_quartic':str(quartic),'epsilon_critical':float(critical),'double_root_lambda':float(sp.sqrt(u)-1),'epsilon_one_roots':[str(z) for z in sp.nroots(quartic.subs(e,1))],'checks':checks,'convention':'q_t=P d_n q; physical speeds are minus eigenvalues'}
(out/'radial-limit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
