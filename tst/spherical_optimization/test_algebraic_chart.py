#!/usr/bin/env python3
"""Qualification of the exact algebraic chart, separate from physical constraints."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from audit import eigenspace_audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);k=Kernel(a.library,separated=True)
 independent=[0,1,3,5,6,7,8,9,10,11];rows=[]
 for r in (.02,.2,.7,.85,.95,.99):
  q,*_=k.initial(np.array([r]));q=q[0];P=k.free_radial_symbol(r,q)
  J=np.eye(12)[:,independent];J[2,1]=-.5*q[2]/q[1];J[4,2]=-.5*q[2]/q[1]
  assert np.isfinite(P).all() and np.isfinite(J).all()
  product=np.einsum('ij,jk->ik',P,J,optimize=False);reduced=product[independent]
  error=float(np.max(abs(product-np.einsum('ij,jk->ik',J,reduced,optimize=False))))
  assert error<1e-9,error
  audit=eigenspace_audit(reduced);assert not audit['defects'],audit
  rows.append(dict(r=r,free=eigenspace_audit(P),chart=audit,invariant_subspace_error=error,
                   free_eigenvalues=np.linalg.eigvals(P).real.tolist(),chart_eigenvalues=np.linalg.eigvals(reduced).real.tolist()))
 # Finite radial differentiation independently checks both product-rule jets.
 q=np.array([[.8,1.1,.03,-.5,.01,.02,.6,-.1,.04,.08]])
 d=np.array([[.03,.04,.02,.01,.02,.01,.05,.02,.01,.04]])
 dd=np.array([[.01,.02,.03,.02,.01,.02,.01,.04,.03,.02]])
 Q,D,DD=k.expand_chart(q,d,dd);eps=1e-3
 plus=k.expand_chart(q+eps*d+eps**2*dd/2)[0];minus=k.expand_chart(q-eps*d+eps**2*dd/2)[0]
 first=float(np.max(abs((plus-minus)/(2*eps)-D)));second=float(np.max(abs((plus-2*Q+minus)/eps**2-DD)))
 assert first<1e-8 and second<1e-8
 result=dict(passed=True,first_jet_error=first,second_jet_error=second,points=rows,
  qualification='Interior point audit and exact algebraic identities; not evolution or endpoint well-posedness.')
 (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));k.close()
if __name__=='__main__':main()
