#!/usr/bin/env python3
"""Rational limiting pencil, shared-gauge parameter algebra and constraints.

Rational coefficients are reconstructed from compiled-kernel distance limits;
this is a checked numerical identification of the leading algebra, not an
independent proof of every omitted continuum remainder.
"""
import argparse,json,hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.linalg import svd
from physical_constraints import evaluate
from audit_scri_indicial import WEIGHTS,ACTIVE,INDEPENDENT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pencils',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);saved=np.load(a.pencils/'pencils.npz');x=np.array([.001,.0003,.0001]);design=np.column_stack([np.ones(3),x,x*x]);limit=np.linalg.solve(design,np.array([saved[f'x{z}-pencil'] for z in x]).reshape(3,-1))[0].reshape(3,9,9)
 matrices=[sp.Matrix([[sp.Rational(Fraction(float(v)).limit_denominator(300)) for v in row] for row in mat]) for mat in limit];difference=float(max(np.max(abs(np.array(r,dtype=float)-v)) for r,v in zip(matrices,limit)));assert difference<2e-6
 s,R,xi=sp.symbols('s R xi');matrices[0][7,7]=1-R;matrices[0][6,6]=-2*xi;pencil=matrices[0]+s*matrices[1]+s*s*matrices[2];determinant=sp.factor(pencil.det());expected=sp.Rational(8,15)*s*(s-2)*(s+1)*(31*s-269)*(xi*(s*s+(R+3)*s+2)-2*R);assert sp.simplify(determinant-expected)==0
 root=float((-7+sp.sqrt(73))/2);P=np.array(pencil.subs({s:root,R:4,xi:1}),dtype=float);_,sing,V=svd(P);v=V[-1];v/=v[6];amplitude=np.zeros(10);amplitude[ACTIVE]=v;constraints=[]
 for distance in (.01,.003,.001,.0003,.0001):
  r=1-distance;Q=np.zeros(12);Q[[0,1,2]]=1;Q[5]=-3;Q[8]=(1+r*r)/2;Q[9]=-r;D=np.zeros(12);D[8]=r;D[9]=-1;DD=np.zeros(12);DD[8]=1;exponent=root+WEIGHTS
  perturb=[amplitude*distance**exponent,-exponent*amplitude*distance**(exponent-1),exponent*(exponent-1)*amplitude*distance**(exponent-2)]
  for base,p in zip((Q,D,DD),perturb):
   full=np.zeros(12);full[INDEPENDENT]=p;full[2]=-.5*p[1];full[4]=-.5*p[2]
   # Linearized exact chart at flat g=1,A=0; epsilon² terms do not
   # contribute to the independent physical constraint directional derivative.
   base.shape=(1,12)
   if base is Q:q=base+1j*1e-25*full
   elif base is D:d=base+1j*1e-25*full
   else:dd=base+1j*1e-25*full
  omega=np.array([[(1-r*r)/2,-r,-1]])
  value=evaluate(np.array([r,r]),np.vstack([q,q]),np.vstack([d,d]),np.vstack([dd,dd]),np.vstack([omega,omega]))[0].imag/1e-25/distance**(root+1)
  constraints.append(dict(distance=distance,leading_scaled_H_M_Theta_Z=value.tolist()))
 candidates=[]
 for damping in (1.,.8,.4,.3,.25,.2,.1):
  positive=(-7+np.sqrt(49-4*(2-8/damping)))/2;candidates.append(dict(outer_lapse_damping=damping,positive_weighted_power=float(positive),metric_lapse_power=float(positive+1)))
 coefficients=[[[str(v) for v in row] for row in np.array(mat.subs({R:4,xi:1})).tolist()] for mat in matrices]
 out=dict(rational_limit_max_error=difference,default_matrices=coefficients,determinant=str(determinant),parameter_identification='Shared outer shift source contributes (1-R)*beta; shared lapse source contributes -2*xi*alpha in M0. xi is the scri value of c.xi; all other rows retain the default shared geometric coefficients.',default_positive_power=root,default_metric_lapse_power=root+1,default_normalized_mode=v.tolist(),chart_field_order=['chi','gr','Ar','Khat','Theta','Lambda','alpha','beta','D'],mode_residual=float(np.linalg.norm(P@v)),constraint_checks=constraints,candidates=candidates,sympy_version=sp.__version__,inputs={str(a.pencils/'pencils.npz'):hashlib.sha256((a.pencils/'pencils.npz').read_bytes()).hexdigest()},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Weighted leading spatial balance on the smooth native/flat outer limit. Time derivatives are one power higher in this ansatz. Finite-range observed tails and this linear mode do not prove nonlinear excitation or stability. Parameter predictions require consistent shared sources, pole-normal changes and fresh equivalence/evolution checks.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(determinant=str(determinant),positive=root,mode=v.tolist(),constraints=constraints,candidates=candidates)))
if __name__=='__main__':main()
