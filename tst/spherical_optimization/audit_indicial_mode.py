#!/usr/bin/env python3
"""Direct shared-RHS residual and evolved-tail polarization checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from audit_scri_indicial import WEIGHTS,ACTIVE,INDEPENDENT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--balance',type=Path,required=True);ap.add_argument('--pencils',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);balance=json.loads((a.balance/'audit.json').read_text());s=balance['default_positive_power'];amp=np.zeros(10);amp[ACTIVE]=balance['default_normalized_mode'];k=Kernel(a.library,separated=True,compact_end=.97);direct=[]
 for x in (.01,.003,.001,.0003,.0001):
  r=1-x;Q=np.zeros((1,10));Q[0,[0,1]]=1;Q[0,3]=-3;Q[0,6]=(1+r*r)/2;Q[0,7]=-r;D=np.zeros_like(Q);D[0,6]=r;D[0,7]=-1;DD=np.zeros_like(Q);DD[0,6]=1;_,source,omega=k.initial(np.array([r]));exponent=s+WEIGHTS;delta=[amp*x**exponent,-exponent*amp*x**(exponent-1),exponent*(exponent-1)*amp*x**(exponent-2)]
  residuals=[]
  for eps in (.01,.003):
   rates=[]
   for sign in (-1,1):
    q,d,dd=k.expand_chart(*[v+sign*eps*p for v,p in zip([Q,D,DD],delta)]);R,S,_,_=k.parts(np.array([r]),q,d,dd,source);rates.append((R+S/omega[:,:1])[0,INDEPENDENT])
   value=(rates[1]-rates[0])/(2*eps)*x**(1-s-WEIGHTS);residuals.append(value)
  direct.append(dict(distance=x,scaled_RHS_norm=float(np.linalg.norm(residuals[-1])),scaled_RHS=residuals[-1].tolist(),increment_difference_norm=float(np.linalg.norm(residuals[0]-residuals[1]))))
 k.close();saved=np.load(a.pencils/'pencils.npz');distances=np.array([.001,.0003,.0001]);design=np.column_stack([np.ones(3),distances,distances*distances]);coeff=np.linalg.solve(design,np.array([saved[f'x{x}-pencil'] for x in distances]).reshape(3,-1))[0].reshape(3,9,9);M=coeff[0]+s*coeff[1]+s*s*coeff[2];_,sv,V=np.linalg.svd(M);predicted=V[-1]/V[-1,6]
 final=np.load(a.run/'last-accepted.npz');initial=np.load(a.run/'initial.npz');x=1-final['r'];residual=final['q']-initial['q'];residual-=residual[-1];fits=[]
 for outer in (.015,.03):
  for exclude in (0,2):
   mask=(x>exclude*(final['r'][-1]-final['r'][-2]))&(x<=outer);z=x[mask]/outer;amplitudes=[];errors=[]
   for j in ACTIVE:
    y=residual[mask,INDEPENDENT[j]]
    if WEIGHTS[j]:y=y/x[mask];basis=np.column_stack([np.ones_like(z),z**s,z,z*z]);column=1
    else:basis=np.column_stack([z**s,z,z**(s+1),z*z]);column=0
    c=np.linalg.lstsq(basis,y,rcond=None)[0];amplitudes.append(float(c[column]/outer**s));errors.append(float(np.sqrt(np.mean((basis@c-y)**2))))
   normalized=np.array(amplitudes)/amplitudes[6];fits.append(dict(outer=outer,excluded_cells=exclude,amplitudes=amplitudes,normalized=normalized.tolist(),relative_polarization_difference=float(np.linalg.norm(normalized-predicted)/np.linalg.norm(predicted)),field_fit_RMS=errors))
 out=dict(power=s,direct_rhs=direct,predicted_outer_metric_mode=predicted.tolist(),leading_matrix_smallest_singular=float(sv[-1]),evolved_polarization=fits,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.balance/'audit.json',a.pencils/'pencils.npz',a.run/'last-accepted.npz',a.run/'initial.npz']},binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Direct continuum directional residual and finite-window evolved-mode fits. Neither uses an evolution boundary reset. Fits depend on finite range, analytic remainder ansatz and endpoint error; matching is evidence of excitation, not a nonlinear theorem.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(direct=direct,predicted=predicted.tolist(),fits=fits)))
if __name__=='__main__':main()
