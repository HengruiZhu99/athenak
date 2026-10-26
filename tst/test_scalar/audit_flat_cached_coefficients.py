#!/usr/bin/env python3
"""Check cached Minkowski coefficients against independent closed formulas."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--operators',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing cache audit is immutable')
x=np.memmap(a.operators/'kernel.f64',dtype='f8',mode='r').reshape(-1,56);rho=np.linalg.norm(x[:,33:36],axis=1);mask=(rho<=8)|(rho>=15.9);indices=np.flatnonzero(mask);maxima=np.zeros(33);worst=None
for begin in range(0,len(indices),65536):
 ii=indices[begin:begin+65536];xyz=np.array(x[ii,33:36]);r=rho[ii];n=xyz/r[:,None];o=np.where(r<=8,1,1-r/24);op=np.where(r<=8,0,-1/24);root=np.sqrt(r*r+100*o*o);rp=(r+100*o*op)/root;reference=np.empty((len(ii),33));g=reference[:,:16].reshape(-1,4,4)
 g[:,0,0]=-100/(root*root);g[:,0,1:]=g[:,1:,0]=-xyz/root[:,None];g[:,1:,1:]=np.eye(3)[None,:,:]+(o*o-1)[:,None,None]*n[:,:,None]*n[:,None,:]
 reference[:,16]=-3/root+r*rp/(root*root);reference[:,17:20]=(2*o*op+2*(o*o-1)/r)[:,None]*n;reference[:,20:29]=(-root[:,None,None]/100*np.eye(3)[None,:,:]-rp[:,None,None]/100*n[:,:,None]*xyz[:,None,:]).reshape(-1,9);reference[:,29:32]=rp[:,None]*n/10;reference[:,32]=-2*o*op/r
 delta=abs(x[ii,:33]-reference)/(1+abs(reference));local=delta.max(axis=0);maxima=np.maximum(maxima,local);at=np.unravel_index(delta.argmax(),delta.shape)
 if worst is None or float(delta[at])>worst['scaled_error']:worst=dict(row=int(ii[at[0]]),column=int(at[1]),rho=float(r[at[0]]),scaled_error=float(delta[at]),cached=float(x[ii[at[0]],at[1]]),analytic=float(reference[at]))
result=dict(passed=bool(maxima.max()<1e-10),points=len(indices),coefficients=33,max_scaled_error=float(maxima.max()),column_maxima=maxima.tolist(),worst=worst,tolerance=1e-10,scope='Independent closed Minkowski hyperboloid formulas for the full inverse metric, divergence, lapse/shift gradients and actual conformal potential. Check every cached point with rho<=8 or rho>=15.9. The C-infinity correction to its outer linear factor at15.9 is below1e-28 in these coefficients; the remaining transition is covered by independent pullback audits.',operator_metadata=json.loads((a.operators/'operator.json').read_text()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','points','max_scaled_error','worst']},indent=2))
if not result['passed']:raise SystemExit(1)
