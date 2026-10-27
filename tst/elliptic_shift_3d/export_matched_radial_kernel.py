#!/usr/bin/env python3
"""Independent radial kernel fixtures from the hash-verified successful 1D source."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
reference=a.reference.resolve()
receipt=json.loads((reference.parent/'receipt.json').read_text())
for name,digest in receipt['source_hashes'].items():
    if not name.endswith(('.dylib','.so')):
        assert hashlib.sha256((reference/name).read_bytes()).hexdigest()==digest, name
sys.path.insert(0,str(reference))
import g1_rhs
from gauge import restoring_coefficient,collapsed_background_correction
from telegraph import window
class Solve:
    def solve(self,integrand,outer):
        # Pointwise fixtures supply a shift value; its first/second jets below
        # are still constrained by the independent radial gauge equation.
        return np.linspace(-.12,.08,len(integrand))
r=np.linspace(.02,.98,49);ops=SimpleNamespace(r=r,shift_ode_solver=Solve())
rng=np.random.default_rng(101127);rows=[]
for scale in [1.,.01,1e-6,1e-10]:
    q=rng.normal(0,.03,(len(r),8));d=rng.normal(0,.04,q.shape);dd=rng.normal(0,.05,q.shape)
    q[:,1]=d[:,1]=dd[:,1]=0
    chi=.9+.1*r;alpha=scale*(1+r*r)/2
    q[:,0]=chi-1;q[:,6]=alpha-(1+r*r)/2
    q[:,2]*=r*r;q[:,3]+=2.7
    R,S,_,beta,db,ddb=g1_rhs.parts(ops,q,d,dd,restoring=0,total_alpha=alpha,total_chi=chi)
    O=(1-r*r)/2;A=(1+r*r)/2;u=q[:,6]
    S[:,6]-=2*restoring_coefficient(r,1,profile='physical-log')*alpha*u
    S[:,6]+=collapsed_background_correction(r,u)
    expected=R+S/O[:,None]
    w,wp=window(r,width=.24,inner_profile='af-regular')
    B=.013*r+.007*r*r;Bp=.013+.014*r
    eta=(1+1e-4)*alpha/(alpha+1e-4*A)
    expected[:,6]+=eta*chi*(w*Bp+(wp+2*w/r)*B)
    flux=beta*Bp+(.005*w*d[:,6]-B)/.02
    for i in range(len(r)):
        rows.append(np.r_[r[i],chi[i],alpha[i],q[i],d[i],dd[i],beta[i],db[i],ddb[i],B[i],Bp[i],expected[i,:7],flux[i]])
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:
    f.write(str(len(rows))+'\n');np.savetxt(f,np.asarray(rows),fmt='%.17g')
print(json.dumps({'cases':len(rows),'alpha_scale':[1,.01,1e-6,1e-10],'nonzero_Theta_Lambda':True,'nonzero_telegraph_flux':True,'reference_source':str(a.reference)}))
