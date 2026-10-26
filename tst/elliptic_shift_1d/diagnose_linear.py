#!/usr/bin/env python3
"""Discrete Jacobian diagnostic, not a continuum or nonlinear stability theorem."""
import argparse,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evolve import Evolution,sha

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--n',type=int,default=24)
p.add_argument('--regularity-sat',type=float,default=0.);p.add_argument('--scri-sat',type=float,default=0.);p.add_argument('--endpoint',default='raw-lhopital');p.add_argument('--epsilon',type=float,default=1e-6)
p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
opts=SimpleNamespace(n=args.n,width=7,library=args.library,eta=0.,kappa1=4.,kappa2=-.99,theta='appendix-c',slicing=1.,endpoint=args.endpoint,scri_sat=args.scri_sat,regularity_sat=args.regularity_sat)
ev=Evolution(opts);shape=(args.n,8);indices=[i*8+f for i in range(args.n) for f in range(7) if not (i==args.n-1 and f==6)]
J=np.empty((len(indices),len(indices)));start=time.perf_counter()
for col,k in enumerate(indices):
    q=np.zeros(shape);q.flat[k]=args.epsilon;plus=ev.rhs(q).ravel()[indices]
    q.flat[k]=-args.epsilon;minus=ev.rhs(q).ravel()[indices]
    J[:,col]=(plus-minus)/(2*args.epsilon)
values,vectors=np.linalg.eig(J);k=np.argmax(values.real);v=np.zeros(shape,dtype=complex);v.flat[indices]=vectors[:,k]
end_fraction=float(np.sum(abs(v[ev.r>.9])**2)/np.sum(abs(v)**2))
fields=['chi','g','Arr','Khat','Theta','Lambda','alpha','beta'];shares=np.sum(abs(v)**2,axis=0)/np.sum(abs(v)**2)
# Minimal model: T_t=-c T/Omega, with l'Hopital endpoint T_t=c T'.
# Its endpoint-only eigenvalue is +c D_1(S,S); c=4.04 for the damping term.
D,_=ev.op.derivatives(np.eye(args.n),1);minimal_rate=4.04*D[-1,-1]
record={'n':args.n,'h':ev.h,'endpoint':args.endpoint,'scri_sat':args.scri_sat,'regularity_sat':args.regularity_sat,'epsilon':args.epsilon,
 'dominant_real':float(values[k].real),'dominant_imag':float(values[k].imag),
 'h_times_dominant_real':float(ev.h*values[k].real),'endpoint_eigenvector_energy_fraction':end_fraction,
 'field_energy_fraction':dict(zip(fields,shares.tolist())),
 'minimal_scalar_raw_endpoint_rate':float(minimal_rate),'runtime_seconds':time.perf_counter()-start,
 'source_sha256':sha(__file__),'library_sha256':sha(args.library),
 'interpretation':'Finite-dimensional raw nodal discretization; perturbations need not belong to the compatible continuum weighted phase. Does not refute the supplied theorem.'}
np.savez(args.output/'jacobian.npz',J=J,r=ev.r,eigenvalues=values,dominant_mode=v)
(args.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
