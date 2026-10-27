#!/usr/bin/env python3
"""Analytic initial-time ADM/constraint identities for a Minkowski lapse pulse."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from numerics import RadialOperators
from kernel import Kernel,PARITY

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[];kernel=Kernel(a.library)
for n in [64,128,256,512,1024]:
 r=(np.arange(n)+.5)/(n-.5);ops=RadialOperators(r);O=(1-r*r)/2
 amp=.001;c=.25;s=.1
 f=amp*np.exp(-(r*r-c*c)**2/(4*s**4))
 L1=-r*(r*r-c*c)/s**4;L2=-(3*r*r-c*c)/s**4
 fp=f*L1;fpp=f*(L2+L1*L1)
 q=np.zeros((n,8));q[:,6]=f
 d,dd=ops.derivatives(q,PARITY)
 R,S,con=kernel.parts(r,q,d,dd,-r,-np.ones(n),np.zeros(n))
 rate=R.copy();rate[:-1]+=S[:-1]/O[:-1,None]
 dS,_=ops.derivatives(S,PARITY);rate[-1]=R[-1]-dS[-1]
 expected_A=-2/3*(fpp-fp/r)
 expected_p=-O*(fpp+2*fp/r)-3*r*fp+3*f
 expected_chi=np.zeros(n);expected_chi[:-1]=-2*f[:-1]/O[:-1]
 drate,ddrate=ops.derivatives(rate,PARITY)
 Mt=O*(drate[:,2]+3*rate[:,2]/r)-2*drate[:,3]/3+2*r*rate[:,2]
 Ht=2*O*O*(ddrate[:,0]+2*drate[:,0]/r)-4*rate[:,3]+4*O*(-3*rate[:,0]+.5*r*drate[:,0])-6*r*r*rate[:,0]
 bulk=r<.8
 row={'n':n,'h':1/(n-.5),'max_A_rate_error':float(max(abs(rate[:,2]-expected_A))),
 'max_p_rate_error':float(max(abs(rate[:,3]-expected_p))),
 'max_chi_rate_error':float(max(abs(rate[:,0]-expected_chi))),
 'bulk_M_t_max':float(max(abs(Mt[bulk]))),'bulk_H_t_max':float(max(abs(Ht[bulk]))),
 'Theta_rate_max':float(max(abs(rate[:,4]))),'Lambda_rate_max':float(max(abs(rate[:,5])))}
 rows.append(row)
result={'identity':'For exact constraint-free Minkowski data with arbitrary lapse, initial H_t=M_t=Theta_t=Lambda_t=0; A_t,p_t,chi_t have independent ADM formulas.',
 'pulse':'Published Gaussian-like exp(-(r^2-c^2)^2/(4 sigma^4)); amplitude=.001,c=.25,sigma=.1',
 'scope':'Initial-time consistency check; not nonlinear evolution or endpoint admission',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
