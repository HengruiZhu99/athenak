#!/usr/bin/env python3
"""Off-constraint angular Ricci and analytic radial scalar checks."""
import argparse,json
from pathlib import Path
import numpy as np
from numerics import RadialOperators
from kernel import Kernel,PARITY
import g1_rhs,scalar_rhs
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path);p.add_argument('--damping-form',choices=['constant','reference-lapse'],default='constant');a=p.parse_args()
r=(np.arange(96)+.5)/95.5;op=RadialOperators(r);kernel=Kernel(a.library);errors=[]
for seed in range(4):
 rng=np.random.default_rng(seed);q=np.zeros((len(r),8))
 for f in [0,2,3,4,5,6]:
  q[:,f]=sum(rng.normal()*.003*r**(2*k) for k in range(4))
  if f==2:q[:,f]*=r*r
  if f==5:q[:,f]*=r
 q[-1,6]=0;d,dd=op.derivatives(q,PARITY)
 for appendix in [True,False]:
  R,S,diag,b,db,ddb=g1_rhs.parts(op,q,d,dd,appendix_c=appendix,restoring=1)
  R0,S0,_=kernel.parts(r,q,d,dd,b,db,ddb,appendix_c=appendix,lapse_restoring=1)
  if a.damping_form=='reference-lapse':
   effective=4*((1+r*r)/2)/((1+r*r)/2+q[:,6])
   rows=[kernel.parts(r[i:i+1],q[i:i+1],d[i:i+1],dd[i:i+1],b[i:i+1],db[i:i+1],ddb[i:i+1],kappa1=effective[i],appendix_c=appendix,lapse_restoring=1) for i in range(len(r))]
   R0=np.concatenate([z[0] for z in rows]);S0=np.concatenate([z[1] for z in rows])
  O=(1-r*r)/2;A0=(1+r*r)/2;chi=1+q[:,0];cp=d[:,0];alpha=A0+q[:,6];alphap=r+d[:,6];K=q[:,3]-3+2*q[:,4];Ar=q[:,2];w=b*r/alpha
  rate=R0[:-1]+S0[:-1]/O[:-1,None]
  D0K=rate[:,3]+2*rate[:,4]-b[:-1]*(d[:-1,3]+2*d[:-1,4])
  D0Ar=rate[:,2]-b[:-1]*d[:-1,2]
  D0k=D0K/3-O[:-1]*D0Ar/2-alpha[:-1]*w[:-1]*Ar[:-1]/2
  k=K/3-O*Ar/2
  spatial=np.sqrt(chi)*(A0/r-.5*O*cp/chi)
  R3=O*O*(.5*dd[:,0]-.75*cp*cp/chi+1.5*cp/r)+O*(-4*chi+r*cp)-2*chi*r*r
  Hess=chi*(O*alphap+alpha*r)/alpha*(A0/r-.5*O*cp/chi)
  direct=R3[:-1]+K[:-1]*k[:-1]-O[:-1]/alpha[:-1]*D0k-Hess[:-1]
  predicted=scalar_rhs.angular_ricci(r,q,d,b,appendix_c=appendix,damping_form=a.damping_form)[:-1]
  err=float(max(abs(direct-predicted)));errors.append(err);assert err<2e-12,err
# Analytic fixed-Minkowski solution, evaluated independently of the transport oracle.
spatial=[]
for n in [128,256,512]:
 r=(np.arange(n)+.5)/(n-.5);op=RadialOperators(r);q=np.zeros((n,8));t=.6
 u=t+(1-r)/(1+r);v=np.zeros_like(r);v[:-1]=t+(1+r[:-1])/(1-r[:-1])
 def F(z):
  s=(z-2)/.2;f=np.exp(-s*s);return f,-10*s*f,(100*s*s-50)*f
 fu,fup,fupp=F(u);fv=np.zeros(n);fvp=np.zeros(n);fvpp=np.zeros(n);fv[:-1],fvp[:-1],fvpp[:-1]=F(v[:-1])
 ur=-2/(1+r)**2;vr=np.zeros(n);vr[:-1]=2/(1-r[:-1])**2
 psi=fu-fv;psit=fup-fvp;psitt=fupp-fvpp;psir=fup*ur-fvp*vr;psitr=fupp*ur-fvpp*vr
 A0=(1+r*r)/2
 fields=np.stack([psi/r,(psit+r*psir)/(A0*r)],axis=1)
 expected=np.stack([psit/r,(psitt+r*psitr)/(A0*r)],axis=1)
 df,ddf=op.derivatives(fields,np.ones(2));d=np.zeros_like(q)
 got,info=scalar_rhs.rhs(op,q,d,fields,df,ddf,-r,np.zeros(n),damping_form=a.damping_form)
 error=float(np.sqrt(np.mean((got-expected)**2)));spatial.append(error)
 assert np.max(abs(info['potential']))==0
orders=np.log2(np.array(spatial[:-1])/np.array(spatial[1:]));assert min(orders)>4.,orders
record={'damping_form':a.damping_form,'angular_Ricci_native_errors':errors,'analytic_scalar_rhs_errors':spatial,'orders':orders.tolist(),
 'scope':'Component geometry/analytic-wave audit; coupled evolution has separate evidence and acceptance gates'}
if a.output:a.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
