#!/usr/bin/env python3
"""Independent finite-radius matrix pullbacks, including paper worldtube radii."""
import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nonradiative_analytic as f
M=1/3;report={}
for height in (1,16):
 for family,param in [('schwarzschild',0),('rotation',.8),('bounce',2),('gauge',10)]:
  errors=[]
  radii=[.8,.9,.95]+[(np.sqrt(1+(M*R)**2)-1)/(M*R) for R in (15,20,25,30)]
  for r in radii:
   n=np.array([.3,.4,np.sqrt(.75)]);x=r*n;om=(1-r*r)/2;R=r/om;u=8.3+height*(1-r)/(1+r)
   dR=np.r_[0,(1+r*r)/(2*om*om)*n];du=np.r_[1,-2*height*n/(1+r)**2]
   dy=np.zeros((3,4));dy[:,1:]=np.eye(3)/om+np.outer(x,x)/(om*om)
   if family=='rotation':T=u;dT=du
   else:T=u+R+4*M*np.log(R/(2*M)-1);dT=du+(1+4*M/(R-2*M))*dR
   y=R*n
   if family=='bounce':
    angle=2*np.pi*T/(40*M);a=2*M*np.sin(angle)**4;ap=16*np.pi/40*np.sin(angle)**3*np.cos(angle)
    y[0]-=a;dy[0]-=ap*dT
   jac=np.vstack((dT,dy));physical=f.physical(T,y[None,:],family,parameter=param)[0][0]
   numerical=om**2*jac.T@physical@jac
   expected=f.compact(8.3,x[None,:],'schwarzschild' if family=='bounce' else family,parameter=param,height=height)[0][0]
   errors.append(dict(r=r,R_over_M=R/M,max_error=float(np.max(abs(numerical-expected)))))
  report[f'{family}-H{height}']=errors
print(json.dumps(report,indent=2))
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
assert max(row['max_error'] for rows in report.values() for row in rows)<2e-11
