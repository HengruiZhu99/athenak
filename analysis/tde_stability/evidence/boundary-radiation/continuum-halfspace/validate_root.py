from pathlib import Path
import json
import numpy as np
from scipy import linalg,optimize
from schur_check import *

def chart(lam,ky,mode='radiation',damping=True):
 rec,data=assess_schur(lam,ky,mode,damping,True);U,D,T,B,R,ss,vv,Us,norms=data
 # q-boundary-value chart removes arbitrary Schur column phases/rotations.
 Bq=B@np.linalg.inv(U[QP]);Bq/=np.linalg.norm(Bq,axis=1)[:,None]
 return np.linalg.det(Bq),np.linalg.cond(U[QP]),rec,data

for x in [1.22,1.23,1.24]:print('chart',x,chart(x,2*np.pi)[:2],flush=True)
phase=np.exp(-1j*np.angle(chart(1.22,2*np.pi)[0]));out=[]
for damp in [False,True]:
 for ky in [np.pi,2*np.pi,4*np.pi]:
  func=lambda x:float((phase*chart(x,ky,damping=damp)[0]).real)
  grid=np.linspace(.17*ky,.23*ky,20);brackets=[(a,b) for a,b in zip(grid[:-1],grid[1:]) if func(a)*func(b)<0]
  if not brackets:raise RuntimeError(('No bracket',damp,ky))
  a,b=brackets[0];rt=optimize.brentq(func,a,b,xtol=3e-14,rtol=1e-14)
  det,cond,rec,data=chart(rt,ky,damping=damp);U,D,T,B,R,ss,vv,Us,norms=data
  coeff=np.linalg.solve(R,vv[-1].conj());state=U@coeff;scale=np.max(abs(state));coeff/=scale;state/=scale
  # Direct original20field PDE + boundary residual on a smooth normal profile.
  V0,_=volume(0,ky,damping=damp);Vp,_=volume(1,ky,damping=damp);Vm,_=volume(-1,ky,damping=damp);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
  residuals=[];profiles=[];constraints=[]
  for x in np.linspace(-2,0,81):
   cc=linalg.expm(T*x)@coeff;u=U@cc;du=D@cc;ddu=D@T@cc
   rhs=V0@u+V1@du+V2@ddu
   residuals.append(np.linalg.norm(rt*u-rhs)/max(np.linalg.norm(rhs),1e-300))
   h=[[u[1],u[3],u[4]],[u[3],u[2],u[5]],[u[4],u[5],-u[1]-u[2]]]
   dh=[[du[1],du[3],du[4]],[du[3],du[2],du[5]],[du[4],du[5],-du[1]-du[2]]]
   q=np.array([u[13+i]-dh[i][0]-1j*ky*h[i][1] for i in range(3)])
   profiles.append(np.linalg.norm(u));constraints.append([abs(u[7]),np.linalg.norm(q)])
  # Compare boundary operator unnormalized to sum of row contributions.
  bdef=np.linalg.norm(B@coeff)/max(np.linalg.norm(B)*np.linalg.norm(coeff),1e-300)
  control=assess_schur(rt,ky,'zero_rate',damp)
  result=dict(damping=damp,k_y=ky,lambda_real=rt,lambda_imag_actual_tangent_beta=-.01503577649949646*ky,chart_determinant_abs=abs(det),chart_q_condition=cond,**rec,full20_bulk_max_relative_defect=max(residuals),boundary_relative_defect=bdef,Theta_peak=max(a[0] for a in constraints),Q_peak=max(a[1] for a in constraints),state_peak_normalization=1.,normal_profile_norm_ratio_at_minus2=profiles[0]/profiles[-1],zero_rate_sigma_at_same_lambda=control['sigma_min'])
  out.append(result);print(result,flush=True)
  if damp and ky==2*np.pi:np.savez_compressed(ROOT/'validated-halfspace-mode.npz',U=U,D=D,T=T,coeff=coeff,boundary=B,lambda_real=rt,ky=ky,x=np.linspace(-2,0,81),state_norm=profiles,constraint_norm=constraints)
(ROOT/'validated-root.json').write_text(json.dumps(out,indent=2)+'\n')
