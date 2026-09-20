"""Full-state spherical generalized eigenproblem; boundary closure under audit.

Read accompanying diagnostics before interpreting spectra. Full-state and
induced physical-constraint residuals are both reported, including endpoints.
"""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
from scipy import linalg
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT.parent/'radial-volume'))
import radial_operator as volume
from boundary_operator import *

def build(degree,inner=.2,outer=4,rate=.1,falloff='areal'):
 r,D=chebyshev(degree,inner,outer);n=len(r);D2=D@D
 A2,A1,A0=volume.coefficients(r,rate)
 raw=np.block([[A2[:,i,j,None]*D2+A1[:,i,j,None]*D+np.diag(A0[:,i,j]) for j in range(8)] for i in range(8)])
 A=raw.copy();B=np.eye(8*n);rows,labels,C=physical_boundary_rows(r,D,rate,False,falloff)
 replacements=[2*n,5*n,3*n,4*n,6*n-1]  # outer k/Gamma/Theta/A_TF; inner Gamma
 for index,row in zip(replacements,rows):
  scale=max(np.linalg.norm(row),1e-300);A[index]=row/scale;B[index]=0
 return A,B,raw,r,D,C,rows,labels,replacements

def inspect(lam,vec,raw,r,D,C,rows,labels,replacements,falloff,profiles=False):
 n=len(r);u=vec.reshape(8,n);th=C['Theta']@vec;q=C['Q']@vec;hh=C['H']@vec;mm=C['M']@vec
 a=r/(1+r);b=r/(1+r)**2
 # Full PDE residual at the five replaced boundary rows is a separate check.
 rr=lam*vec-raw@vec;den=abs(lam)*np.linalg.norm(vec)+np.linalg.norm(raw@vec)
 boundary=np.array(replacements);interior=np.ones(8*n,bool);interior[boundary]=False
 normalizedBC=rows@vec/np.maximum(np.linalg.norm(rows,axis=1)*np.linalg.norm(vec),1e-300)
 omegaT=0 if falloff=='none' else (1/r[0] if falloff=='coordinate' else 1/(r[0]+1));omegaQ=0 if falloff=='none' else 1/r[0]
 dynbc=np.array([lam*th[0]+(a[0]**2-b[0])*((D@th)[0]+omegaT*th[0]),lam*q[0]+(a[0]**2-b[0])*((D@q)[0]+omegaQ*q[0])])
 dynamicscales=np.array([abs(lam*th[0])+abs((a[0]**2-b[0])*(D@th)[0])+abs((a[0]**2-b[0])*omegaT*th[0]),abs(lam*q[0])+abs((a[0]**2-b[0])*(D@q)[0])+abs((a[0]**2-b[0])*omegaQ*q[0])])
 cn=np.linalg.norm(np.array([th,q,hh,mm]));un=np.linalg.norm(u)
 result={'real_per_M':float(lam.real),'imag_per_M':float(lam.imag),'full_PDE_relative_residual_interior':float(np.linalg.norm(rr[interior])/max(den,1e-300)),'full_PDE_relative_residual_replaced_endpoint_rows':float(np.linalg.norm(rr[boundary])/max(den,1e-300)),'algebraic_boundary_relative_residuals':abs(normalizedBC).tolist(),'radiative_constraint_dynamic_residuals_absolute':abs(dynbc).tolist(),'radiative_constraint_dynamic_residuals_relative':(abs(dynbc)/np.maximum(dynamicscales,1e-300)).tolist(),'constraint_L2_over_state_L2':float(cn/max(un,1e-300)),'constraint_maxima':{name:float(np.max(abs(z))) for name,z in [('Theta',th),('Q',q),('H',hh),('M',mm)]}}
 if profiles:
  scale=max(np.max(abs(th)),1e-300);phase=np.exp(-1j*np.angle(th[np.argmax(abs(th))]));result['profile']={'r_M':r.tolist(),'normalization':'max_abs_Theta'}
  for name,z in [('Theta',th),('Q',q),('H',hh),('M',mm)]:result['profile'][name+'_real']=(phase*z/scale).real.tolist();result['profile'][name+'_imag']=(phase*z/scale).imag.tolist()
  result['full_state_real']=(phase*u/scale).real.tolist();result['full_state_imag']=(phase*u/scale).imag.tolist()
 return result

def spectrum(degree,inner=.2,outer=4,rate=.1,falloff='areal'):
 start=time.monotonic();A,B,raw,r,D,C,rows,labels,replaced=build(degree,inner,outer,rate,falloff);values,vectors=linalg.eig(A,B,check_finite=False)
 finite=np.isfinite(values);ii=np.flatnonzero(finite);stats={'finite_count':int(finite.sum()),'infinite_count':int((~finite).sum()),'max_real_per_M':float(values[ii].real.max()),'positive_count_1e-8':int(np.sum(values[ii].real>1e-8))}
 low=ii[abs(values[ii])<2];low=low[np.argsort(values[low].real)[::-1]][:30]
 # Also explicitly retain the branch nearest the independent constraint-only value.
 target={'none':.0262079071663,'coordinate':-.1123134495271,'areal':-.072790693446}[falloff]
 nearest=ii[np.argsort(abs(values[ii]-target))[:4]]
 result={'degree':degree,'inner_M':inner,'outer_M':outer,'kappa1':rate,'sigma':'kappa1*alpha','falloff':falloff,'state_order':['u','h','k','Theta','A_TF','Gamma','ell','shift'],'gauge_boundary':'homogeneous incoming characteristic amplitudes, equivalent to stationary zero_rate for nonzero eigenvalues','characteristic_count':characteristic_count(inner,outer),'boundary_rows':labels,'spectrum_summary':stats,'fastest_finite_mode':inspect(values[ii[np.argmax(values[ii].real)]],vectors[:,ii[np.argmax(values[ii].real)]],raw,r,D,C,rows,labels,replaced,falloff),'wall_seconds':time.monotonic()-start,'low_frequency_modes':[inspect(values[j],vectors[:,j],raw,r,D,C,rows,labels,replaced,falloff) for j in low],'near_constraint_only_branch':[inspect(values[j],vectors[:,j],raw,r,D,C,rows,labels,replaced,falloff,True) for j in nearest]}
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--degree',type=int,default=32);p.add_argument('--inner',type=float,default=.2);p.add_argument('--outer',type=float,default=4);p.add_argument('--rate',type=float,default=.1);p.add_argument('--falloff',choices=['none','coordinate','areal'],default='areal');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=spectrum(a.degree,a.inner,a.outer,a.rate,a.falloff);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'degree':a.degree,'summary':out['spectrum_summary'],'low':out['low_frequency_modes'][:3],'constraint_mode':out['near_constraint_only_branch'][0]},default=str)[:6500])
