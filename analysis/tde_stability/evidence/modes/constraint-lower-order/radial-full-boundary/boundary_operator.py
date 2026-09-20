"""Linear spherical full-state boundary/constraint operators.

Order U=(delta_chi,h_nn,delta_Khat,Theta,A_TF_nn,Gamma_radial,delta_alpha,delta_beta).
No AthenaK source. Gauge rows are the existing zero-rate principal characteristic
rows linearized on the stationary background; nonzero temporal eigenvalues and
homogeneous incoming gauge data imply zero characteristic amplitudes.
"""
import numpy as np

NVAR=8
U,H,K,TH,A,GM,LAP,BETA=range(NVAR)

def chebyshev(degree,inner,outer):
 x=np.cos(np.pi*np.arange(degree+1)/degree)
 cc=np.r_[2.,np.ones(degree-1),2.]*(-1.)**np.arange(degree+1)
 z=x[:,None]-x[None,:]
 D=(cc[:,None]/cc[None,:])/(z+np.eye(degree+1));D-=np.diag(D.sum(axis=1));D*=2/(outer-inner)
 r=(inner+outer)/2+(outer-inner)*x/2
 assert np.max(abs(D@r-1))<1e-8
 return r,D

def constraints(r,D):
 """Independent physical H,m,Q,Theta definitions, including all background terms."""
 n=len(r);R=1+r;al=r/R;chi=al*al;kk=1/R**2
 cp=2*r/R**3;cpp=2*(1-2*r)/R**4
 I=np.eye(n);Z=np.zeros_like(I);D2=D@D
 rows={name:[Z.copy() for _ in range(8)] for name in ('H','M','Q','Theta','C1','C2')}
 hh=rows['H'];hh[U]=2*D2+np.diag(4/r-5*cp/chi)@D+np.diag(2.5*cp**2/chi**2)
 hh[H]=np.diag(chi)@D2+np.diag(5*chi/r-2*cp)@D+np.diag(3*chi/r**2-2*cpp-4*cp/r+2.5*cp**2/chi+8*kk**2/3)
 hh[K]=np.diag(4*kk/3);hh[TH]=np.diag(8*kk/3);hh[A]=np.diag(4*kk)
 mm=rows['M'];mm[U]=np.diag(2*kk/chi)@D-np.diag(2*kk*cp/chi**2)
 mm[H]=np.diag(5*kk/3)@D+np.diag(2*kk/(3*R))
 mm[K]=-2*D/3;mm[TH]=-4*D/3;mm[A]=D+np.diag(3/R)
 rows['Q'][GM]=I.copy();rows['Q'][H]=-D-np.diag(3/r)
 rows['Theta'][TH]=I.copy()
 rows['C1'][TH]=np.diag(al);rows['C1'][GM]=np.diag(chi/2);rows['C1'][U]=D.copy()
 rows['C2'][K]=np.diag(4/(3*al));rows['C2'][TH]=np.diag(2/(3*al));rows['C2'][A]=-np.diag(2/al);rows['C2'][GM]=-I;rows['C2'][H]=D.copy()
 return {name:np.hstack(v) for name,v in rows.items()}

def gauge_characteristic(r,D,index,mode,sign,shift_driver=2.):
 """Global increasing-r derivative convention; sign is root mu/|mu|."""
 n=len(r);R=1+r[index];al=r[index]/R;ch=al**2
 row=np.zeros(8*n);ei=np.eye(n)[index]
 if mode=='lapse':
  mu=sign*np.sqrt(2*al*ch)
  row[K*n:(K+1)*n]=-mu/ch*ei;row[LAP*n:(LAP+1)*n]=D[index]
 elif mode=='shift':
  sq=4*shift_driver/3;mu=sign*np.sqrt(sq);sepL=ch*2*al-mu**2;sepC=ch*al**2-mu**2
  row[K*n:(K+1)*n]=al*mu**2*sepC*ei
  row[TH*n:(TH+1)*n]=.5*al*sq*sepL*ei
  row[GM*n:(GM+1)*n]=.25*mu*(4*ch*al**2-3*mu**2)*sepL*ei
  row[U*n:(U+1)*n]=.5*al**2*mu*sepL*D[index]
  row[LAP*n:(LAP+1)*n]=-ch*al*mu*sepC*D[index]
  row[BETA*n:(BETA+1)*n]=sepL*sepC*D[index]
 else:raise ValueError(mode)
 return row

def physical_boundary_rows(r,D,rate=.1,scaled=False,falloff='areal'):
 n=len(r);al=r/(1+r);ch=al**2;b=r/(1+r)**2;c=al**2;v=c-b;ss=rate if scaled else rate*al
 C=constraints(r,D);omegaT={'none':0*r,'coordinate':1/r,'areal':1/(1+r)}[falloff];omegaQ=0*r if falloff=='none' else 1/r
 Fth=np.diag(al/2)@C['H']+np.diag(c)@D@C['Theta']+np.diag(al*ch/2)@D@C['Q']+np.diag(al*ch/r)@C['Q']+np.diag(v*omegaT-2*ss)@C['Theta']
 Fq=np.diag(2*al)@C['M']+np.diag(2*al)@D@C['Theta']+np.diag(c)@D@C['Q']+np.diag(v*omegaQ-2*ss)@C['Q']
 rows=[gauge_characteristic(r,D,0,'lapse',1),gauge_characteristic(r,D,0,'shift',1),Fth[0],Fq[0],gauge_characteristic(r,D,n-1,'shift',-1)]
 labels=['outer_lapse_gauge','outer_shift_gauge','outer_Theta_radiation','outer_Q_radiation','inner_shift_gauge']
 return np.array(rows),labels,C

def characteristic_count(inner,outer,shift_driver=2.):
 out={}
 for name,r in [('inner',inner),('outer',outer)]:
  a=r/(1+r);b=r/(1+r)**2; speeds={q:[-b-z,-b+z] for q,z in [('light',a*a),('lapse',np.sqrt(2*a**3)),('shift',np.sqrt(4*shift_driver/3))]}
  out[name]={'r_M':r,'physical_coordinate_speeds':speeds,'incoming_counts':{q:int(sum(v>0 if name=='inner' else v<0 for v in vv)) for q,vv in speeds.items()}}
 return out

if __name__=='__main__':
 import json
 r,D=chebyshev(48,.2,4);rows,labels,C=physical_boundary_rows(r,D)
 print(json.dumps({'characteristics':characteristic_count(.2,4),'boundary_rows':labels,'row_rank':int(np.linalg.matrix_rank(rows))},indent=2))
