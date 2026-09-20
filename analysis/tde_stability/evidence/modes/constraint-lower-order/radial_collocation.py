"""Independent original four-field annulus eigenproblem, Chebyshev collocation.

The two incoming outer wave conditions lambda W+(c-beta)W'=0 replace
outer H/M evolution rows. Optional wave_power=p adds p*W/r to the
spatial derivative in this BC; p=1 is a spherical outgoing approximation.
No inner data are imposed inside the horizon.
This diagnostic has no first-order wave-reduction constraint variables.
"""
import radial_modes as rm
import numpy as np
from scipy import linalg
import json,time
from pathlib import Path

def build(n,inner,outer,rate,scaled,wave_power=0.,theta_areal=False):
    x=np.cos(np.pi*np.arange(n+1)/n)
    cc=np.r_[2,np.ones(n-1),2]*(-1.)**np.arange(n+1)
    dx=x[:,None]-x[None,:]
    dr=(cc[:,None]/cc[None,:])/(dx+np.eye(n+1));dr-=np.diag(dr.sum(axis=1));dr*=2/(outer-inner)
    rad=(outer+inner)/2+(outer-inner)*x/2
    assert np.max(abs(dr@rad-1))<1e-8
    nn=n+1;v=rm.values(rad,rate,scaled)
    a,ch,b,kk,ss,da,dc,db,_=v[:9]
    sp=np.zeros(nn) if scaled else rate*da
    Ad=np.zeros((4,4,nn));Al=np.zeros((4,4,nn))
    # Order: Theta,Q,H,M. Current-Z4c constraint subsystem.
    Ad[0,0]=b;Ad[0,1]=a*ch/2;Al[0,0]=-2*ss;Al[0,1]=a*ch/rad;Al[0,2]=a/2
    Ad[1,0]=2*a;Ad[1,1]=b;Al[1,1]=-2*ss;Al[1,3]=2*a
    Ad[2,1]=4*a*ch*kk;Ad[2,2]=b;Ad[2,3]=-2*a*ch
    Al[2,0]=-4*ss*kk;Al[2,2]=2*a*kk;Al[2,3]=-4*a*ch/rad+a*dc-4*ch*da
    Ad[3,0]=2*ss;Ad[3,1]=-a*dc;Ad[3,2]=-a/2;Ad[3,3]=b
    Al[3,0]=2*sp;Al[3,1]=-(2*ch*da+a*dc)/rad;Al[3,2]=-da;Al[3,3]=db+a*kk
    mat=np.block([[Ad[i,j,:,None]*dr+np.diag(Al[i,j]) for j in range(4)] for i in range(4)])
    mass=np.eye(4*nn)
    for z in range(2):
      row=(z+2)*nn
      mat[row]=0;mass[row]=0
      mat[row,z*nn:(z+1)*nn]=-(a[0]**2-b[0])*dr[0]
      mat[row,z*nn]-=(a[0]**2-b[0])*wave_power/(rad[0]+(1 if theta_areal and z==0 else 0))
      mass[row,z*nn]=1
    return mat,mass,rad,v,dr,Ad,Al

def spectrum(n,inner,outer,rate,scaled,wave_power=0.,theta_areal=False):
    t=time.monotonic();A,B,r,v,D,Ad,Al=build(n,inner,outer,rate,scaled,wave_power,theta_areal)
    ev,uu=linalg.eig(A,B,check_finite=False)
    finite=ev[np.isfinite(ev)]
    stats={"finite_eigenvalue_count":len(finite),"max_real_per_M":float(np.max(finite.real)),"positive_count_threshold_1e-8":int(np.sum(finite.real>1e-8))}
    good=np.flatnonzero(np.isfinite(ev)&(abs(ev)<2));indices=good[np.argsort(ev[good].real)[::-1]][:20]
    rows=[];profiles=[];nn=len(r)
    for j in indices:
      lam=ev[j];u=uu[:,j].reshape(4,nn);du=np.array([D@x for x in u]);rhs=np.einsum('abn,bn->an',Ad,du)+np.einsum('abn,bn->an',Al,u)
      res=lam*u-rhs;rel=[]
      for c in range(4):rel.append(float(np.linalg.norm(res[c,1:])/max(np.linalg.norm((lam*u[c])[1:])+np.linalg.norm(rhs[c,1:]),1e-300)))
      wbc=lam*u[:2,0]+(v[0,0]**2-v[2,0])*(du[:2,0]+wave_power*u[:2,0]/np.array([r[0]+int(theta_areal),r[0]]))
      row={'eigenvalue_real_per_M':float(lam.real),'eigenvalue_imag_per_M':float(lam.imag),'original_equation_relative_residuals_excluding_outer_endpoint':rel,'outer_characteristic_boundary_residual':float(np.linalg.norm(wbc)/np.linalg.norm(u)),'Theta_peak_r_M':float(r[np.argmax(abs(u[0]))])}
      rows.append(row)
      if len(profiles)<3:
        norm=max(abs(u[0]));phase=np.exp(-1j*np.angle(u[0,np.argmax(abs(u[0]))]));p={'r_M':r.tolist()}
        for c,name in enumerate(('Theta','Q','H','M')):p[name+'_real_normalized']=(u[c]*phase/norm).real.tolist();p[name+'_imag_normalized']=(u[c]*phase/norm).imag.tolist()
        profiles.append(dict(row,profile=p))
    return {'method':'Chebyshev generalized eigenproblem in original four fields; all finite eigenvalues with abs(lambda)<2','polynomial_degree':n,'inner_M':inner,'outer_M':outer,'rate_per_M':rate,'lapse_scaled':scaled,'outer_wave_power':wave_power,'theta_areal_weight':theta_areal,'wall_seconds':time.monotonic()-t,'all_finite_spectrum_summary':stats,'modes':rows,'leading_profiles':profiles}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=48);p.add_argument('--inner',type=float,default=.2);p.add_argument('--outer',type=float,default=4);p.add_argument('--rate',type=float,default=.1);p.add_argument('--scaled',action='store_true');p.add_argument('--wave-power',type=float,default=0.);p.add_argument('--theta-areal',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=spectrum(a.n,a.inner,a.outer,a.rate,a.scaled,a.wave_power,a.theta_areal);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['modes'][:6],indent=2))
