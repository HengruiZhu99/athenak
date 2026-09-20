"""Variable-coefficient radial FD compatibility diagnostic, NOT full3D discretization.
Uses exact radial continuum coefficients with AthenaK D6/Dxx6/Lx6/KO8 stencils,
polynomial residual ghosts and mixed-order physical-constraint boundary helper.
"""
from pathlib import Path
import sys,json,argparse,math
import numpy as np
from scipy.linalg import eig
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'radial'))
from first_order import volume,characteristics,PREV
sys.path.insert(0,str(PREV/'radial-full-boundary'))
from boundary_operator import gauge_characteristic

def weight(nodes,x,der=0):
 z=np.asarray(nodes)-x;return np.linalg.solve(np.array([z**k for k in range(len(nodes))]),np.array([float(math.factorial(der)) if k==der else 0. for k in range(len(nodes))]))

def extension(n,points):
 E=np.zeros((n+8,n));E[4:4+n]=np.eye(n)
 for j in range(-4,0):E[j+4,:points]=weight(np.arange(points),j)
 for j in range(n,n+4):E[j+4,n-points:]=weight(np.arange(n-points,n),j)
 return E

def stencil(n,E,offset,coef,h,power=1):
 return np.array([np.asarray(coef)@E[4+i+np.asarray(offset)] for i in range(n)])/h**power

def activeD(n,h,order):
 D=np.zeros((n,n));w=order+1
 for i in range(n):
  start=min(max(i-order//2,0),n-w);js=np.arange(start,start+w);D[i,js]=weight(js,i,1)/h
 return D

def build(n,extrap=4,inner=.2,outer=4,helper_order=2,tau=1,rate=.1,eps=.5,transport_order=2):
 r=np.linspace(inner,outer,n);h=r[1]-r[0];E=extension(n,extrap)
 Dc=stencil(n,E,range(-3,4),[-1/60,3/20,-3/4,0,3/4,-3/20,1/60],h)
 D2=stencil(n,E,range(-3,4),[1/90,-3/20,1.5,-49/18,1.5,-3/20,1/90],h,2)
 Du=stencil(n,E,range(-2,5),[1/30,-2/5,-7/12,4/3,-.5,2/15,-1/60],h)
 KO=-eps/256*stencil(n,E,range(-4,5),[1,-8,28,-56,70,-56,28,-8,1],h)
 a=r/(r+1);chi=a*a;b=r/(r+1)**2;v=chi-b;A2,A1,A0=volume.coefficients(r,rate)
 raw=np.block([[A2[:,i,j,None]*D2+A1[:,i,j,None]*Dc+np.diag(A0[:,i,j])+(b[:,None]*(Du-Dc)+KO if i==j else 0) for j in range(8)] for i in range(8)])
 Db=activeD(n,h,2);Dh=activeD(n,h,helper_order) if helper_order else Dc;Dt=activeD(n,h,transport_order) if transport_order else Dc
 I=np.eye(n);Q=np.zeros((n,8*n));Q[:,5*n:6*n]=I;Q[:,n:2*n]=-Dh-np.diag(3/r)
 T=np.zeros((n,8*n));T[:,3*n:4*n]=I
 # Reference covariant transport uses derivative of the reference physical metric.
 omegaq=1/(r+1)-.5*chi*(Db@(1/chi))
 Ft=T@raw+np.diag(v)@Dt@T+np.diag(v/(r+1))@T
 Fq=Q@raw+np.diag(v)@Dt@Q+np.diag(v*omegaq)@Q
 out=raw.copy();L,lam,_=characteristics(r[-1]);end=n-1;ids=np.array([2,3,4,5])*n+end
 Wg=np.array([gauge_characteristic(r,Db,end,'lapse',1),gauge_characteristic(r,Db,end,'shift',1)])
 delta=np.array([-Wg[0]@raw,-Wg[1]@raw,-tau*lam[2]*Ft[end]/a[end],tau*lam[3]*Fq[end]/chi[end]])
 out[ids]+=np.linalg.solve(L[:4,:4],delta)
 wi=gauge_characteristic(r,Db,0,'shift',-1);out[5*n]-=wi@raw/wi[5*n]
 return out,raw,r,Q,T

def spectrum(n,**kw):
 A,raw,r,Q,T=build(n,**kw);lam,vec=eig(A);ix=np.argsort(lam.real)[::-1];rows=[]
 for j in ix[:8]:
  z=vec[:,j].reshape(8,n);th=z[3];amp=np.sum(abs(z)**2,axis=0);rows.append({'real':float(lam[j].real),'imag':float(lam[j].imag),'state_peak_r':float(r[np.argmax(amp)]),'Theta_peak_r':float(r[np.argmax(abs(th))]),'outer3_fraction':float(amp[-3:].sum()/amp.sum()),'inner3_fraction':float(amp[:3].sum()/amp.sum())})
 return {'n':n,'options':kw,'scope':'Variable radial continuum coefficients discretized with AthenaK 1D FD stencil family; angular/background discrete terms and 3D edges/corners not represented.','max_real':float(lam.real.max()),'positive_count':int(np.sum(lam.real>1e-7)),'fastest':rows}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=[]
 for n in [16,32,64]:
  for ext in [2,4]:
   for helper in [2,4]:
    for tau in [.25,.5,1.]:
     o=spectrum(n,extrap=ext,helper_order=helper,tau=tau);out.append(o);print(n,ext,helper,tau,o['max_real'],o['fastest'][0]['state_peak_r'],flush=True)
 a.output.write_text(json.dumps(out,indent=2)+'\n')
