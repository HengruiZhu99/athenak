#!/usr/bin/env python3
"""Value-only physical metric pullback and finite-difference curvature oracle."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import mpmath as mp
mp.mp.dps=70
SMOOTH_LAYER=False

def smooth(z):
 if z<=0:return 0
 if z>=1:return 1
 if SMOOTH_LAYER:
  a=np.exp(-1/z);b=np.exp(-1/(1-z));return a/(a+b)
 return z**3*(10-15*z+6*z*z)
def smooth_mp(z):
 if z<=0:return mp.mpf(0)
 if z>=1:return mp.mpf(1)
 if SMOOTH_LAYER:
  a=mp.exp(-1/z);b=mp.exp(-1/(1-z));return a/(a+b)
 return z**3*(10-15*z+6*z*z)
def compact(rho,S=24):
 z=(rho-8)/8;w=smooth(z);wp=(w*(1-w)*(1/z**2+1/(1-z)**2) if SMOOTH_LAYER else 30*z*z*(1-z)**2)/8 if 0<z<1 else 0
 return 1-w*rho/S,-(w+rho*wp)/S

def physical(x,M,a):
 r=np.linalg.norm(x);R=r+M;n=x/r;w=np.array([-n[1],n[0],0]);v=r*w;c=np.sqrt(M*M-a*a)
 sigma=R*R+a*a*n[2]**2;A=R*R+a*a;X=A*A-a*a*(x[0]**2+x[1]**2)
 alpha=r*np.sqrt(sigma/X);beta=(c*A*x-a*(2*M*r+M*M+a*a)*v)/X
 gamma=(sigma*np.eye(3)+a*a*(1+2*M*R/sigma)*np.outer(w,w)-a*c*(np.outer(n,w)+np.outer(w,n)))/r**2
 g=np.empty((4,4));g[1:,1:]=gamma;g[0,1:]=g[1:,0]=gamma@beta;g[0,0]=-alpha**2+beta@gamma@beta
 return g

def conformal(y,M,a):
 rho=np.linalg.norm(y);o,op=compact(rho);n=y/rho;r=rho/o;lam=o-rho*op
 hp=r/np.sqrt(r*r+100)
 if M:hp=smooth((r-8)/8)*(hp+(2*M+np.sqrt(M*M-a*a))/r)
 dx=(np.eye(3)-np.outer(n,n))/o+lam/o**2*np.outer(n,n)
 jac=np.zeros((4,4));jac[0,0]=1;jac[0,1:]=hp*lam/o**2*n;jac[1:,1:]=dx
 return o*o*jac.T@physical(y/o,M,a)@jac

def conformal_mp(y,M,a):
 y=mp.matrix([mp.mpf(str(v)) for v in y]);M=mp.mpf(str(M));a=mp.mpf(str(a));rho=mp.sqrt(mp.fdot(y,y));n=y/rho
 z=(rho-8)/8;w=smooth_mp(z);wp=(w*(1-w)*(1/z**2+1/(1-z)**2) if SMOOTH_LAYER else 30*z*z*(1-z)**2)/8 if 0<z<1 else 0
 o=1-w*rho/24;op=-(w+rho*wp)/24;lam=o-rho*op;r=rho/o;R=r+M;az=mp.matrix([-n[1],n[0],0]);c=mp.sqrt(M*M-a*a)
 sig=R*R+a*a*n[2]**2;A=R*R+a*a;X=A*A-a*a*r*r*(1-n[2]**2)
 alpha=r*mp.sqrt(sig/X);beta=(c*A*r*n-a*(2*M*r+M*M+a*a)*r*az)/X
 gamma=(sig*mp.eye(3)+a*a*(1+2*M*R/sig)*az*az.T-a*c*(n*az.T+az*n.T))/(r*r)
 g=mp.matrix(4);g[0,0]=-alpha*alpha+(beta.T*gamma*beta)[0]
 for i in range(3):
  g[0,i+1]=g[i+1,0]=(gamma*beta)[i]
  for j in range(3):g[i+1,j+1]=gamma[i,j]
 hp=r/mp.sqrt(r*r+100)
 if M:
  hz=(r-8)/8;hw=smooth_mp(hz)
  hp=hw*(hp+(2*M+c)/r)
 jac=mp.matrix(4);jac[0,0]=1;dx=(mp.eye(3)-n*n.T)/o+lam/o**2*n*n.T
 for i in range(3):
  jac[0,i+1]=hp*lam/o**2*n[i]
  for j in range(3):jac[i+1,j+1]=dx[i,j]
 return o*o*jac.T*g*jac

def oracle(y,M,a):
 if np.linalg.norm(y)>=23.9:
  cov=conformal_mp(y,M,a);iv=cov**-1;b=mp.matrix(4,1);step=mp.mpf('1e-7')
  for i in range(3):
   yy=mp.matrix([mp.mpf(str(v)) for v in y]);values=[]
   for k in [-2,-1,1,2]:
    shifted=yy.copy();shifted[i]+=k*step;g=conformal_mp(shifted,M,a);values.append(mp.sqrt(-mp.det(g))*(g**-1))
   for j in range(4):b[j]+=(values[0][i+1,j]-8*values[1][i+1,j]+8*values[2][i+1,j]-values[3][i+1,j])/(12*step)
  b/=mp.sqrt(-mp.det(cov))
  return np.array(cov.tolist(),dtype=float),np.array(list(b),dtype=float)
 cov=conformal(y,M,a);b=np.zeros(4);h=1e-4*np.linalg.norm(y)
 for i in range(3):
  d=np.eye(3)[i]*h;v=[]
  for k in [-2,-1,1,2]:
   g=conformal(y+k*d,M,a);v.append(np.sqrt(-np.linalg.det(g))*np.linalg.inv(g)[i+1])
  b+=(v[0]-8*v[1]+8*v[2]-v[3])/(12*h)
 return cov,b/np.sqrt(-np.linalg.det(cov))

def curvature(y,M,a,h):
 g=conformal(y,M,a);iv=np.linalg.inv(g);dg=np.zeros((4,4,4));dd=np.zeros((4,4,4,4));eye=np.eye(3)
 for i in range(3):
  gp=conformal(y+h*eye[i],M,a);gm=conformal(y-h*eye[i],M,a)
  dg[i+1]=(gp-gm)/(2*h);dd[i+1,i+1]=(gp-2*g+gm)/h**2
  for j in range(i):
   dd[i+1,j+1]=dd[j+1,i+1]=(conformal(y+h*(eye[i]+eye[j]),M,a)-conformal(y+h*(eye[i]-eye[j]),M,a)-conformal(y+h*(-eye[i]+eye[j]),M,a)+conformal(y-h*(eye[i]+eye[j]),M,a))/(4*h*h)
 conn=np.zeros((4,4,4));dconn=np.zeros((4,4,4,4))
 for k in range(4):
  div=-iv@dg[k]@iv
  for u in range(4):
   for v in range(4):
    for w in range(4):
     for z in range(4):
      s=dg[v,z,w]+dg[w,z,v]-dg[z,v,w]
      conn[u,v,w]+=.5*iv[u,z]*s/4 # same connection accumulated four times
      dconn[k,u,v,w]+=.5*(div[u,z]*s+iv[u,z]*(dd[k,v,z,w]+dd[k,w,z,v]-dd[k,z,v,w]))
 ric=np.zeros((4,4))
 for i in range(4):
  for j in range(4):
   for k in range(4):
    ric[i,j]+=dconn[k,k,i,j]-dconn[j,k,i,k]
    for l in range(4):ric[i,j]+=conn[k,k,l]*conn[l,i,j]-conn[k,j,l]*conn[l,i,k]
 return float(np.sum(iv*ric))

def main():
 global SMOOTH_LAYER
 p=argparse.ArgumentParser();p.add_argument('geometry',type=Path);p.add_argument('--smooth-layer',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();SMOOTH_LAYER=a.smooth_layer
 records=[];maxg=maxb=maxv=0
 for row in csv.DictReader(a.geometry.open()):
  M=float(row['mass']);spin=float(row['spin']);y=np.array([float(row[k]) for k in ['x','y','z']]);rho=np.linalg.norm(y)
  expected=np.array([float(row[f'g{i}{j}']) for i in range(4) for j in range(4)]).reshape(4,4)
  if rho>=24-1e-10:continue
  cov,b=oracle(y,M,M*spin);ge=float(np.max(abs(cov@expected-np.eye(4)))/(1+np.max(abs(cov))*np.max(abs(expected))))
  target=np.array([float(row[f'b{i}']) for i in range(4)])
  be=float(np.max(abs(b-target))/(1+np.max(abs(target))));maxg=max(maxg,ge);maxb=max(maxb,be)
  if spin in [0,.5] and M in [0,1] and abs(rho-12)<1e-8:
   hs=[.004,.002,.001];vals=[curvature(y,M,M*spin,h0)/6 for h0 in hs];targetv=float(row['potential']);ve=abs(vals[-1]-targetv)/(1+abs(targetv));maxv=max(maxv,ve)
   records.append(dict(mass=M,spin=spin,y=y.tolist(),potential=targetv,steps=hs,curvature_values=vals,error=ve))
 result=dict(inverse_error=maxg,divergence_error=maxb,curvature_error=maxv,curvature_probes=records,passed=maxg<1e-10 and maxb<2e-6 and maxv<2e-6,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='curvature_probes'},indent=2))
if __name__=='__main__':main()
