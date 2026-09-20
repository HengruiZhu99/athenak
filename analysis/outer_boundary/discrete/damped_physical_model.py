"""Bounded weak-field 2D differential damped-constraint boundary model.
Compare active D4/D2 physical-Q helper versus volume D6E/D6E definitions.
The retained gauge/TT conditions remain the original zero_rate choices.
"""
import argparse,json
from pathlib import Path
import numpy as np
from corner_model import build,analyze,active_derivative,char

def matrix(n=8,h=32.,degree=1,helper='active',tt='zero_rate'):
  old,x=build(n=n,h=h,degree=degree,return_aux=True)
  V,S,H,A,G,B,ds,d2,dbs=[x[j] for j in ['V','S','H','A','G','B','ds','d2','dbs']]
  N=n*n;dim=20*N;I=np.eye(n);zero=np.zeros((N,N))
  if helper=='volume':dm=do=ds
  elif helper=='active':
    d4=active_derivative(n,h,4);dm=[np.kron(I,d4),np.kron(d4,I),zero];do=dbs
  else:raise ValueError(helper)
  Q=[G[i]-sum(dm[j]@H[i][j] for j in range(3)) for i in range(3)]
  Qt=[q@V for q in Q]
  left,_=char.scalar_left(1.,1.,2.,1.,1);lp=left[:,:4];ld=left[:,4:]
  vh=[[V[N:2*N],V[3*N:4*N],V[4*N:5*N]],[V[3*N:4*N],V[2*N:3*N],V[5*N:6*N]],[V[4*N:5*N],V[5*N:6*N],-V[N:2*N]-V[2*N:3*N]]]
  va=[[V[8*N:9*N],V[10*N:11*N],V[11*N:12*N]],[V[10*N:11*N],V[9*N:10*N],V[12*N:13*N]],[V[11*N:12*N],V[12*N:13*N],-V[8*N:9*N]-V[9*N:10*N]]]
  proj=lambda T,u,v:sum(u[i]*v[j]*T[i][j] for i in range(3) for j in range(3))
  L=V.copy()
  for y in range(n):
   for xx in range(n):
    sx=-1 if xx==0 else 1 if xx==n-1 else 0;sy=-1 if y==0 else 1 if y==n-1 else 0
    if not(sx or sy):continue
    pos=y*n+xx;nu=np.array([sx,sy,0.])/np.hypot(sx,sy);t1=np.array([0.,0.,1.]);t2=np.cross(nu,t1)
    dn=sum(nu[j]*dbs[j][pos] for j in range(3));df=sum(nu[j]*do[j][pos] for j in range(3))
    qn=sum(nu[j]*Q[j][pos] for j in range(3))
    ft=V[7*N+pos]+df@S[7]+.1*S[7][pos]-.05*qn
    fq=[Qt[i][pos]+df@Q[i]+.1*Q[i][pos] for i in range(3)]
    dqh=np.array([[dn@vh[i][j] for j in range(3)] for i in range(3)])
    dqb=np.array([dn@V[(17+i)*N:(18+i)*N] for i in range(3)])
    dq=np.vstack([dn@V[:N],proj(dqh,nu,nu),dn@V[16*N:17*N],nu@dqb])
    pr=np.vstack([V[6*N+pos],V[7*N+pos],proj(va,nu,nu)[pos],sum(nu[i]*V[(13+i)*N+pos] for i in range(3))])
    rates=lp@pr+ld@dq;delta=-rates;delta[2]=-ft;delta[3]=sum(nu[i]*fq[i] for i in range(3))
    pr+=np.linalg.solve(lp,delta)
    gam=nu[:,None]*pr[3]
    tensor=(nu[:,None]*nu[None,:]-.5*(t1[:,None]*t1[None,:]+t2[:,None]*t2[None,:]))[:,:,None]*pr[2]
    for tang in [t1,t2]:
      ga=-(tang@dqb)
      gorig=sum(tang[i]*V[(13+i)*N+pos] for i in range(3))
      an=proj(va,nu,tang)[pos]-.5*(sum(tang[i]*fq[i] for i in range(3))+ga-gorig)
      gam+=tang[:,None]*ga
      tensor+=(nu[:,None]*tang[None,:]+tang[:,None]*nu[None,:])[:,:,None]*an
    ap=.25*(proj(dqh,t1,t1)-proj(dqh,t2,t2));ac=.5*proj(dqh,t1,t2)
    if tt=='weyl':
      dn6=sum(nu[j]*ds[j] for j in range(3));dt1=sum(t1[j]*ds[j] for j in range(3));dt2=sum(t2[j]*ds[j] for j in range(3))
      def dd(u,v):return sum(u[i]*v[j]*(d2[i] if i==j else ds[i]@ds[j]) for i in range(3) for j in range(3))
      aplus=.5*(proj(A,t1,t1)-proj(A,t2,t2));across=proj(A,t1,t2)
      an1=proj(A,nu,t1);an2=proj(A,nu,t2);q1=sum(t1[j]*Q[j] for j in range(3));q2=sum(t2[j]*Q[j] for j in range(3))
      ap=(-dn6@aplus+.5*(dt1@an1-dt2@an2)-.5*(dd(t1,t1)-dd(t2,t2))@S[16]+.5*(dt1@q1-dt2@q2))[pos]
      ac=(-dn6@across+.5*(dt1@an2+dt2@an1)-dd(t1,t2)@S[16]+.5*(dt1@q2+dt2@q1))[pos]
    tensor+=(t1[:,None]*t1[None,:]-t2[:,None]*t2[None,:])[:,:,None]*ap
    tensor+=(t1[:,None]*t2[None,:]+t2[:,None]*t1[None,:])[:,:,None]*ac
    L[6*N+pos]=pr[0];L[7*N+pos]=pr[1]
    for f,(i,j) in zip(range(8,13),[(0,0),(1,1),(0,1),(0,2),(1,2)]):L[f*N+pos]=tensor[i,j]
    for i in range(3):L[(13+i)*N+pos]=gam[i]
  return L

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=8);p.add_argument('--h',type=float,default=32.);p.add_argument('--degree',type=int,default=1);p.add_argument('--helper',default='active');p.add_argument('--tt',default='zero_rate');p.add_argument('--out',required=True);a=p.parse_args()
 param=dict(n=a.n,h=a.h,degree=a.degree,helper=a.helper,tt=a.tt)
 result=dict(parameters=param,**analyze(matrix(**param),a.n,a.h));Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
