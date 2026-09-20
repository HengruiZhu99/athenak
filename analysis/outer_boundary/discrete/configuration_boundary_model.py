"""Reflecting/outgoing all-q comparison with compatible p-only differential BC.
Not a physical-constraint-preserving production boundary proposal.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
from corner_model import build
QP=[0,1,2,3,4,5,16,17,18,19];PP=list(range(6,16))

def matrix(n,h,mode='dirichlet',degree=1):
  _,x=build(n=n,h=h,degree=degree,return_aux=True);V=x['V'];db=x['dbs'];N=n*n
  qids=np.concatenate([np.arange(j*N,(j+1)*N) for j in QP]);vq=V[qids]
  c=V[np.ix_([j*N for j in QP],[j*N for j in PP])];ci=np.linalg.inv(c)
  L=V.copy()
  for y in range(n):
   for xx in range(n):
    sx=-1 if xx==0 else 1 if xx==n-1 else 0;sy=-1 if y==0 else 1 if y==n-1 else 0
    if not(sx or sy):continue
    pos=y*n+xx;qrows=[j*N+pos for j in QP]
    rhs=V[np.ix_(qrows,qids)]@vq
    if mode=='outgoing':
      dn=(sx*db[0][pos]+sy*db[1][pos])/np.hypot(sx,sy)
      for j,f in enumerate(QP):rhs[j]+=dn@V[f*N:(f+1)*N]
    elif mode!='dirichlet':raise ValueError(mode)
    L[[j*N+pos for j in PP]]=-ci@rhs
  return L

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=12);p.add_argument('--mode',default='dirichlet');p.add_argument('--degree',type=int,default=1);p.add_argument('--out',required=True);a=p.parse_args()
 tick=time.monotonic();L=matrix(a.n,32.,a.mode,a.degree);w=eigvals(L,check_finite=False);i=np.argmax(w.real)
 out=dict(parameters=dict(n=a.n,h=32.,mode=a.mode,degree=a.degree),real=float(w[i].real),imag=float(w[i].imag),positive=int(sum(w.real>1e-8)),near_neutral=int(sum(abs(w)<1e-8)),wall_seconds=time.monotonic()-tick)
 Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
