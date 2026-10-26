#!/usr/bin/env python3
import sys,json,math,time,hashlib
from pathlib import Path
import numpy as np
np.seterr(all='raise')
base,sbp,out=map(Path,sys.argv[1:4]);out.mkdir(exist_ok=True)
s=json.loads(sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
def apply(a,b):return np.einsum('ij,jk->ik',a,b)
def spatial(n):
 m=len(weights);h=np.ones(n);h[:m]=weights;h[-m:]=weights[::-1];D=np.zeros((n,n));width=rows.shape[1];D[:m,:width]=rows;D[-m:,-width:]=-rows[::-1,::-1]
 for i in range(m,n-m):
  for j,c in [(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)]:D[i,i+j]=c
 tl=np.zeros(n);tl[:m]=trace;tr=tl[::-1];assert np.linalg.norm(h[:,None]*D+D.T*h[None,:]-(np.outer(tr,tr)-np.outer(tl,tl)))<1e-10
 return h/n,D*n,np.vstack([tl,tr])
def run(family,sample,mat,jet,n,half):
 start=time.monotonic();tag=f'{family}-N{n}'+('-half' if half else '');receipt=out/(tag+'.json');payload=out/(tag+'.npz')
 if receipt.exists() or payload.exists():raise RuntimeError('Refuse duplicate '+tag)
 h,D,E=spatial(n);x=(np.arange(n)+.5)/n;H=h[:,None];Q=h[:,None]*D;Bface=np.diag([-1.,1.]);QN=.5*np.block([[Q-Q.T,apply(E.T,Bface)],[-apply(Bface,E),Bface]])
 direction=np.array(jet['coordinate']);direction/=np.linalg.norm(direction);G=np.array(mat['G']).reshape(3,3);beta=float(np.dot(mat['beta'],direction));c=mat['N']*math.sqrt(float(np.einsum('i,ij,j->',direction,G,direction)));lo=-beta-c;hi=-beta+c;B=np.array(mat['source_matrix_4x20']).reshape(4,20);Bu=B[:,:4];Bv=B[:,4:8];Bq=np.einsum('kij,j->ki',B[:,8:].reshape(4,4,3),direction)
 kmax=max(math.pi*n,float(np.max(np.sum(abs(D),axis=1))));eigen=[]
 for k in np.linspace(0,kmax,33):
  F=np.block([[1j*beta*k*np.eye(4),np.eye(4)],[Bu+1j*k*Bq-c*c*k*k*np.eye(4),Bv+1j*beta*k*np.eye(4)]]);eigen.extend(np.linalg.eigvals(F))
 eigen=np.array(eigen);dtbase=min(.05/(1.1*max(abs(lo),abs(hi))*max(np.sum(abs(D),axis=1))),.01/max(1.,max(abs(eigen))))/(2 if half else 1)
 z=eigen*dtbase;poly=1+z+z*z/2+z*z*z/6+z*z*z*z/24;rkerror=float(max(abs(poly-np.exp(z))/np.maximum(1.,abs(np.exp(z)))))
 if rkerror>1e-5:raise RuntimeError('Local high-k admission failed')
 def coeff(xx,t):
  hp=hi*(1+.04*(xx-.5)+.02*np.sin(t));lp=lo*(1+.03*(xx-.5)-.01*np.sin(t));ht=.02*hi*np.cos(t)+np.zeros_like(xx);lt=-.01*lo*np.cos(t)+np.zeros_like(xx);return -(hp+lp)/2,(hp-lp)/2,-hp*lp,-(ht+lt)/2,-(lt*hp+lp*ht),hp,lp
 maxidentity=0.
 def rhs(U,w,t):
  nonlocal maxidentity
  b,cc,a,bt,at,hp,lp=coeff(x,t);bf,cf,af,_,_,hf,lf=coeff(np.array([0.,1.]),t);bx=np.einsum('ij,j->i',D,b);ax=np.einsum('ij,j->i',D,a);lower=bt-b*bx-ax;q=apply(D,U);ew=apply(E,w);eq=apply(E,q);pw=np.vstack([w,ew]);bext=np.r_[b,bf][:,None]
  flux=apply(QN,bext*pw)+bext*apply(QN,pw);shift=(flux[:n]+apply(E.T,flux[n:]))/H-bx[:,None]*w
  weak=(-apply(D.T,H*a[:,None]*q)+apply(E.T,np.array([-1.,1.])[:,None]*af[:,None]*eq))/H
  V=w-b[:,None]*q;source=apply(U,Bu.T)+apply(V,Bv.T)+apply(q,Bq.T)
  wd=weak+lower[:,None]*q+shift+source
  wd-=E[0,:,None]/H*hf[0]*(ew[0]+lf[0]*eq[0]);wd+=E[1,:,None]/H*lf[1]*(ew[1]+hf[1]*eq[1])
  qw=apply(D,w);actual=float(np.sum(H*w*wd)+np.sum(H*a[:,None]*q*qw)+np.sum(H*U*w)+.5*np.sum(H*at[:,None]*q*q));expected=float(-np.sum(cf[:,None]*ew*ew)-np.sum(H*bx[:,None]*w*w)+np.sum(H*lower[:,None]*q*w)+np.sum(H*U*w)+np.sum(H*w*source)+.5*np.sum(H*at[:,None]*q*q));maxidentity=max(maxidentity,abs(actual-expected)/max(1.,abs(actual),abs(expected)))
  return w,wd,expected
 def energy(U,w,t):
  a=coeff(x,t)[2];q=apply(D,U);return float(.5*np.sum(H*(w*w+a[:,None]*q*q+U*U)))
 sig=(x-.4)/.18;bump=np.zeros(n);db=np.zeros(n);active=abs(sig)<1;z0=1-sig[active]**2;bump[active]=np.exp(1-1/z0);db[active]=bump[active]*(-2*sig[active]/(.18*z0*z0));pol=np.array([1,.3,-.2,.1]);U=1e-3*bump[:,None]*pol;w=-coeff(x,0)[5][:,None]*1e-3*db[:,None]*pol;t=0.;work=0.;E0=energy(U,w,0);Epeak=E0;maxbalance=0.;cycles=0;snapU=[];snapw=[];snapQ=[];history=[]
 for target in [.005,.01,.02]:
  while t<target:
   if time.monotonic()-start>180:raise RuntimeError('CPU timeout '+tag)
   dt=min(dtbase,target-t);k1=rhs(U,w,t);k2=rhs(U+dt*k1[0]/2,w+dt*k1[1]/2,t+dt/2);k3=rhs(U+dt*k2[0]/2,w+dt*k2[1]/2,t+dt/2);k4=rhs(U+dt*k3[0],w+dt*k3[1],t+dt)
   U+=dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6;w+=dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6;work+=dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6;t+=dt;cycles+=1;en=energy(U,w,t);Epeak=max(Epeak,en);maxbalance=max(maxbalance,abs(en-E0-work)/max(E0,Epeak))
   if not np.all(np.isfinite(U)) or not math.isfinite(en) or en/E0>1e12:raise RuntimeError('Diagnostic cap '+tag)
  snapU.append(U.copy());snapw.append(w.copy());snapQ.append(apply(D,U));history.append({'time':t,'energy':en,'integrated_work':work,'relative_balance':abs(en-E0-work)/max(E0,Epeak)})
 np.savez_compressed(payload,x=x,h=h,times=[.005,.01,.02],U=snapU,w=snapw,Q=snapQ)
 r=dict(family=family,sample=sample,radius=jet['radius'],nodes=n,half=half,cycles=cycles,dt=dtbase,high_k_max=kmax,high_k_local_RK_error=rkerror,energy_identity_error=maxidentity,energy_balance_error=maxbalance,energy_amplification=Epeak/E0,history=history,elapsed=time.monotonic()-start,payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest());receipt.write_text(json.dumps(r,indent=2)+'\n');print(tag,cycles,r['elapsed'],maxbalance,flush=True)
for family in ['synthetic','extended','massive']:
 mats=[json.loads(x) for x in (base/(family+'-matrices.jsonl')).read_text().splitlines()];jets=[json.loads(x) for x in (base/(family+'-jets.jsonl')).read_text().splitlines()];sample=next(i for i,j in enumerate(jets) if j['radius']==.98)
 for n,half in [(32,False),(48,False),(64,False),(128,False),(160,False),(48,True)]:run(family,sample,mats[sample],jets[sample],n,half)
