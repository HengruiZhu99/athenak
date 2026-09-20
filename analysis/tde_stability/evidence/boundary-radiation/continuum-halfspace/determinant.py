from pathlib import Path
import sys,json,contextlib,io
import numpy as np
from scipy import linalg,optimize
ROOT=Path(__file__).resolve().parent
import characteristic_rows as char
QP=[0,1,2,3,4,5,16,17,18,19];PP=list(range(6,16));I=np.eye(20,dtype=complex)
alpha=.6531769730886202;chi=.4266401581732121;bn=.2255366474924469

def volume(s,ky,alpha=alpha,chi=chi,beta=(bn,0.,0.),damping=True):
 S=I;c=S[0];k=S[6];theta=S[7];lapse=S[16]
 H=np.array([[S[1],S[3],S[4]],[S[3],S[2],S[5]],[S[4],S[5],-S[1]-S[2]]]);A=np.array([[S[8],S[10],S[11]],[S[10],S[9],S[12]],[S[11],S[12],-S[8]-S[9]]]);G=S[13:16];B=S[17:20]
 D=np.array([s,1j*ky,0]);lap=sum(D*D);divB=D@B;divG=D@G;V=np.zeros((20,20),complex)
 V[0]=2*alpha*chi/3*(k+2*theta)-2*chi/3*divB;V[6]=-chi*lap*lapse;V[7]=alpha*chi/2*divG+alpha*lap*c;V[16]=-2*alpha*k
 ac=[]
 for (i,j),f in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(1,6)):
  V[f]=-2*alpha*A[i,j]+D[i]*B[j]+D[j]*B[i]-(2*divB/3 if i==j else 0)
 a=np.array([[-chi*D[i]*D[j]*lapse+alpha*(-chi/2*lap*H[i,j]+chi/2*(D[i]*G[j]+D[j]*G[i])+.5*D[i]*D[j]*c) for j in range(3)] for i in range(3)])
 tr=sum(a[i,i] for i in range(3))
 for (i,j),f in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(8,13)):V[f]=a[i,j]-(tr/3 if i==j else 0)
 Q=G-np.einsum('j,ijk->ik',D,H)
 for i in range(3):
  V[13+i]=-4*alpha/3*D[i]*k-2*alpha/3*D[i]*theta+lap*B[i]+D[i]*divB/3
  V[17+i]=2*G[i]
  if damping:V[13+i]-=.2*alpha*Q[i];V[17+i]-=2*B[i]
 if damping:V[6]+=.1*alpha*theta;V[7]-=.2*alpha*theta
 V+=sum(beta*D)*I
 return V,Q

def stable_basis(lam,ky,damping=True):
 V0,_=volume(0,ky,damping=damping);Vp,_=volume(1,ky,damping=damping);Vm,_=volume(-1,ky,damping=damping);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 block=lambda V,a,b:V[np.ix_(a,b)]
 C=block(V0,QP,PP);Ci=np.linalg.inv(C);assert np.max(abs(block(V1,QP,PP)))<1e-13 and np.max(abs(block(V2,QP,PP)))<1e-13
 Q0=lam*np.eye(10)-block(V0,QP,QP);Q1=-block(V1,QP,QP)
 P0=lam*np.eye(10)-block(V0,PP,PP);P1=-block(V1,PP,PP)
 M0=P0@Ci@Q0-block(V0,PP,QP);M1=P0@Ci@Q1+P1@Ci@Q0-block(V1,PP,QP);M2=P1@Ci@Q1-block(V2,PP,QP)
 AA=np.block([[np.zeros((10,10)),np.eye(10)],[-M0,-M1]]);BB=np.block([[np.eye(10),np.zeros((10,10))],[np.zeros((10,10)),M2]])
 ss,ww=linalg.eig(AA,BB);ids=np.where(np.isfinite(ss)&(ss.real>1e-7))[0]
 if len(ids)!=10:raise RuntimeError(('stable count',len(ids),lam,ky,ss))
 modes=[];errs=[]
 for z in ids:
  s=ss[z];q=ww[:10,z];p=Ci@(Q0+s*Q1)@q;u=np.zeros(20,complex);u[QP]=q;u[PP]=p;u/=np.linalg.norm(u);V,_=volume(s,ky,damping=damping)
  errs.append(np.linalg.norm((lam*I-V)@u)/max(np.linalg.norm(V@u),abs(lam)));modes.append((s,u))
 return modes,max(errs)

def boundary(lam,s,ky,u,mode='radiation'):
 left,_=char.scalar_left(alpha,chi,2*alpha,2,1)
 p=u[[6,7,8,13]];d=s*u[[0,1,16,17]];sc=left[:,:4]@p+left[:,4:]@d
 c=alpha*np.sqrt(chi);rate=(lam+(c-bn)*s);_,Q=volume(s,ky)
 rows=[sc[0],sc[1]]
 if mode=='radiation':rows.extend([rate*u[7],rate*(Q[0]@u)])
 else:rows.extend([sc[2],sc[3]])
 for ai,gi,hi,bi in [(10,14,3,18),(11,15,4,19)]:
  rows.append(np.sqrt(2)*u[gi]+s*u[bi]);rows.append(rate*(Q[gi-13]@u) if mode=='radiation' else -2*u[ai]/np.sqrt(chi)-u[gi]+s*u[hi])
 rows.extend([-2*(u[9]+.5*u[8])/np.sqrt(chi)+s*(u[2]+.5*u[1]),-2*u[12]/np.sqrt(chi)+s*u[5]])
 return np.array(rows)

def assess(lam,ky,mode='radiation',damping=True):
 basis,err=stable_basis(lam,ky,damping);B=np.stack([boundary(lam,s,ky,u,mode) for s,u in basis],axis=1)
 # Factor mode-basis conditioning out before measuring boundary rank.
 states=np.stack([u for s,u in basis],axis=1);U,R=np.linalg.qr(states);B=B@np.linalg.inv(R)
 B/=np.maximum(np.linalg.norm(B,axis=1),1e-300)[:,None]
 sing=np.linalg.svd(B,compute_uv=False)
 return {'sigma_min':float(sing[-1]),'sigma_ratio':float(sing[-1]/sing[0]),'bulk_eigen_residual':float(err),'stable_roots':[[s.real,s.imag] for s,u in basis],'basis_condition':float(np.linalg.cond(states))}
if __name__=='__main__':
 out=[]
 for k in [np.pi,2*np.pi,4*np.pi]:
  for damping in [False,True]:
   for mode in ['zero_rate','radiation']:
    for x in [.01,.1,.5,1.,1.2,1.5,2.,4.,8.]:
     a=assess(complex(x),k,mode,damping);out.append(dict(k=k,lam=[x,0],mode=mode,damping=damping,**a))
 (ROOT/'real-scan.json').write_text(json.dumps(out,indent=2)+'\n')
 for k in [np.pi,2*np.pi,4*np.pi]:
  for mode in ['zero_rate','radiation']:
   rows=[q for q in out if q['k']==k and q['mode']==mode and q['damping']]
   m=min(rows,key=lambda q:q['sigma_min']);print(k,mode,m['lam'],m['sigma_min'],m['bulk_eigen_residual'],flush=True)
