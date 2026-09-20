"""Frozen weak-field production zero_rate boundary symbol, standalone diagnostic.
Exact scalar characteristic rows are imported from the immutable production snapshot.
The volume retains constant alpha/chi/beta and production kappa/eta/lapse damping;
trumpet coefficient gradients, background K/A and finite-difference closures are
intentionally absent. Domain x<0; modes exp(lambda*t+s*x+i*k*y) decay for Re(s)>0.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import contextlib,importlib.util,io,json
import numpy as np
from scipy import linalg,optimize
ROOT=Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
 import characteristic_reference as char
QP=np.array([0,1,2,3,4,5,16,17,18,19]);PP=np.arange(6,16);I=np.eye(20,dtype=complex)
@dataclass(frozen=True)
class Config:
 alpha:float=1.
 chi:float=1.
 beta_n:float=0.
 G:float=1.
 kappa:float=.1
 eta:float=2.
 lapse_damping:float=.1

def volume(s,k,cfg):
 a,c,b,G,kap,eta,ld=cfg.alpha,cfg.chi,cfg.beta_n,cfg.G,cfg.kappa,cfg.eta,cfg.lapse_damping
 S=I;dc=S[0];kh=S[6];theta=S[7];lapse=S[16]
 H=np.array([[S[1],S[3],S[4]],[S[3],S[2],S[5]],[S[4],S[5],-S[1]-S[2]]]);A=np.array([[S[8],S[10],S[11]],[S[10],S[9],S[12]],[S[11],S[12],-S[8]-S[9]]]);gam=S[13:16];bet=S[17:20]
 D=np.array([s,1j*k,0]);lap=D@D;divbet=D@bet;divgam=D@gam;V=np.zeros((20,20),complex)
 V[0]=2*a*c/3*(kh+2*theta)-2*c/3*divbet
 V[6]=-c*lap*lapse+kap*a*theta
 V[7]=a*c/2*divgam+a*lap*dc-2*kap*a*theta
 V[16]=-2*a*kh-ld*lapse
 for (i,j),f in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(1,6)):
  V[f]=-2*a*A[i,j]+D[i]*bet[j]+D[j]*bet[i]-(2*divbet/3 if i==j else 0)
 AA=np.array([[-c*D[i]*D[j]*lapse+a*(-c/2*lap*H[i,j]+c/2*(D[i]*gam[j]+D[j]*gam[i])+.5*D[i]*D[j]*dc) for j in range(3)] for i in range(3)])
 tr=sum(AA[i,i] for i in range(3))
 for (i,j),f in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(8,13)):V[f]=AA[i,j]-(tr/3 if i==j else 0)
 Q=gam-np.einsum('j,ijk->ik',D,H)
 for i in range(3):
  V[13+i]=-4*a/3*D[i]*kh-2*a/3*D[i]*theta+lap*bet[i]+D[i]*divbet/3-2*kap*a*Q[i]
  V[17+i]=G*gam[i]-eta*bet[i]
 V+=b*s*I
 return V,Q

def schur(lam,k,cfg):
 V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 block=lambda V,a,b:V[np.ix_(a,b)]
 C=block(V0,QP,PP);Ci=np.linalg.inv(C)
 Q0=lam*np.eye(10)-block(V0,QP,QP);Q1=-block(V1,QP,QP);P0=lam*np.eye(10)-block(V0,PP,PP);P1=-block(V1,PP,PP)
 M0=P0@Ci@Q0-block(V0,PP,QP);M1=P0@Ci@Q1+P1@Ci@Q0-block(V1,PP,QP);M2=P1@Ci@Q1-block(V2,PP,QP)
 J=np.block([[np.zeros((10,10)),np.eye(10)],[-np.linalg.solve(M2,M0),-np.linalg.solve(M2,M1)]])
 T,Z,sdim=linalg.schur(J,output='complex',sort=lambda z:z.real>1e-9)
 if sdim!=10:raise ValueError(f'stable count {sdim} at {lam},k={k}')
 W=Z[:,:10];T=T[:10,:10];q=W[:10];dq=W[10:];p=Ci@(Q0@q+Q1@dq)
 U=np.zeros((20,10),complex);U[QP]=q;U[PP]=p
 dU=np.zeros((20,10),complex);dU[QP]=dq;dU[PP]=Ci@(Q0@dq+Q1@dq@T)
 residual=np.linalg.norm(J@W-W@T)/max(np.linalg.norm(J@W),1e-300)
 return U,dU,T,residual

def boundary(U,D,cfg):
 left,_=char.scalar_left(cfg.alpha,cfg.chi,2*cfg.alpha,cfg.G,1)
 sc=left[:,:4]@U[[6,7,8,13]]+left[:,4:]@D[[0,1,16,17]]
 rows=list(sc)
 for ai,gi,hi,bi in [(10,14,3,18),(11,15,4,19)]:
  rows.extend([np.sqrt(cfg.G)*U[gi]+D[bi],-2*U[ai]/np.sqrt(cfg.chi)-U[gi]+D[hi]])
 rows.extend([-2*(U[9]+.5*U[8])/np.sqrt(cfg.chi)+D[2]+.5*D[1],-2*U[12]/np.sqrt(cfg.chi)+D[5]])
 return np.stack(rows)

def assess(lam,k,cfg,details=False):
 U,D,T,err=schur(lam,k,cfg);B=boundary(U,D,cfg);Us,R=np.linalg.qr(U);Bi=B@np.linalg.inv(R);norms=np.linalg.norm(Bi,axis=1);Bs=Bi/norms[:,None]
 u,s,vh=np.linalg.svd(Bs)
 out={'sigma_min':float(s[-1]),'sigma_ratio':float(s[-1]/s[0]),'schur_residual':float(err),'stable_trace_condition':float(np.linalg.cond(U))}
 if details:return out,(U,D,T,B,R,vh,norms)
 return out

if __name__=='__main__':
 xyz=np.array([2000.,2000.,-976.]);r=np.linalg.norm(xyz);a=r/(r+1);b=np.sqrt(2)*2000/(r+1)**2
 cases={'flat':Config(),'outer_G1':Config(alpha=a,chi=a*a,beta_n=b),'outer_G2':Config(alpha=a,chi=a*a,beta_n=b,G=2.)}
 rows=[]
 for name,c in cases.items():
  for k in [0.,.003,.01,.025,.05,.1,1.]:
   for lam in np.geomspace(1e-5,2.,45):
    try:q=assess(complex(lam),k,c);rows.append(dict(case=name,k=k,lambda_real=float(lam),**q))
    except ValueError as e:rows.append(dict(case=name,k=k,lambda_real=float(lam),error=str(e)))
  good=[p for p in rows if p['case']==name and 'sigma_min'in p];m=min(good,key=lambda p:p['sigma_min']);print(name,m,flush=True)
 (ROOT/'halfspace-real-scan.json').write_text(json.dumps({'scope':__doc__,'cases':{n:asdict(c)for n,c in cases.items()},'data':rows},indent=2)+'\n')
