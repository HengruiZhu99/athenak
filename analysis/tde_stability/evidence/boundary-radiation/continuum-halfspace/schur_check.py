from pathlib import Path
import sys,json
import numpy as np
from scipy import linalg,optimize
from determinant import *

def schur(lam,ky,damping=True):
 V0,_=volume(0,ky,damping=damping);Vp,_=volume(1,ky,damping=damping);Vm,_=volume(-1,ky,damping=damping);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 block=lambda V,a,b:V[np.ix_(a,b)]
 C=block(V0,QP,PP);Ci=np.linalg.inv(C)
 Q0=lam*np.eye(10)-block(V0,QP,QP);Q1=-block(V1,QP,QP);P0=lam*np.eye(10)-block(V0,PP,PP);P1=-block(V1,PP,PP)
 M0=P0@Ci@Q0-block(V0,PP,QP);M1=P0@Ci@Q1+P1@Ci@Q0-block(V1,PP,QP);M2=P1@Ci@Q1-block(V2,PP,QP)
 J=np.block([[np.zeros((10,10)),np.eye(10)],[-np.linalg.solve(M2,M0),-np.linalg.solve(M2,M1)]])
 T,Z,sdim=linalg.schur(J,output='complex',sort=lambda z:z.real>1e-7)
 assert sdim==10,(sdim,lam,ky)
 W=Z[:,:10];T=T[:10,:10];q=W[:10];dq=W[10:];p=Ci@(Q0@q+Q1@dq)
 U=np.zeros((20,10),complex);U[QP]=q;U[PP]=p
 dU=np.zeros((20,10),complex);dU[QP]=dq;dU[PP]=Ci@(Q0@dq+Q1@dq@T)
 residual=np.linalg.norm(J@W-W@T)/np.linalg.norm(J@W)
 return U,dU,T,residual,np.linalg.cond(M2)

def assess_schur(lam,ky,mode='radiation',damping=True,details=False):
 U,D,T,err,mcond=schur(lam,ky,damping)
 left,_=char.scalar_left(alpha,chi,2*alpha,2,1)
 sc=left[:,:4]@U[[6,7,8,13]]+left[:,4:]@D[[0,1,16,17]]
 c=alpha*np.sqrt(chi);H=[[U[1],U[3],U[4]],[U[3],U[2],U[5]],[U[4],U[5],-U[1]-U[2]]];DH=[[D[1],D[3],D[4]],[D[3],D[2],D[5]],[D[4],D[5],-D[1]-D[2]]]
 Q=np.stack([U[13+i]-DH[i][0]-1j*ky*H[i][1] for i in range(3)])
 dQ=Q@T
 FQ=lam*Q+(c-bn)*dQ;Ft=lam*U[7]+(c-bn)*D[7]
 rows=[U[16],U[17]] if mode=='radiation_dirichlet' else [sc[0],sc[1]]
 rows.extend([Ft,FQ[0]] if mode.startswith('radiation') else [sc[2],sc[3]])
 for ai,gi,hi,bi in [(10,14,3,18),(11,15,4,19)]:rows.extend([U[bi] if mode=='radiation_dirichlet' else np.sqrt(2)*U[gi]+D[bi],FQ[gi-13] if mode.startswith('radiation') else -2*U[ai]/np.sqrt(chi)-U[gi]+D[hi]])
 if mode=='radiation_weyl':
  rows.extend([lam*(U[9]+.5*U[8])+(c-bn)*(D[9]+.5*D[8])-.5*c*1j*ky*U[10]-.5*chi*ky*ky*U[16],lam*U[12]+(c-bn)*D[12]-.5*c*1j*ky*U[11]])
 else:
  rows.extend([-2*(U[9]+.5*U[8])/np.sqrt(chi)+D[2]+.5*D[1],-2*U[12]/np.sqrt(chi)+D[5]])
 B=np.stack(rows);Us,R=np.linalg.qr(U);Bi=B@np.linalg.inv(R);norms=np.linalg.norm(Bi,axis=1);Bs=Bi/norms[:,None]
 uu,ss,vv=np.linalg.svd(Bs)
 result={'sigma_min':float(ss[-1]),'sigma_ratio':float(ss[-1]/ss[0]),'schur_residual':float(err),'state_basis_condition':float(np.linalg.cond(U)),'quadratic_leading_condition':float(mcond),'normal_roots':[[z.real,z.imag] for z in np.linalg.eigvals(T)]}
 if details:return result,(U,D,T,B,R,ss,vv,Us,norms)
 return result
if __name__=='__main__':
 out=[]
 for k in [np.pi,2*np.pi,4*np.pi]:
  for damping in [False,True]:
   for mode in ['zero_rate','radiation']:
    for x in [.01,.1,.5,1.,1.2,1.5,2.,4.,8.]:
     a=assess_schur(complex(x),k,mode,damping);out.append(dict(k=k,lam=[x,0],mode=mode,damping=damping,**a))
 (ROOT/'schur-real-scan.json').write_text(json.dumps(out,indent=2)+'\n')
 roots=[]
 for k in [np.pi,2*np.pi,4*np.pi]:
  for damping in [False,True]:
   fn=lambda z:assess_schur(complex(z),k,'radiation',damping)['sigma_min']
   fit=optimize.minimize_scalar(fn,bounds=(.02*k,.45*k),method='bounded',options={'xatol':1e-12})
   a=assess_schur(complex(fit.x),k,'radiation',damping);roots.append(dict(k=k,damping=damping,lambda_real=float(fit.x),minimum=float(fit.fun),**a));print(roots[-1],flush=True)
 (ROOT/'schur-real-candidates.json').write_text(json.dumps(roots,indent=2)+'\n')
