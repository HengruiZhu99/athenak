#!/usr/bin/env python3
import json,math,hashlib,sys
from pathlib import Path
import numpy as np
np.seterr(all='raise')
base=Path(sys.argv[1]);sbp=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(exist_ok=True)
s=json.loads(sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
def mm(a,b):return np.einsum('ij,jk->ik',a,b)
def spatial(n):
 m=len(weights);h=np.ones(n);h[:m]=weights;h[-m:]=weights[::-1];d=np.zeros((n,n));width=rows.shape[1];d[:m,:width]=rows;d[-m:,-width:]=-rows[::-1,::-1]
 for i in range(m,n-m):
  for j,c in [(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)]:d[i,i+j]=c
 tl=np.zeros(n);tl[:m]=trace;tr=tl[::-1];assert np.linalg.norm(h[:,None]*d+d.T*h[None,:]-(np.outer(tr,tr)-np.outer(tl,tl)))<1e-10
 return h/n,d*n,tl,tr
records=[];stiff=[]
for family in ['synthetic','extended','massive']:
 mats=[json.loads(x) for x in (base/(family+'-matrices.jsonl')).read_text().splitlines()];jets=[json.loads(x) for x in (base/(family+'-jets.jsonl')).read_text().splitlines()]
 chosen=set()
 for sample,(mat,jet) in enumerate(zip(mats,jets)):
  nvec=np.array(jet['coordinate']);nvec/=np.linalg.norm(nvec);G=np.array(mat['G']).reshape(3,3);beta=float(np.dot(mat['beta'],nvec));c=mat['N']*math.sqrt(float(np.einsum('i,ij,j->',nvec,G,nvec)));lo=-beta-c;hi=-beta+c;B=np.array(mat['source_matrix_4x20']).reshape(4,20);Bu=B[:,:4];Bv=B[:,4:8];Bq=np.einsum('kij,j->ki',B[:,8:].reshape(4,4,3),nvec)
  for k in [0,1,8,32]:
   F=np.block([[1j*beta*k*np.eye(4),np.eye(4)], [Bu+1j*k*Bq-c*c*k*k*np.eye(4),Bv+1j*beta*k*np.eye(4)]])
   eig=np.linalg.eigvals(F);dtwave=.1/(48*max(abs(lo),abs(hi)));dtsource=min(dtwave,.01/max(1.,max(abs(eig))))
   def rkerr(dt):
    z=eig*dt;rk=1+z+z*z/2+z*z*z/6+z*z*z*z/24;exact=np.exp(z);return float(max(abs(rk-exact)/np.maximum(1.,abs(exact))))
   stiff.append(dict(family=family,sample=sample,radius=jet['radius'],k=k,max_real=float(max(eig.real)),max_abs=float(max(abs(eig))),wave_dt=dtwave,source_dt=dtsource,wave_RK4_error=rkerr(dtwave),source_RK4_error=rkerr(dtsource)))
  radius=jet['radius']
  if radius not in [.9,.98,.99] or radius in chosen:continue
  chosen.add(radius)
  for n in [24,48]:
   h,D,tl,tr=spatial(n);H=np.diag(h);I=np.eye(n);Z=np.zeros((n,n));E=np.vstack([tl,tr]);faceB=np.diag([-1.,1.]);Q=h[:,None]*D;QN=.5*np.block([[Q-Q.T,mm(E.T,faceB)],[-mm(faceB,E),faceB]]);P=np.vstack([I,E]);x=(np.arange(n)+.5)/n
   for t in [0.,.37]:
    def coeff(xx):
     hp=hi*(1+.04*(xx-.5)+.02*np.sin(t));lp=lo*(1+.03*(xx-.5)-.01*np.sin(t));ht=.02*hi*np.cos(t)+np.zeros_like(xx);lt=-.01*lo*np.cos(t)+np.zeros_like(xx);return -(hp+lp)/2,(hp-lp)/2,-hp*lp,-(ht+lt)/2,-(lt*hp+lp*ht),hp,lp
    b,cc,a,bt,at,hp,lp=coeff(x);bf,cf,af,_,_,hf,lf=coeff(np.array([0.,1.]));bx=np.einsum('ij,j->i',D,b);ax=np.einsum('ij,j->i',D,a);lower=bt-b*bx-ax;bn=np.diag(np.r_[b,bf]);shift=mm(mm(P.T,mm(QN,bn)+mm(bn,QN)),P)/h[:,None]-np.diag(bx);potential=mm(D.T,H*a[None,:]);potential=mm(potential,D);weak=(-potential+mm(mm(mm(E.T,faceB),np.diag(af)),mm(E,D)))/h[:,None]
    A=weak+lower[:,None]*D;K=shift.copy();A-=hf[0]*np.outer(tl/h,lf[0]*np.einsum('i,ij->j',tl,D));A-=-lf[1]*np.outer(tr/h,hf[1]*np.einsum('i,ij->j',tr,D));K-=hf[0]*np.outer(tl/h,tl)-lf[1]*np.outer(tr/h,tr)
    srcq=np.kron(I,Bu)+np.kron(D,Bq)-np.kron(b[:,None]*D,Bv);srcw=np.kron(I,Bv)
    L=np.block([[np.zeros((4*n,4*n)),np.eye(4*n)],[np.kron(A,np.eye(4))+srcq,np.kron(K,np.eye(4))+srcw]])
    W=np.block([[np.kron(potential+H,np.eye(4)),np.zeros((4*n,4*n))],[np.zeros((4*n,4*n)),np.kron(H,np.eye(4))]])
    Wdot=np.zeros_like(W);Wdot[:4*n,:4*n]=np.kron(mm(mm(D.T,H*at[None,:]),D),np.eye(4))
    cross=mm(D.T,H*lower[None,:])+H;boundary=-2*(cf[0]*np.outer(tl,tl)+cf[1]*np.outer(tr,tr));expected=np.block([[Wdot[:4*n,:4*n],np.kron(cross,np.eye(4))],[np.kron(cross.T,np.eye(4)),np.kron(boundary-2*H*bx[None,:],np.eye(4))]])
    hs=np.kron(H,np.eye(4));sq=mm(hs,srcq);sw=mm(hs,srcw);expected[:4*n,4*n:]+=sq.T;expected[4*n:,:4*n]+=sq;expected[4*n:,4*n:]+=sw+sw.T
    actual=mm(W,L)+mm(L.T,W)+Wdot;defect=float(np.linalg.norm(actual-expected)/max(1.,np.linalg.norm(expected)))
    # Pointwise source work independently agrees with its assembled blocks.
    U=np.sin(np.arange(4*n)*.17).reshape(n,4);w=np.cos(np.arange(4*n)*.11).reshape(n,4);q=np.einsum('ij,jk->ik',D,U);V=w-b[:,None]*q;source=np.einsum('ij,nj->ni',Bu,U)+np.einsum('ij,nj->ni',Bv,V)+np.einsum('ij,nj->ni',Bq,q);assembled=(np.einsum('ij,j->i',srcq,U.ravel())+np.einsum('ij,j->i',srcw,w.ravel())).reshape(n,4);serr=float(np.linalg.norm(source-assembled)/max(1.,np.linalg.norm(source)))
    naive=expected.copy();naive[:4*n,4*n:]-=sq.T;naive[4*n:,:4*n]-=sq;naive[4*n:,4*n:]-=sw+sw.T;negative=float(np.linalg.norm(actual-naive)/max(1.,np.linalg.norm(expected)))
    records.append(dict(family=family,sample=sample,radius=radius,nodes=n,time=t,energy_defect=defect,reconstruction_error=serr,negative_omitted_source=negative,passed=bool(defect<=1e-10 and serr<=1e-12)))
r=dict(passed=all(x['passed'] for x in records) and max(x['negative_omitted_source'] for x in records)>1e-7 and max(x['source_RK4_error'] for x in stiff)<=1e-5,records=records,stiffness=stiff,scope='Actual frozen lower-source matrices with synthetic variable principal speeds; no time evolution/reflection/native background admission',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'review.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(dict(passed=r['passed'],energy_tests=len(records),max_energy_defect=max(x['energy_defect'] for x in records),max_reconstruction=max(x['reconstruction_error'] for x in records),max_local_source_growth=max(x['max_real'] for x in stiff),max_wave_RK_error=max(x['wave_RK4_error'] for x in stiff),max_source_RK_error=max(x['source_RK4_error'] for x in stiff)))
