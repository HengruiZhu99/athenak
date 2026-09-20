"""Frozen Minkowski, production-G1, full 20-field 2D zero_rate model.

Uniform grid, x/y physical faces, z-independent. Not the variable-background
production domain. All writes remain in this isolated audit directory.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from stencils import operators,active_derivative,char

names=['chi','hxx','hyy','hxy','hxz','hyz','k','theta','Axx','Ayy','Axy','Axz','Ayz','Gx','Gy','Gz','alpha','bx','by','bz']

def build(n=8,h=32.,degree=1,damping=True,corner='combined',bc_order=2,treatment='overwrite',tau=2.,kappa=.1,eta=2.,lapse_damping=.1,stencil='production',return_aux=False,sponge_rate=0.,sponge_cells=4.,sponge_order='before',sponge_sector='all',kappa_taper_cells=0.):
    D,D2,adv,KO,E=operators(n,h,degree,0.,.5,4)
    if stencil=='sbp2':
      D=np.zeros((n,n));D2=np.zeros((n,n));KO*=0
      for j in range(1,n-1):D[j,j-1:j+2]=[-.5,0,.5];D2[j,j-1:j+2]=[1,-2,1]
      D[0,:2]=[-1,1];D[-1,-2:]=[-1,1]
      D2[0,:3]=[1,-2,1];D2[-1,-3:]=[1,-2,1]
      D/=h;D2/=h*h
    N=n*n; dim=20*N; I=np.eye(n); zero=np.zeros((N,N))
    ds=[np.kron(I,D),np.kron(D,I),zero]
    d2=[np.kron(I,D2),np.kron(D2,I),zero]
    db=active_derivative(n,h,bc_order)
    dbs=[np.kron(I,db),np.kron(db,I),zero]
    ko=np.kron(I,KO)+np.kron(KO,I)
    state_eye=np.eye(dim);S=[state_eye[j*N:(j+1)*N] for j in range(20)]
    kap=kappa
    if kappa_taper_cells>0:
      depth=np.minimum.reduce([np.tile(np.arange(n),n),np.tile(n-np.arange(n)-1,n),np.repeat(np.arange(n),n),np.repeat(n-np.arange(n)-1,n)])
      kr=np.clip(depth/(kappa_taper_cells-.5),0,1)
      kap=(kappa*kr**3*(10-15*kr+6*kr*kr))[:,None]
    c,k,t,a=S[0],S[6],S[7],S[16]
    H=[[S[1],S[3],S[4]],[S[3],S[2],S[5]],[S[4],S[5],-S[1]-S[2]]]
    A=[[S[8],S[10],S[11]],[S[10],S[9],S[12]],[S[11],S[12],-S[8]-S[9]]]
    G=S[13:16];B=S[17:20];lap=sum(d2)
    def d(i,u):return ds[i]@u
    def dd(i,j,u):return (d2[i] if i==j else ds[i]@ds[j])@u
    divB=sum(d(i,B[i]) for i in range(3));divG=sum(d(i,G[i]) for i in range(3))
    V=np.zeros((dim,dim))
    def put(j,u):V[j*N:(j+1)*N]=u
    put(0,2/3*(k+2*t)-2/3*divB)
    put(6,-lap@a+(kap*t if damping else 0));put(7,.5*divG+lap@c-(2*kap*t if damping else 0))
    put(16,-2*k-(lapse_damping*a if damping else 0))
    hdot=[[None]*3 for _ in range(3)];adot=[[None]*3 for _ in range(3)]
    for i in range(3):
      for j in range(3):
        hdot[i][j]=-2*A[i][j]+d(i,B[j])+d(j,B[i])-(2/3*divB if i==j else 0)
        adot[i][j]=-dd(i,j,a)-.5*lap@H[i][j]+.5*(d(i,G[j])+d(j,G[i]))+.5*dd(i,j,c)
    tr=sum(adot[i][i] for i in range(3))
    for (i,j),hi,ai in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(1,6),range(8,13)):
      put(hi,hdot[i][j]);put(ai,adot[i][j]-(tr/3 if i==j else 0))
    for i in range(3):
      g=-4/3*d(i,k)-2/3*d(i,t)+lap@B[i]+sum(dd(i,j,B[j]) for j in range(3))/3
      if damping:g-=2*kap*(G[i]-sum(d(j,H[i][j]) for j in range(3)))
      put(13+i,g);put(17+i,G[i]-(eta*B[i] if damping else 0))
    for j in range(20):V[j*N:(j+1)*N]+=ko@S[j]
    # The production face sponge is sampled at cell centers and added by
    # UserRHS before CPBC. A post-CPBC variant is a distinct diagnostic.
    distance=np.minimum.reduce([np.tile(np.arange(n)+.5,n),np.tile(n-np.arange(n)-.5,n),np.repeat(np.arange(n)+.5,n),np.repeat(n-np.arange(n)-.5,n)])
    ramp=np.clip(1-distance/sponge_cells,0,1)
    sigma=sponge_rate*ramp**3*(10-15*ramp+6*ramp*ramp)
    fields=range(20) if sponge_sector=='all' else range(6,16) if sponge_sector=='p' else [7,13,14,15]
    sponge=np.zeros_like(V)
    for f in fields:sponge[f*N:(f+1)*N]=-sigma[:,None]*S[f]
    if sponge_order=='before':V+=sponge
    R=np.zeros_like(V)
    if damping:
      R[6*N:7*N]=kap*t;R[7*N:8*N]=-2*kap*t;R[16*N:17*N]=-lapse_damping*a
      for i in range(3):
        R[(13+i)*N:(14+i)*N]=-2*kap*(G[i]-sum(d(j,H[i][j]) for j in range(3)))
        R[(17+i)*N:(18+i)*N]=-eta*B[i]
    L=V.copy();left,_=char.scalar_left(1.,1.,2.,1.,1);lp=left[:,:4];ld=left[:,4:]
    read=V-R if treatment=='source_preserved' else V if treatment=='overwrite' else np.eye(dim)
    vq=[read[j*N:(j+1)*N] for j in range(20)]
    vh=[[vq[1],vq[3],vq[4]],[vq[3],vq[2],vq[5]],[vq[4],vq[5],-vq[1]-vq[2]]]
    def closure(pos,normal):
      normal=np.array(normal,dtype=float);normal/=np.linalg.norm(normal)
      t1=np.array([0.,0.,1.]);t2=np.cross(normal,t1)
      dn=sum(normal[j]*dbs[j][pos] for j in range(3))
      dh=np.array([[dn@vh[i][j] for j in range(3)] for i in range(3)])
      dbeta=np.array([dn@vq[17+i] for i in range(3)])
      project=lambda T,x,y:np.einsum('i,j,ijm->m',x,y,T)
      dq=np.vstack([dn@vq[0],project(dh,normal,normal),dn@vq[16],normal@dbeta])
      pr=-np.linalg.solve(lp,ld@dq)
      out=np.zeros((10,dim));out[0]=pr[0];out[1]=pr[1]
      gamma=normal[:,None]*pr[3]
      tensor=(normal[:,None]*normal[None,:]-.5*(t1[:,None]*t1[None,:]+t2[:,None]*t2[None,:]))[:,:,None]*pr[2]
      for tangent in [t1,t2]:
        ga=-(tangent@dbeta)
        an=.5*(project(dh,normal,tangent)-ga)
        gamma+=tangent[:,None]*ga
        tensor+=(normal[:,None]*tangent[None,:]+tangent[:,None]*normal[None,:])[:,:,None]*an
      ap=.25*(project(dh,t1,t1)-project(dh,t2,t2));ac=.5*project(dh,t1,t2)
      tensor+=(t1[:,None]*t1[None,:]-t2[:,None]*t2[None,:])[:,:,None]*ap
      tensor+=(t1[:,None]*t2[None,:]+t2[:,None]*t1[None,:])[:,:,None]*ac
      for z,(i,j) in enumerate([(0,0),(1,1),(0,1),(0,2),(1,2)]):out[2+z]=tensor[i,j]
      out[7:10]=gamma
      return out
    for y in range(n):
      for x in range(n):
        sx=-1 if x==0 else 1 if x==n-1 else 0
        sy=-1 if y==0 else 1 if y==n-1 else 0
        if not(sx or sy):continue
        pos=y*n+x
        if corner=='combined' or not(sx and sy):out=closure(pos,[sx,sy,0])
        elif corner in ['face_average','face_sum']:out=.5*(closure(pos,[sx,0,0])+closure(pos,[0,sy,0]))
        elif corner=='first_face':out=closure(pos,[sx,0,0])
        else:raise ValueError(corner)
        for f,j in enumerate([6,7,8,9,10,11,12,13,14,15]):
          if treatment in ['overwrite','source_preserved']:L[j*N+pos]=out[f]+(R[j*N+pos] if treatment=='source_preserved' else 0)
          else:L[j*N+pos]+=(tau/h)*(2 if corner=='face_sum' and sx and sy else 1)*(out[f]-S[j][pos])
    if sponge_order=='after':L+=sponge
    if return_aux:return L,dict(V=V,S=S,H=H,A=A,G=G,B=B,ds=ds,d2=d2,dbs=dbs,ko=ko)
    return L

def analyze(L,n,h):
    vals,vec=eig(L,check_finite=False);idx=np.argmax(vals.real);lam=vals[idx];v=vec[:,idx]
    a=v.reshape(20,n,n).copy();a[[6,7,8,9,10,11,12,13,14,15]]*=h
    en=np.sum(abs(a)**2,axis=0);b=np.zeros((n,n),bool);b[[0,-1],:]=True;b[:,[0,-1]]=True
    corner=np.zeros((n,n),bool);corner[:2,:2]=True;corner[:2,-2:]=True;corner[-2:,:2]=True;corner[-2:,-2:]=True
    return dict(real=float(lam.real),imag=float(lam.imag),positive=int(sum(vals.real>1e-8/h)),
      residual=float(np.linalg.norm(L@v-lam*v)/max(1,np.linalg.norm(L@v))),
      boundary_fraction=float(en[b].sum()/en.sum()),corner_2cell_fraction=float(en[corner].sum()/en.sum()),
      field_fraction={names[j]:float(np.sum(abs(a[j])**2)/en.sum()) for j in range(20)})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=8);p.add_argument('--h',type=float,default=32.)
    p.add_argument('--degree',type=int,default=1);p.add_argument('--corner',default='combined');p.add_argument('--bc-order',type=int,default=2)
    p.add_argument('--treatment',default='overwrite');p.add_argument('--tau',type=float,default=2.)
    p.add_argument('--kappa',type=float,default=.1);p.add_argument('--eta',type=float,default=2.);p.add_argument('--lapse-damping',type=float,default=.1)
    p.add_argument('--stencil',default='production')
    p.add_argument('--no-damping',action='store_true');p.add_argument('--out',required=True);args=p.parse_args()
    params=dict(n=args.n,h=args.h,degree=args.degree,damping=not args.no_damping,corner=args.corner,bc_order=args.bc_order,treatment=args.treatment,tau=args.tau,kappa=args.kappa,eta=args.eta,lapse_damping=args.lapse_damping,stencil=args.stencil)
    ans=dict(parameters=params,**analyze(build(**params),args.n,args.h))
    Path(args.out).write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(ans),flush=True)
