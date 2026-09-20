"""Compatible continuum first-order radial reduction with incoming Bjørhus targets.
Auxiliary derivative constraints are explicitly measured, not presumed preserved.
"""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
from scipy import linalg
ROOT=Path(__file__).resolve().parent
REPO=next(p for p in ROOT.parents if (p/'analysis/tde_stability').is_dir())
PREV=REPO/'analysis/tde_stability/evidence/modes/constraint-lower-order'
sys.path.insert(0,str(PREV/'radial-volume'));import radial_operator as volume
sys.path.insert(0,str(PREV/'radial-full-boundary'));from boundary_operator import chebyshev
QIDX=[0,1,6,7];PIDX=[2,3,4,5]

def coeff(r,rate=.1):
    A2,A1,A0=volume.coefficients(np.asarray(r),rate)
    a2,a1,a0=volume.coefficients(np.asarray(r,dtype=complex)+1e-25j,rate)
    A0p=a0.imag/1e-25;A1p=a1.imag/1e-25
    # X=(q4,p4,d4), with q=(u,h,ell,s), p=(k,Theta,A,Gamma), d=q'.
    sh=np.shape(r)+(12,12);P=np.zeros(sh);S=np.zeros(sh)
    for a,i in enumerate(QIDX):
        for b,j in enumerate(QIDX):
            S[...,a,b]=A0[...,i,j];S[...,a,8+b]=A1[...,i,j]
            P[...,8+a,8+b]=A1[...,i,j]
            S[...,8+a,b]=A0p[...,i,j]
            S[...,8+a,8+b]=A1p[...,i,j]+A0[...,i,j]
        for b,j in enumerate(PIDX):
            S[...,a,4+b]=A0[...,i,j]
            P[...,8+a,4+b]=A0[...,i,j]
            S[...,8+a,4+b]=A0p[...,i,j]
    for a,i in enumerate(PIDX):
        for b,j in enumerate(QIDX):
            P[...,4+a,8+b]=A2[...,i,j]
            S[...,4+a,b]=A0[...,i,j];S[...,4+a,8+b]=A1[...,i,j]
        for b,j in enumerate(PIDX):
            P[...,4+a,4+b]=A1[...,i,j];S[...,4+a,4+b]=A0[...,i,j]
    return P,S

def characteristics(r):
    a=r/(1+r);chi=a*a;b=r/(1+r)**2;rows=[];speeds=[];labels=[]
    for sign in [1,-1]:
        lapse=sign*np.sqrt(2*a*chi);shift=sign*np.sqrt(8/3)
        L=np.zeros(8);L[0]=-lapse/chi;L[6]=1
        rows.append(L);speeds.append(b+lapse);labels.append(('lapse',sign))
        sepL=chi*2*a-shift**2;sepC=chi*a*a-shift**2
        L=np.zeros(8);L[0]=a*shift**2*sepC;L[1]=.5*a*(8/3)*sepL;L[3]=.25*shift*(4*chi*a*a-3*shift**2)*sepL
        L[4]=.5*a*a*shift*sepL;L[6]=-chi*a*shift*sepC;L[7]=sepL*sepC
        rows.append(L);speeds.append(b+shift);labels.append(('shift',sign))
        L=np.zeros(8);L[1]=sign*a;L[3]=chi/2;L[4]=1
        rows.append(L);speeds.append(b+sign*a*a);labels.append(('C1',sign))
        L=np.zeros(8);L[0]=sign*4/(3*a);L[1]=sign*2/(3*a);L[2]=-sign*2/a;L[3]=-1;L[5]=1
        rows.append(L);speeds.append(b+sign*a*a);labels.append(('C2',sign))
    return np.array(rows),np.array(speeds),labels

def operators(r,D):
    n=len(r);R=r+1;a=r/R;chi=a*a;K=1/R**2;cp=2*r/R**3;cpp=2*(1-2*r)/R**4
    I=np.eye(n);Z=np.zeros_like(I);h=[Z.copy() for _ in range(12)];m=[z.copy() for z in h];q=[z.copy() for z in h];th=[z.copy() for z in h]
    h[0]=np.diag(2.5*cp**2/chi**2)
    h[1]=np.diag(3*chi/r**2-2*cpp-4*cp/r+2.5*cp**2/chi+8*K*K/3)
    h[4]=np.diag(4*K/3);h[5]=np.diag(8*K/3);h[6]=np.diag(4*K)
    h[8]=2*D+np.diag(4/r-5*cp/chi);h[9]=np.diag(chi)@D+np.diag(5*chi/r-2*cp)
    m[0]=-np.diag(2*K*cp/chi**2);m[1]=np.diag(2*K/(3*R));m[4]=-2*D/3;m[5]=-4*D/3;m[6]=D+np.diag(3/R);m[8]=np.diag(2*K/chi);m[9]=np.diag(5*K/3)
    q[7]=I.copy();q[9]=-I;q[1]=-np.diag(3/r);th[5]=I.copy()
    C={key:np.hstack(z) for key,z in [('H',h),('M',m),('Q',q),('Theta',th)]}
    E=np.zeros((4*n,12*n))
    for j in range(4):E[j*n:(j+1)*n,j*n:(j+1)*n]=-D;E[j*n:(j+1)*n,(8+j)*n:(9+j)*n]=I
    C['reduction']=E
    return C

def build(degree,inner=.2,outer=4,rate=.1,falloff='areal',reduction_damping=0,boundary='physical_rate'):
    r,D=chebyshev(degree,inner,outer);n=len(r);P,S=coeff(r,rate)
    raw=np.block([[P[:,i,j,None]*D+np.diag(S[:,i,j]) for j in range(12)] for i in range(12)])
    C=operators(r,D);E=C['reduction']
    # A pure auxiliary constraint addition; identically zero on d=q'.
    for j in range(4):raw[(8+j)*n:(9+j)*n]-=reduction_damping*E[j*n:(j+1)*n]
    a=r/(1+r);chi=a*a;b=r/(1+r)**2;v=chi-b;sigma=rate*a
    wT={'none':0*r,'coordinate':1/r,'areal':1/(1+r)}[falloff];wQ=0*r if falloff=='none' else 1/r
    FT=np.diag(a/2)@C['H']+np.diag(chi)@D@C['Theta']+np.diag(a*chi/2)@D@C['Q']+np.diag(a*chi/r)@C['Q']+np.diag(v*wT-2*sigma)@C['Theta']
    FQ=np.diag(2*a)@C['M']+np.diag(2*a)@D@C['Theta']+np.diag(chi)@D@C['Q']+np.diag(v*wQ-2*sigma)@C['Q']
    out=raw.copy();info=[]
    for index,incoming in [(0,[0,1,2,3]),(n-1,[5])]:
        L,speeds,labels=characteristics(r[index]);Ri=np.linalg.inv(L)[:,incoming]
        wave_indices=np.arange(4,12)*n+index;wave_raw=raw[wave_indices]
        delta=[]
        for j in incoming:
            label,sign=labels[j]
            if label in ['lapse','shift']:delta.append(-L[j]@wave_raw)
            elif label=='C1':delta.append(-speeds[j]*FT[index]/a[index])
            elif label=='C2':delta.append(speeds[j]*FQ[index]/chi[index])
        if boundary=='physical_rate' and index==0:
            # Target the ACTUAL derivative of both physical constraints, using
            # full incoming eigenvectors. Gauge eigenvectors have zero Theta/Q.
            radiation=np.array([(C['Theta']@raw+np.diag(v)@D@C['Theta']+np.diag(v*wT)@C['Theta'])[index],(C['Q']@raw+np.diag(v)@D@C['Q']+np.diag(v*wQ)@C['Q'])[index]])
            M=np.array([Ri[1],Ri[3]-Ri[5]])
            target=np.array(delta);target[2:]=np.linalg.solve(M[:,2:],-radiation-M[:,:2]@target[:2]);delta=target
        correction=Ri@np.array(delta);out[wave_indices]+=correction
        info.append({'endpoint':index,'incoming_labels':[labels[j] for j in incoming],'principal_left_error':float(np.max(abs(L@P[index,4:,4:]-speeds[:,None]*L))),'outgoing_correction_error':float(np.max(abs(L[[j for j in range(8) if j not in incoming]]@correction)))})
    return out,raw,r,D,C,info

def spectrum(degree,**kwargs):
    t0=time.monotonic();A,raw,r,D,C,info=build(degree,**kwargs);vals,vec=linalg.eig(A,check_finite=False)
    idx=np.argsort(vals.real)[::-1];n=len(r)
    def mode(j,profile=False):
        lam=vals[j];z=vec[:,j];q=C['Q']@z;th=C['Theta']@z;re=C['reduction']@z
        a=r/(1+r);v=a*a-r/(1+r)**2;falloff=kwargs.get('falloff','areal');wt={'none':0,'coordinate':1/r[0],'areal':1/(1+r[0])}[falloff];wq=0 if falloff=='none' else 1/r[0]
        bcs=np.array([lam*th[0]+v[0]*((D@th)[0]+wt*th[0]),lam*q[0]+v[0]*((D@q)[0]+wq*q[0])])
        ref=np.array([abs(lam*th[0])+abs(v[0]*(D@th)[0])+abs(v[0]*wt*th[0]),abs(lam*q[0])+abs(v[0]*(D@q)[0])+abs(v[0]*wq*q[0])])
        o={'lambda_real':float(lam.real),'lambda_imag':float(lam.imag),'reduction_over_state':float(np.linalg.norm(re)/np.linalg.norm(z)),'dynamic_radiation_relative':(abs(bcs)/np.maximum(ref,1e-300)).tolist(),'eigen_residual':float(np.linalg.norm(A@z-lam*z)/(np.linalg.norm(A@z)+abs(lam)*np.linalg.norm(z)))}
        if profile:o['state_real']=z.reshape(12,n).real.tolist();o['state_imag']=z.reshape(12,n).imag.tolist()
        return o
    near=np.argsort(abs(vals+.072790693446))[:3]
    out={'degree':degree,'options':kwargs,'boundary_audit':info,'max_real':float(vals.real.max()),'positive_count_1e-8':int(np.sum(vals.real>1e-8)),'fastest':[mode(j) for j in idx[:10]],'near_constraint_reference':[mode(j,True) for j in near],'wall_seconds':time.monotonic()-t0}
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--degree',type=int,default=32);p.add_argument('--reduction-damping',type=float,default=0);p.add_argument('--falloff',default='areal');p.add_argument('--boundary',default='physical_rate',choices=['physical_rate','derivative_target']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=spectrum(a.degree,reduction_damping=a.reduction_damping,falloff=a.falloff,boundary=a.boundary);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='near_constraint_reference'},indent=2))
