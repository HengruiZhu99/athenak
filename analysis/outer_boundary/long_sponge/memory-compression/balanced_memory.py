"""Balanced truncation of exact full-source RK3 exterior, with closed-loop audit.

External prototype only. Same discrete Fourier symbol, all20 fields. No assumed
energy argument: reduced exterior stability is NOT closed-domain stability.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

MODEL=Path(os.environ.get('ATHENAK_BOUNDARY_DISCRETE_MODEL',
    Path(__file__).resolve().parent.parent/'robust-boundary'/'discrete')).resolve()
sys.path.insert(0,str(MODEL))
from bulk_symbol import symbol
from strip_reference import full_matrix

p=argparse.ArgumentParser()
p.add_argument('--total',type=int,default=32)
p.add_argument('--inside',type=int,default=8)
p.add_argument('--tangent-divisor',type=float,default=8)
p.add_argument('--orders',type=int,nargs='+',default=[16,32,64,128,256,384])
args=p.parse_args();tick=time.monotonic();N=args.total;n=args.inside
h=32.;dt=.6;angle=np.pi/args.tangent_divisor
weights=np.ones(20);weights[6:16]=h
V=np.stack([symbol([a,angle,0],h=h,kappa=.1,eta=2,ldamp=.1) for a in 2*np.pi*np.fft.fftfreq(N)])
K=np.fft.ifft(V*weights[None,:,None]/weights[None,None,:],axis=0)
L=np.block([[K[(i-j)%N] for j in range(N)] for i in range(N)])
X=dt*L;R=np.eye(20*N)+X+X@X/2+X@X@X/6
start=(N-n)//2;inside=np.arange(start*20,(start+n)*20)
outside=np.setdiff1d(np.arange(20*N),inside)
A=R[np.ix_(inside,inside)];B=R[np.ix_(outside,inside)]
C=R[np.ix_(inside,outside)];E=R[np.ix_(outside,outside)]
rho=lambda M:float(abs(la.eigvals(M,check_finite=False)).max())
P=la.solve_discrete_lyapunov(E,B@B.conj().T,method='bilinear')
Q=la.solve_discrete_lyapunov(E.conj().T,C.conj().T@C,method='bilinear')
P=(P+P.conj().T)/2;Q=(Q+Q.conj().T)/2
def root(M):
    w,U=la.eigh(M,check_finite=False)
    return U*np.sqrt(np.maximum(w,0)),float(w.min()),float(w.max())
Ps,pmin,pmax=root(P);Qs,qmin,qmax=root(Q)
U,s,Vh=la.svd(Qs.conj().T@Ps,full_matrices=False,check_finite=False)
out={'scope':__doc__,'parameters':vars(args)|{'h':h,'dt':dt,'angle_y':angle,'kappa':.1,'eta':2,'lapse_damping':.1},
     'full_periodic_rho':rho(R),'exterior_rho':rho(E),
     'gramians':{'reachability_relative_residual':float(la.norm(P-E@P@E.conj().T-B@B.conj().T)/la.norm(B@B.conj().T)),
                  'observability_relative_residual':float(la.norm(Q-E.conj().T@Q@E-C.conj().T@C)/la.norm(C.conj().T@C)),
                  'P_eigen_min':pmin,'P_eigen_max':pmax,'Q_eigen_min':qmin,'Q_eigen_max':qmax},
     'hankel_singular_values':s.tolist(),'runs':[]}
old,_=full_matrix(n=n,h=h,angle_y=angle,degree=1,damping=True,lapse_damping=.1)
wo,vo=la.eig(old,check_finite=False);j=int(np.argmax(wo.real))
uold=(vo[:,j].reshape(20,n)*weights[:,None]).T.reshape(-1);uold/=la.norm(uold)
out['old_zero_rate_leading_eigenvalue']=[float(wo[j].real),float(wo[j].imag)]
u0=np.zeros(20*N,complex);u0[inside]=uold
times=[0,100,1000,5000,10000,50000]
def evolve(M,x):
    powers=[M]
    for _ in range(1,round(max(times)/dt).bit_length()):powers.append(powers[-1]@powers[-1])
    states=[]
    for t in times:
        m=round(t/dt);v=x.copy();i=0
        while m:
            if m&1:v=powers[i]@v
            i+=1;m>>=1
        states.append(v)
    return states
ref=evolve(R,u0);refinside=[v[inside] for v in ref]
out['reference']=[{'time':round(t/dt)*dt,'norm':float(la.norm(v)),'inside_norm':float(la.norm(v[inside]))} for t,v in zip(times,ref)]
for r in args.orders:
    if r>len(E):continue
    if s[r-1]<s[0]*1e-13:
        out['runs'].append({'order':r,'rejected':'Gramian rank inadequate for accurate square-root balancing','last_hankel_value':float(s[r-1])});continue
    Vr=Ps@Vh.conj().T[:,:r]/np.sqrt(s[:r])
    Wr=Qs@U[:,:r]/np.sqrt(s[:r])
    Er=Wr.conj().T@E@Vr;Br=Wr.conj().T@B;Cr=C@Vr
    Rr=np.block([[A,Cr],[Br,Er]])
    closedrho=rho(Rr);ext_rho=rho(Er)
    rec={'order':r,'exterior_rho':ext_rho,'closed_rho':closedrho,'closed_growth':float(np.log(closedrho)/dt),
         'biorthogonality':float(la.norm(Wr.conj().T@Vr-np.eye(r))),
         'formal_Hinf_error_bound_2sum_discarded':float(2*sum(s[r:])),
         'zero_step_max':float(abs(Rr@np.zeros(len(Rr),complex)).max()),'records':[]}
    if closedrho<1.001:
        states=evolve(Rr,np.r_[uold,np.zeros(r,complex)])
        for t,v,true in zip(times,states,refinside):
            physical=Vr@v[len(inside):]
            rec['records'].append({'time':round(t/dt)*dt,'inside_norm':float(la.norm(v[:len(inside)])),
                 'inside_relative_error':float(la.norm(v[:len(inside)]-true)/la.norm(true)),
                 'reconstructed_total_norm':float(np.hypot(la.norm(v[:len(inside)]),la.norm(physical))),
                 'finite':bool(np.isfinite(v).all())})
    out['runs'].append(rec);out['seconds']=time.monotonic()-tick
    path=Path(__file__).with_name(f'balanced-N{N}-I{n}-tangent{args.tangent_divisor:g}.json')
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
print('elapsed',time.monotonic()-tick,flush=True)
