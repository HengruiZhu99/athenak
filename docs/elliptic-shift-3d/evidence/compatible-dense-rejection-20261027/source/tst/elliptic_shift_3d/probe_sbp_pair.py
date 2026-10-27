#!/usr/bin/env python3
"""Cell-centered compatible derivative/mortar construction; standalone only.

Original constrained algebra, guided by generalized SBP definitions in
Lundquist/Winters/Nordstrom, arXiv:2305.18195. No evolution projections.
The fourth-order target is tested separately from lower-degree controls.
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg

def mm(a,b):
    # Avoid platform BLAS matmul floating-status warnings; no warning suppression.
    if a.ndim==2 and b.ndim==2:return np.einsum('ij,jk->ik',a,b,optimize=False)
    if a.ndim==2 and b.ndim==1:return np.einsum('ij,j->i',a,b,optimize=False)
    raise ValueError('unexpected matrix-product shape')


def basis(x, degree):
    V=leg.legvander(2*x-1,degree)
    dV=np.column_stack([np.zeros_like(x) if j==0 else 2*leg.legval(2*x-1,leg.legder(np.eye(degree+1)[j])) for j in range(degree+1)])
    return V,dV


def interpolation(nodes,query):
    out=np.ones(len(nodes))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i!=j:out[i]*=(query-nodes[j])/(nodes[i]-nodes[j])
    return out


def local(n,degree=3):
    if n<degree+2:raise ValueError('insufficient active nodes')
    x=(np.arange(n)+.5)/n
    # Degree 2p moments permit equally accurate norm-adjoint face transfers.
    # Legendre scaling avoids ill-conditioned monomial moment solves.
    moments=leg.legvander(2*x-1,2*degree).T
    target=np.zeros(2*degree+1);target[0]=1
    H=np.full(n,1/n)+np.linalg.lstsq(moments,target-mm(moments,np.full(n,1/n)),rcond=None)[0]
    if np.max(abs(mm(moments,H)-target))>2e-12 or min(H)<=0:
        raise ValueError('positive degree-2p quadrature unavailable on these nodes')
    trace=np.zeros((2,n))
    trace[0,:degree+1]=interpolation(x[:degree+1],0)
    trace[1,-degree-1:]=interpolation(x[-degree-1:],1)
    B=np.outer(trace[1],trace[1])-np.outer(trace[0],trace[0])
    # Start from an ordinary local degree-four derivative; enforce polynomial
    # reproduction and the SBP identity simultaneously, not by symmetrization alone.
    D0=np.zeros((n,n));width=min(n,max(5,degree+1))
    for i in range(n):
        first=max(0,min(n-width,i-width//2));ids=np.arange(first,first+width)
        offsets=(x[ids]-x[i])*n;A=np.array([offsets**k for k in range(width)])
        b=np.zeros(width);b[1]=n;D0[i,ids]=np.linalg.solve(A,b)
    V,dV=basis(x,degree);U,R=np.linalg.qr(V,mode='reduced');dU=np.linalg.solve(R.T,dV.T).T
    S0=(H[:,None]*D0-D0.T*H[None,:])/2
    F=H[:,None]*dU-mm(B,U)/2-mm(S0,U)
    compatibility=np.max(abs(mm(U.T,F)+mm(F.T,U)))
    if compatibility>2e-12:raise ValueError('incompatible SBP moments')
    S=S0+mm(F,U.T)-mm(U,F.T)-mm(mm(U,mm(U.T,F)),U.T)
    Q=(S-S.T)/2+B/2;D=Q/H[:,None]
    # Positive high-difference form annihilates every retained polynomial.
    L=np.zeros((n-degree-1,n))
    for j in range(len(L)):
        L[j,j:j+degree+2]=[(-1)**k*math.comb(degree+1,k) for k in range(degree+2)]
    Rplus=(mm(L.T,L))/(2**(2*degree+2))
    Dplus=(Q+Rplus)/H[:,None];Dminus=(Q-Rplus)/H[:,None]
    return dict(x=x,H=H,trace=trace,D=D,Dplus=Dplus,Dminus=Dminus,R=Rplus,V=V,dV=dV,
                compatibility=compatibility)


def mortar(left,right):
    # Transfer right values to the left face and its exact norm-adjoint reverse.
    # Both retain degree p iff the two degree-2p Gram moments agree.
    VL=left['V'];VR=right['V'];HL=left['H'];HR=right['H']
    UL,RL=np.linalg.qr(np.sqrt(HL)[:,None]*VL,mode='reduced')
    UR,RR=np.linalg.qr(np.sqrt(HR)[:,None]*VR,mode='reduced')
    # Polynomial transfer from right to left; complement defaults to zero only
    # for interpolation construction, never for an evolved field or gauge residual.
    ILR=mm(VL,np.linalg.solve(RR,UR.T*np.sqrt(HR)[None,:]))
    IRL=(ILR.T*HL[None,:])/HR[:,None]
    return ILR,IRL


def chain(n,degree,widths,periodic=False,biased=True):
    loc=local(n,degree);count=n*len(widths);H=np.concatenate([h*loc['H'] for h in widths]);x=np.concatenate([sum(widths[:i])+h*loc['x'] for i,h in enumerate(widths)])
    Q=np.zeros((count,count));R=np.zeros_like(Q);boundary=np.zeros((2,count));boundary[0,:n]=loc['trace'][0];boundary[1,-n:]=loc['trace'][1]
    for i,h in enumerate(widths):
        sl=slice(i*n,(i+1)*n);Q[sl,sl]=loc['H'][:,None]*loc['D'];R[sl,sl]=loc['R']
    for i in range(len(widths) if periodic else len(widths)-1):
        j=(i+1)%len(widths);l=np.zeros(count);r=np.zeros(count);l[i*n:(i+1)*n]=loc['trace'][1];r[j*n:(j+1)*n]=loc['trace'][0]
        # Central interface flux cancels the two diagonal face terms exactly.
        Q+=(-np.outer(l,l)+np.outer(r,r)+np.outer(l,r)-np.outer(r,l))/2
        jump=l-r;R+=np.outer(jump,jump)/2
    return dict(x=x,H=H,Q=Q,R=R,Dplus=(Q+(R if biased else 0))/H[:,None],Dminus=(Q-(R if biased else 0))/H[:,None],trace=boundary)


def evaluate():
    result={'scope':'Component construction only; no 3D gauge/evolution adoption','local':[],'mortars':[],'chains':[],'periodic':[]}
    for degree in [2,3,4]:
        for n in [6,8,10,12,16,24,32,48]:
            try:a=local(n,degree)
            except ValueError as error:
                result['local'].append(dict(n=n,degree=degree,status='unsupported',reason=str(error)));continue
            H=a['H'];B=np.outer(a['trace'][1],a['trace'][1])-np.outer(a['trace'][0],a['trace'][0])
            sbp=H[:,None]*a['Dplus']+a['Dminus'].T*H[None,:]-B
            x=a['x'];exact=2*np.pi*np.cos(2*np.pi*x);error=mm(a['Dminus'],np.sin(2*np.pi*x))-exact
            result['local'].append(dict(n=n,degree=degree,status='constructed',minimum_weight=float(min(H)),sbp_max=float(np.max(abs(sbp))),polynomial_max=float(np.max(abs(mm(a['Dminus'],a['V'])-a['dV']))),smooth_rms=float(np.sqrt(sum(H*error**2))),smooth_max=float(max(abs(error)))))
        for n in [8,10,12]:
            try:l=local(n,degree);r=local(2*n,degree)
            except ValueError:continue
            ILR,IRL=mortar(l,r)
            result['mortars'].append(dict(n=n,degree=degree,adjoint_max=float(np.max(abs(l['H'][:,None]*ILR-IRL.T*r['H'][None,:]))),forward_polynomial_max=float(np.max(abs(mm(ILR,r['V'])-l['V']))),reverse_polynomial_max=float(np.max(abs(mm(IRL,l['V'])-r['V']))),forward_L1=float(np.max(np.sum(abs(ILR),axis=1))),reverse_L1=float(np.max(np.sum(abs(IRL),axis=1)))))
    for degree in [3,4]:
        for n in [12,16,24,32]:
            # Nonuniform radial widths include the sharp shell-spacing transitions.
            a=chain(n,degree,[.25,.20,.10,.05,.05,.05,.05,.25]);x=a['x'];H=a['H'];B=np.outer(a['trace'][1],a['trace'][1])-np.outer(a['trace'][0],a['trace'][0])
            V=np.column_stack([x**j for j in range(degree+1)]);dV=np.column_stack([np.zeros_like(x)]+[j*x**(j-1) for j in range(1,degree+1)])
            err=mm(a['Dminus'],np.sin(2*np.pi*x))-2*np.pi*np.cos(2*np.pi*x)
            result['chains'].append(dict(n=n,degree=degree,sbp_max=float(np.max(abs(H[:,None]*a['Dplus']+a['Dminus'].T*H[None,:]-B))),polynomial_max=float(np.max(abs(mm(a['Dminus'],V)-dV))),smooth_rms=float(np.sqrt(sum(H*err**2))),smooth_max=float(max(abs(err)))))
        for n in [12,16,24]:
            a=chain(n,degree,[.25]*4,periodic=True);H=a['H'];G=a['Dminus'];coefficient=1+.3*np.sin(2*np.pi*a['x'])
            A=-(mm(a['Dplus']*coefficient[None,:],G));weighted=np.sqrt(H)[:,None]*A/np.sqrt(H)[None,:]
            eig=np.linalg.eigvalsh((weighted+weighted.T)/2)
            checker=(-1.)**np.arange(len(H));cg=mm(G,checker)
            q=np.random.default_rng(1927).normal(size=len(H));lhs=sum(H*q*(mm(A,q)));rhs=sum(H*coefficient*(mm(G,q))**2)
            result['periodic'].append(dict(n=n,degree=degree,variable_coefficient_energy_relative_error=float(abs(lhs-rhs)/max(1,abs(rhs))),symmetry_max=float(np.max(abs(weighted-weighted.T))),minimum_eigenvalue=float(eig[0]),first_positive_eigenvalue=float(eig[1]),largest_eigenvalue=float(eig[-1]),near_null_count=int(sum(abs(eig)<1e-8)),checker_derivative_rms=float(np.sqrt(sum(H*cg**2)))))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    np.seterr(all='raise');start=time.monotonic();result=evaluate();result['seconds']=time.monotonic()-start;result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for key in ['chains','periodic']:print(key,json.dumps(result[key],indent=2))
    # Each criterion covers its stated component only; evolution remains untested.
    ok=all(r['sbp_max']<1e-10 and r['polynomial_max']<1e-8 for r in result['local'] if r['status']=='constructed')
    ok &= all(max(r['adjoint_max'],r['forward_polynomial_max'],r['reverse_polynomial_max'])<1e-10 for r in result['mortars'])
    ok &= all(r['near_null_count']==1 and r['minimum_eigenvalue']>-1e-8 and r['variable_coefficient_energy_relative_error']<1e-12 for r in result['periodic'])
    # The archived v2 probe returned success for algebraic identities alone.
    # That was not a consistency result: its dense closure fails refinement.
    # Keep this rejected construction reproducible, but reject it explicitly.
    orders=[]
    for degree in [2,3,4]:
        family=[r for r in result['local'] if r['degree']==degree and r['status']=='constructed']
        a,b=family[-2:]
        orders.append(float(np.log(a['smooth_rms']/b['smooth_rms'])/np.log(b['n']/a['n'])))
    result['smooth_orders']=orders
    result['gates']={'algebraic_identities':bool(ok),'smooth_refinement':min(orders)>1}
    result['disposition']='Rejected dense closure; do not adopt even if algebraic identities pass.'
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    ok &= result['gates']['smooth_refinement']
    raise SystemExit(0 if ok else 1)
