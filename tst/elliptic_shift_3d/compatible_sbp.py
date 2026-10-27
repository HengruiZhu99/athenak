"""Experimental uniform-cell-center SBP derivative pairs and norm-adjoint transfers.

No physical equations, states or constraints are projected. Local closure
coefficients solve polynomial/SBP equations and grid-independent endpoint
quadrature moments. Inspired by generalized SBP definitions in
https://arxiv.org/abs/2305.18195 and norm-compatible interface accuracy in
https://arxiv.org/abs/1806.01931; the constrained construction here is original.
"""
import functools, math
import numpy as np
from numpy.polynomial import legendre as leg


def mm(a,b):
    if a.ndim==2 and b.ndim==2:return np.einsum('ij,jk->ik',a,b,optimize=False)
    if a.ndim==2 and b.ndim==1:return np.einsum('ij,j->i',a,b,optimize=False)
    raise ValueError('matrix product shape')


def basis(x,p):
    V=leg.legvander(2*x-1,p)
    dV=np.column_stack([np.zeros_like(x) if j==0 else 2*leg.legval(2*x-1,leg.legder(np.eye(p+1)[j])) for j in range(p+1)])
    return V,dV


def interpolation(nodes,x):
    w=np.ones(len(nodes))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i!=j:w[i]*=(x-nodes[j])/(nodes[i]-nodes[j])
    return w


@functools.lru_cache(None)
def closure(p):
    if p not in [2,3,4]:raise ValueError('unqualified closure degree')
    r={2:5,3:8,4:11}[p];half=p+1;m=r+half
    x=np.arange(m)+.5
    V=np.column_stack([(x/m)**k for k in range(p+1)])
    dV=np.column_stack([np.zeros(m)]+[k/m*(x/m)**(k-1) for k in range(1,p+1)])
    trace=np.zeros(m);trace[:p+1]=interpolation(x[:p+1],0)
    offsets=np.arange(-half,half+1,dtype=float);rhs=np.zeros(2*half+1);rhs[1]=1
    fd=np.linalg.solve(np.array([offsets**k for k in range(2*half+1)]),rhs)
    fixed=np.zeros((r,m))
    for i in range(r):
        for j in range(r,min(m,i+half+1)):fixed[i,j]=fd[j-i+half]
    edge=-np.outer(trace,trace)/2
    pairs=[(i,j) for i in range(r) for j in range(i+1,r)]
    rows=[];rhs=[]
    for i in range(r):
        for k in range(p+1):
            row=np.zeros(r+len(pairs));row[i]=-dV[i,k]
            for c,(j,l) in enumerate(pairs):
                if i==j:row[r+c]=V[l,k]
                elif i==l:row[r+c]=-V[j,k]
            rows.append(row);rhs.append(-sum((fixed[i]+edge[i])*V[:,k]))
    # Euler-Maclaurin endpoint corrections for midpoint quadrature. Moment m
    # is zero for even m, B_(m+1)(1/2)/(m+1) for odd m. Extra norm degree is
    # needed by two-sided degree-p norm-compatible transfers.
    moments={1:-1/24,3:7/960,5:-31/8064,7:127/30720}
    for k in range(2*p+1):
        row=np.zeros(r+len(pairs));row[:r]=(x[:r]/r)**k
        rows.append(row);rhs.append(sum(row[:r])+moments.get(k,0)/r**k)
    A=np.array(rows);rhs=np.array(rhs)
    initial=np.r_[np.ones(r),np.zeros(len(pairs))]
    value=initial+np.linalg.lstsq(A,rhs-mm(A,initial),rcond=1e-12)[0]
    residual=float(max(abs(mm(A,value)-rhs)))
    if residual>1e-11 or min(value[:r])<=.05:raise ValueError('incompatible or nonpositive local closure')
    S=np.zeros((r,r))
    for k,(i,j) in enumerate(pairs):S[i,j]=value[r+k];S[j,i]=-value[r+k]
    return r,half,value[:r],S,fd,trace[:p+1],residual


@functools.lru_cache(None)
def local(n,p=2):
    r,half,weights,S,fd,t,error=closure(p)
    if n<2*r:raise ValueError('nonoverlapping fixed boundary closures require n >= '+str(2*r))
    H=np.ones(n)/n;H[:r]=weights/n;H[-r:]=weights[::-1]/n
    x=(np.arange(n)+.5)/n;trace=np.zeros((2,n));trace[0,:p+1]=t;trace[1,-p-1:]=t[::-1]
    skew=np.zeros((n,n))
    for i in range(n):
        for j in range(max(0,i-half),min(n,i+half+1)):
            if not (i<r and j<r) and not (i>=n-r and j>=n-r):skew[i,j]=fd[j-i+half]
    skew[:r,:r]=S;skew[-r:,-r:]=-S[::-1,::-1]
    B=np.outer(trace[1],trace[1])-np.outer(trace[0],trace[0]);Q=skew+B/2
    L=np.zeros((n-p-1,n))
    for j in range(len(L)):L[j,j:j+p+2]=[(-1)**k*math.comb(p+1,k) for k in range(p+2)]
    diss=mm(L.T,L)/(2**(2*p+2));V,dV=basis(x,p)
    return dict(x=x,H=H,trace=trace,D=Q/H[:,None],Dplus=(Q+diss)/H[:,None],Dminus=(Q-diss)/H[:,None],R=diss,Q=Q,V=V,dV=dV,degree=p,closure_width=r,closure_residual=error)


def face(parts,p):
    """A full face or its concatenated child intervals, with exact norm moments."""
    x=[];H=[];offset=0
    for width,n in parts:
        block=local(n,p);x.extend(offset+width*block['x']);H.extend(width*block['H']);offset+=width
    if abs(offset-1)>1e-14:raise ValueError('face intervals must cover [0,1]')
    x=np.array(x);H=np.array(H);V,_=basis(x,p)
    return dict(x=x,H=H,V=V,degree=p)


def mortar_global_candidate(left,right):
    """REJECTED global correction: preserved for failure reproduction only.

    Its polynomial identities pass, but its support does not shrink with h
    and smooth refinement/amplification fail. The local-band mortar below
    replaces it in current experiments. Never apply this to evolved fields.
    """
    if left['degree']!=right['degree']:raise ValueError('mortar degree mismatch')
    p=left['degree'];xl=left['x'];xr=right['x'];HL=left['H'];HR=right['H']
    seed=np.zeros((len(xl),len(xr)));width=min(len(xr),2*p+2)
    for i,x in enumerate(xl):
        first=max(0,min(len(xr)-width,int(np.searchsorted(xr,x))-width//2))
        seed[i,first:first+width]=interpolation(xr[first:first+width],x)
    T=np.sqrt(HL)[:,None]*seed/np.sqrt(HR)[None,:]
    scales=np.sqrt(2*np.arange(p+1)+1)
    BL=np.sqrt(HL)[:,None]*left['V']*scales[None,:]
    BR=np.sqrt(HR)[:,None]*right['V']*scales[None,:]
    # T = BL BR^T + (I-BL BL^T) T_seed (I-BR BR^T), evaluated without
    # constructing the large identity/complement matrices.
    TB=mm(T,BR);BT=mm(BL.T,T)
    T=T-mm(TB,BR.T)-mm(BL,BT)+mm(mm(BL,mm(BL.T,TB)),BR.T)+mm(BL,BR.T)
    ILR=T*np.sqrt(HR)[None,:]/np.sqrt(HL)[:,None]
    IRL=ILR.T*HL[None,:]/HR[:,None]
    return ILR,IRL


def mortar(left,right):
    """Simultaneously constrain both norm-adjoint transfers on a local band.

    A global polynomial correction can satisfy moments yet fail refinement.
    These moment equations use coordinates centered on each target point and
    a fixed stencil radius in cell widths, so the support shrinks with h.
    """
    if left['degree']!=right['degree']:raise ValueError('mortar degree mismatch')
    p=left['degree'];xl=left['x'];xr=right['x'];HL=left['H'];HR=right['H']
    h=max(float(np.max(np.diff(xl))),float(np.max(np.diff(xr))))
    radius=(2*p+2 if p==4 else p+2)*h;nl=len(xl);nr=len(xr)
    pairs=[(i,j) for i in range(nl) for j in range(nr) if abs(xl[i]-xr[j])<=radius+1e-14]
    if len(pairs)>6000:raise ValueError('bounded mortar constructor exceeded')
    A=np.zeros(((nl+nr)*(p+1),len(pairs)));b=np.zeros(A.shape[0])
    for i in range(nl):b[i*(p+1)]=np.sqrt(HL[i]/h)
    for j in range(nr):b[(nl+j)*(p+1)]=np.sqrt(HR[j]/h)
    for c,(i,j) in enumerate(pairs):
        d=(xr[j]-xl[i])/radius
        for k in range(p+1):
            A[i*(p+1)+k,c]=np.sqrt(HR[j]/h)*d**k
            A[(nl+j)*(p+1)+k,c]=np.sqrt(HL[i]/h)*(-d)**k
    # Start with local fine-grid prolongation; its norm-adjoint is a natural
    # restriction seed, not point sampling of the fine-grid high frequencies.
    prolong=np.zeros((nr,nl));width=p+1
    for j,x in enumerate(xr):
        first=max(0,min(nl-width,int(np.searchsorted(xl,x))-width//2))
        prolong[j,first:first+width]=interpolation(xl[first:first+width],x)
    initial=np.array([prolong[j,i]*np.sqrt(HR[j]/HL[i]) for i,j in pairs])
    scale=np.sqrt(np.sum(A*A,axis=1));A/=scale[:,None];b/=scale
    value=initial+np.linalg.lstsq(A,b-mm(A,initial),rcond=1e-12)[0]
    residual=np.max(abs(mm(A,value)-b))
    if residual>1e-10:raise ValueError('local norm-compatible mortar constraints failed')
    T=np.zeros((nl,nr))
    for c,(i,j) in enumerate(pairs):T[i,j]=value[c]
    ILR=T*np.sqrt(HR)[None,:]/np.sqrt(HL)[:,None]
    IRL=T.T*np.sqrt(HL)[None,:]/np.sqrt(HR)[:,None]
    return ILR,IRL


def chain(n,p,widths,periodic=False,biased=True):
    loc=local(n,p);N=n*len(widths);H=np.concatenate([h*loc['H'] for h in widths]);x=np.concatenate([sum(widths[:i])+h*loc['x'] for i,h in enumerate(widths)])
    Q=np.zeros((N,N));R=np.zeros_like(Q);trace=np.zeros((2,N));trace[0,:n]=loc['trace'][0];trace[1,-n:]=loc['trace'][1]
    for i in range(len(widths)):
        sl=slice(i*n,(i+1)*n);Q[sl,sl]=loc['Q'];R[sl,sl]=loc['R']
    for i in range(len(widths) if periodic else len(widths)-1):
        j=(i+1)%len(widths);l=np.zeros(N);r=np.zeros(N);l[i*n:(i+1)*n]=loc['trace'][1];r[j*n:(j+1)*n]=loc['trace'][0]
        Q+=(-np.outer(l,l)+np.outer(r,r)+np.outer(l,r)-np.outer(r,l))/2
        jump=l-r;R+=np.outer(jump,jump)/2
    return dict(x=x,H=H,Q=Q,R=R,Dplus=(Q+(R if biased else 0))/H[:,None],Dminus=(Q-(R if biased else 0))/H[:,None],trace=trace)
