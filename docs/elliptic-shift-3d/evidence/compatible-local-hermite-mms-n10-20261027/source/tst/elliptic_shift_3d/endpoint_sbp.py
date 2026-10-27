"""Uniform endpoint-collocated SBP pair with fixed local closures.

An independent alternative to global Lobatto differentiation. Polynomial
degree two at the boundary and sixth-order centered interior; positive local
high-difference form controls centered high-frequency modes. Endpoint norm
moments are those of corrected trapezoidal quadrature, not midpoint weights.
No field, constraint, source or boundary radius is modified.
"""
import functools,math
import numpy as np
from compatible_sbp import basis,mm


@functools.lru_cache(None)
def closure():
    p=2;r=5;half=3;m=r+half;x=np.arange(m,dtype=float)
    V=np.column_stack([(x/m)**k for k in range(p+1)])
    dV=np.column_stack([np.zeros(m)]+[k/m*(x/m)**(k-1) for k in range(1,p+1)])
    offsets=np.arange(-half,half+1,dtype=float);target=np.zeros(2*half+1);target[1]=1
    fd=np.linalg.solve(np.array([offsets**k for k in range(2*half+1)]),target)
    fixed=np.zeros((r,m))
    for i in range(r):
        for j in range(r,min(m,i+half+1)):fixed[i,j]=fd[j-i+half]
    edge=np.zeros((r,m));edge[0,0]=-.5
    pairs=[(i,j) for i in range(r) for j in range(i+1,r)]
    rows=[];rhs=[]
    for i in range(r):
        for k in range(p+1):
            row=np.zeros(r+len(pairs));row[i]=-dV[i,k]
            for c,(j,l) in enumerate(pairs):
                if i==j:row[r+c]=V[l,k]
                elif i==l:row[r+c]=-V[j,k]
            rows.append(row);rhs.append(-sum((fixed[i]+edge[i])*V[:,k]))
    initial_weights=np.ones(r);initial_weights[0]=.5
    # For trapezoidal nodes, the one-sided moment correction is B_(k+1)/(k+1).
    # k=1: +1/12, k=3: -1/120; even corrected moments vanish.
    moments={1:1/12,3:-1/120}
    for k in range(2*p+1):
        row=np.zeros(r+len(pairs));row[:r]=(x[:r]/r)**k
        rows.append(row);rhs.append(sum(row[:r]*initial_weights)+moments.get(k,0)/r**k)
    A=np.array(rows);rhs=np.array(rhs);initial=np.r_[initial_weights,np.zeros(len(pairs))]
    value=initial+np.linalg.lstsq(A,rhs-mm(A,initial),rcond=1e-12)[0]
    error=float(np.max(abs(mm(A,value)-rhs)))
    if error>1e-11 or min(value[:r])<.05:raise ValueError('invalid uniform endpoint closure')
    S=np.zeros((r,r))
    for k,(i,j) in enumerate(pairs):S[i,j]=value[r+k];S[j,i]=-value[r+k]
    return value[:r],S,fd,error


@functools.lru_cache(None)
def local(n):
    if n<10:raise ValueError('endpoint closures require at least ten nodes')
    weights,S,fd,error=closure();r=5;half=3;p=2;h=1/(n-1)
    H=np.full(n,h);H[:r]=weights*h;H[-r:]=weights[::-1]*h
    x=np.linspace(0,1,n);trace=np.zeros((2,n));trace[0,0]=trace[1,-1]=1
    skew=np.zeros((n,n))
    for i in range(n):
        for j in range(max(0,i-half),min(n,i+half+1)):
            if not (i<r and j<r) and not (i>=n-r and j>=n-r):skew[i,j]=fd[j-i+half]
    skew[:r,:r]=S;skew[-r:,-r:]=-S[::-1,::-1]
    B=np.outer(trace[1],trace[1])-np.outer(trace[0],trace[0]);Q=skew+B/2
    L=np.zeros((n-p-1,n))
    for j in range(len(L)):L[j,j:j+p+2]=[(-1)**k*math.comb(p+1,k) for k in range(p+2)]
    R=mm(L.T,L)/(2**(2*p+2));V,dV=basis(x,p)
    return dict(x=x,H=H,trace=trace,D=Q/H[:,None],R=R,Q=Q,V=V,dV=dV,degree=p,
                closure_width=r,closure_residual=error)


def shell_grid(n,radial=(0,.45,.8,1),transfer_seed='zero'):
    # Constructor-only rule selection. All later operations use the immutable
    # per-block locals; the existing Lobatto factory is restored immediately.
    import mapped_sbp_probe as mapped
    original=mapped.lobatto
    try:
        mapped.lobatto=local
        return mapped.ShellGrid(n,radial=radial,transfer_seed=transfer_seed)
    finally:mapped.lobatto=original
