"""Frozen Cartesian scalar Z4c principal operator with actual p-only face updates.

State [Khat,Theta,A_nn,Gamma_x,chi,h_nn,alpha,beta_x], variable-major.
Metric and A have transverse components -h_nn/2 and -A_nn/2.
No radial 1/r terms; this is not the variable-coefficient trumpet operator.
"""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents
          if (p/'analysis/z4c_characteristic/check_residual_characteristics_numeric.py').is_file())
spec=importlib.util.spec_from_file_location('characteristics',ROOT/'analysis/z4c_characteristic/check_residual_characteristics_numeric.py')
char=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(char)

def weights(x, point, derivative=1):
    x=np.asarray(x,dtype=float)-point
    return np.linalg.solve(np.array([x**k/math.factorial(k) for k in range(len(x))]),np.eye(len(x))[derivative])

def extension(n,degree=3,ng=4):
    E=np.zeros((n+2*ng,n)); E[ng:ng+n]=np.eye(n)
    for pos in range(-ng,n+ng):
        if 0<=pos<n: continue
        ids=np.arange(degree+1) if pos<0 else np.arange(n-degree-1,n)
        E[pos+ng,ids]=weights(ids,pos,0)
    return E

def stencil(n,h,offsets,values,E):
    S=np.zeros((n,E.shape[0])); ng=(E.shape[0]-n)//2
    for i in range(n): S[i,i+ng+np.asarray(offsets)]=values
    return S@E/h

def active_derivative(n,h,order):
    D=np.zeros((n,n)); count=order+1
    for i in range(n):
        first=min(max(i-order//2,0),n-count); ids=np.arange(first,first+count)
        D[i,ids]=weights(ids,i)/h
    return D

def operators(n,h,degree=3,beta=0.,diss=.5,ng=4):
    E=extension(n,degree,ng)
    if ng==4:
        D=stencil(n,h,range(-3,4),[-1/60,3/20,-3/4,0,3/4,-3/20,1/60],E)
        Dxx=stencil(n,h*h,range(-3,4),[1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90],E)
        up_offsets=np.arange(-2,5); up_values=np.array([1/30,-2/5,-7/12,4/3,-1/2,2/15,-1/60])
        ko_offsets=np.arange(-4,5); ko_values=np.array([1,-8,28,-56,70,-56,28,-8,1])*(-diss/256)
    elif ng==2:
        D=stencil(n,h,[-1,0,1],[-.5,0,.5],E)
        Dxx=stencil(n,h*h,[-1,0,1],[1,-2,1],E)
        up_offsets=np.arange(3); up_values=np.array([-1.5,2,-.5])
        ko_offsets=np.arange(-2,3); ko_values=np.array([1,-4,6,-4,1])*(-diss/16)
    else: raise ValueError(ng)
    if beta<0: up_offsets=-up_offsets;up_values=-up_values
    adv=beta*stencil(n,h,up_offsets,up_values,E)
    KO=stencil(n,h,ko_offsets,ko_values,E)
    return D,Dxx,adv,KO,E

def matrix(n=32,h=.125,degree=3,beta=0.,diss=.5,mode='radiation',inner=4,outer=2,tau=1.,alpha=1.,chi=1.,shift=2.,damping=False,ng=4):
    D,Dxx,adv,KO,E=operators(n,h,degree,beta,diss,ng)
    P=char.scalar_matrix(alpha,chi,2*alpha,shift)
    L=np.zeros((8*n,8*n)); eye=np.eye(n)
    sl=lambda j:slice(j*n,(j+1)*n)
    for a in range(8):
        for b in range(8):
            if a<4: deriv=D if b<4 else Dxx
            else: deriv=eye if b<4 else D
            L[sl(a),sl(b)]=P[a,b]*deriv
        L[sl(a),sl(a)]+=adv+KO
    Dm=D if inner=='volume' else active_derivative(n,h,inner)
    Do=D if outer=='volume' else active_derivative(n,h,outer)
    Db=active_derivative(n,h,2)
    if damping:
        sigma=.1*alpha
        L[sl(0),sl(1)]+=sigma*eye
        L[sl(1),sl(1)]-=2*sigma*eye
        L[sl(3),sl(3)]-=2*sigma*eye
        L[sl(3),sl(5)]+=2*sigma*D
        L[sl(7),sl(7)]-=2*eye
    V=L.copy()
    # Uniform conformal metric delta_ij: Q_x = Gamma_x - d_x(h_nn).
    # chi=1 is the Minkowski case used for all default comparisons.
    Q=np.zeros((n,8*n));Q[:,sl(3)]=eye;Q[:,sl(5)]=-Dm
    constraint_deltas=[]
    if mode=='none': return L,{'volume':V,'Q':Q,'E':E}
    for pos,normal in [(0,-1),(n-1,1)]:
        # scalar_left(+1) is the incoming derivative-eigenvalue branch in the outward frame.
        left,speeds=char.scalar_left(alpha,chi,2*alpha,shift,1)
        lp=left[:,:4];ld=left[:,4:]
        p_rhs=np.vstack([V[j*n+pos] for j in range(4)])
        p_rhs[3]*=normal
        d_rhs=np.vstack([normal*Db[pos]@V[sl(j)] for j in range(4,8)])
        d_rhs[3]*=normal
        rates=lp@p_rhs+ld@d_rhs
        delta=-rates
        if mode=='radiation':
            c=alpha*math.sqrt(chi);lam=normal*beta+c;transport=normal*c-beta
            ftheta=V[n+pos].copy();ftheta[sl(1)]+=transport*Do[pos]
            fq=normal*(Q[pos]@V+transport*Do[pos]@Q)
            delta[2]=-tau*lam*ftheta/alpha
            delta[3]=tau*lam*fq/c
        elif mode!='zero_rate': raise ValueError(mode)
        correction=np.linalg.solve(lp,delta)
        correction[3]*=normal
        for j in range(4):L[j*n+pos]+=correction[j]
        constraint_deltas.append(delta[2:].copy())
    return L,{'volume':V,'Q':Q,'E':E,'constraint_deltas':constraint_deltas}

def vector_matrix(n=32,h=.125,degree=3,beta=0.,diss=.5,mode='radiation',inner=4,outer=2,tau=1.,alpha=1.,chi=1.,shift=2.,damping=False,ng=4):
    """One transverse polarization, state [A_xA,Gamma_A,g_xA,beta_A]."""
    D,Dxx,adv,KO,E=operators(n,h,degree,beta,diss,ng)
    P=char.vector_matrix(alpha,chi,shift)
    L=np.zeros((4*n,4*n));eye=np.eye(n);sl=lambda j:slice(j*n,(j+1)*n)
    for a in range(4):
        for b in range(4):
            deriv=(D if b<2 else Dxx) if a<2 else (eye if b<2 else D)
            L[sl(a),sl(b)]=P[a,b]*deriv
        L[sl(a),sl(a)]+=adv+KO
    Dm=D if inner=='volume' else active_derivative(n,h,inner)
    Do=D if outer=='volume' else active_derivative(n,h,outer)
    Db=active_derivative(n,h,2)
    Q=np.zeros((n,4*n));Q[:,sl(1)]=eye;Q[:,sl(2)]=-Dm
    if damping:
        L[sl(1),sl(1)]-=.2*alpha*eye
        L[sl(1),sl(2)]+=.2*alpha*D
        L[sl(3),sl(3)]-=2*eye
    V=L.copy()
    if mode=='none':return L,{'volume':V,'Q':Q,'E':E}
    for pos,s in [(0,-1),(n-1,1)]:
        mu=math.sqrt(shift);c=alpha*math.sqrt(chi);lam=s*beta+c;transport=s*c-beta
        # A_nA and g_nA change sign; Gamma_A and beta_A do not.
        # d_n g_nA therefore equals D_x g_xA on either face.
        gauge_rate=mu*V[n+pos]+s*Db[pos]@V[sl(3)]
        dGamma=-gauge_rate/mu
        constraint_rate=-2*s*V[pos]/math.sqrt(chi)-V[n+pos]+Db[pos]@V[sl(2)]
        delta=-constraint_rate
        if mode=='radiation':delta=tau*lam*(Q[pos]@V+transport*Do[pos]@Q)/c
        elif mode!='zero_rate':raise ValueError(mode)
        dA=-.5*math.sqrt(chi)*(delta+dGamma)
        L[pos]+=s*dA;L[n+pos]+=dGamma
    return L,{'volume':V,'Q':Q,'E':E}

def summary(L,h,cfl=.15,nfields=8):
    eig,vec=np.linalg.eig(L);idx=np.argmax(eig.real);val=eig[idx]
    z=cfl*h*eig;R=1+z+z*z/2+z*z*z/6
    v=vec[:,idx].reshape(nfields,-1);n=v.shape[1]
    # Derivative variables use h scaling only for this localization diagnostic.
    scaled=v.copy();scaled[:nfields//2]*=h
    energy=np.sum(abs(scaled)**2,axis=0)
    return {'max_real':float(val.real),'max_imag_at_peak':float(val.imag),
            'rk3_radius':float(max(abs(R))),'rk3_growth':float(np.log(max(abs(R)))/(cfl*h)),
            'positive_count':int(np.sum(eig.real>1e-7/h)),
            'boundary_four_cell_fraction':float((sum(energy[:4])+sum(energy[-4:]))/sum(energy)),
            'theta_fraction_scaled':float(sum(abs(scaled[1])**2)/sum(energy)),
            'eigen_residual':float(np.linalg.norm(L@vec[:,idx]-val*vec[:,idx])/max(1,np.linalg.norm(L@vec[:,idx])))}

if __name__=='__main__':
    import json
    results=[]
    for n in [16,32,64]:
      for degree in [1,3]:
       for mode in ['zero_rate','radiation']:
        for inner in ([4] if mode=='zero_rate' else [2,4,'volume']):
         for tau in ([1.] if mode=='zero_rate' else [1.,.5,.25]):
          params=dict(n=n,h=4/n,degree=degree,mode=mode,inner=inner,tau=tau)
          L,_=matrix(**params);result={**params,**summary(L,params['h'])}
          results.append(result);print(json.dumps(result),flush=True)
    Path(__file__).with_name('principal-results.json').write_text(json.dumps(results,indent=2)+'\n')
