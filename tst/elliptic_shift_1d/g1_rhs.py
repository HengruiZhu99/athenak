"""Analytically canceled nonlinear spherical Z4c for the g=1 elliptic gauge.

Equivalent to the tensor kernel for metric-derived connection freezing. All
curvature, Khat, Theta and Lambda remain independent. No constraint projection.
"""
import numpy as np
import time


def radial_ratio(value,r,origin_limit):
    if r[0]>0:return value/r
    result=np.empty_like(value);result[1:]=value[1:]/r[1:];result[0]=origin_limit
    return result


def parts(ops,q,d,dd,kappa1=4.,kappa2=-.99,appendix_c=True,slicing=1.,restoring=0.,timing=None,
          total_chi=None,total_alpha=None,total_khat=None,puncture_cancellation=False,trumpet_chi=None):
    r=ops.r;O=(1-r*r)/2;A0=(1+r*r)/2
    c=q[:,0];chi=1+c;Arr=q[:,2];p=q[:,3]-3;T=q[:,4];lam=q[:,5];u=q[:,6]
    cp=d[:,0];cpp=dd[:,0];ap=d[:,2];pp=d[:,3];Tp=d[:,4];lp=d[:,5];up=d[:,6];upp=dd[:,6]
    alpha=A0+u;alphap=r+up;K=p+2*T;dk=q[:,3]+2*T
    if total_chi is not None:chi=np.asarray(total_chi)
    if total_alpha is not None:alpha=np.asarray(total_alpha)
    if total_khat is not None:p=np.asarray(total_khat);K=p+2*T
    if np.any(q[:,1]!=0) or min(min(chi),min(alpha))<=0:
        raise ValueError('g=1 backend requires its conformal metric gauge and positive chi/lapse')
    # Solve beta/r by quadrature. Store its deviation analytically, avoiding
    # subtraction of two O(1) shifts when the perturbation is near roundoff.
    elliptic_start=time.perf_counter()
    Arr_over_r=radial_ratio(Arr,r,0.)
    outer_v=-(c[-1]+(2*u[-1]+u[-1]*u[-1])*chi[-1])/(1+alpha[-1]*np.sqrt(chi[-1]))
    integrand=1.5*alpha*Arr_over_r
    if hasattr(ops,'shift_ode_solver'):v_over_r=ops.shift_ode_solver.solve(integrand,outer_v)
    elif getattr(ops,'elliptic_quadrature','prefix')=='inward':v_over_r=outer_v-ops.tail_integral(integrand)
    else:
        integral=ops.integral(integrand);v_over_r=outer_v+integral-integral[-1]
    v=r*v_over_r;beta=-r+v
    dbeta=-1+v_over_r+1.5*alpha*Arr
    ddbeta=1.5*(alphap*Arr+alpha*ap)+1.5*alpha*Arr_over_r
    if timing is not None:timing['elliptic_seconds']=time.perf_counter()-elliptic_start;timing['beta_deviation']=v;timing['origin_shift_slope']=-1+v_over_r[0]
    divbeta=-3+3*v_over_r+1.5*alpha*Arr
    cp_over_r=radial_ratio(cp,r,cpp[0]);up_over_r=radial_ratio(up,r,upp[0]);lambda_over_r=radial_ratio(lam,r,lp[0])
    w=beta*r/alpha
    R=np.zeros_like(q);S=np.zeros_like(q)
    R[:,0]=beta*cp-2*chi*v_over_r-alpha*chi*Arr
    S[:,0]=2*chi/3*(alpha*dk-3*u-3*r*v)
    # The metric equation is identically Q=0 from the elliptic gauge, not a
    # subtraction of any background/Schwarzschild RHS or a post-step projection.
    R[:,1]=0.
    R[:,2]=(beta*ap-2*chi/3*(upp-up_over_r)-2*cp*alphap/3
        +alpha/3*(cpp-cp_over_r)-alpha*cp*cp/(6*chi)
        +2*alpha*chi/3*(lp-lambda_over_r)+2*alpha*cp*lam/3)
    S[:,2]=4*alpha/3*(-r)*(cp+chi*lam)+alpha*Arr*(K-w)
    R[:,3]=(beta*pp+1.5*O*alpha*Arr*Arr-O*chi*(upp+2*up_over_r)
        +.5*O*cp*alphap-3*chi*r*up-6*A0*c+3*u-3*c*u+.5*alpha*r*cp)
    S[:,3]=alpha*(dk*(K-3)/3-3*c*r*r+kappa1*(1-kappa2)*T)
    R[:,4]=(beta*Tp+alpha*r*cp+O*alpha/2*(2*cpp+4*cp_over_r-2.5*cp*cp/chi
        -1.5*Arr*Arr+chi*(lp+2*lambda_over_r)))
    S[:,4]=alpha*(dk*(K-3)/3-3*c-kappa1*(2+kappa2)*T)
    if not appendix_c:S[:,4]-=3*alpha*w*T
    R[:,5]=beta*lp+2*lam*divbeta/3+2*alpha*(ap+3*Arr_over_r-1.5*Arr*cp/chi)
    S[:,5]=-alpha*(2*K/3+kappa1)*lam-2*alpha/3*(2*pp+Tp)+4*alpha*Arr*r
    F=alpha*alpha+4*slicing*O*O*alpha
    R[:,6]=beta*up+v*r
    S[:,6]=-F*q[:,3]+r*(A0*v-r*u+u*v)-restoring*(alpha+A0)*u
    inner=r<.5 if puncture_cancellation else np.zeros(len(r),dtype=bool)
    if np.any(inner):
        # Equivalent algebra around chi=alpha=Khat=0 at the other AF end.
        # Transfer analytic Omega multiples between R/S before division.
        S[inner,0]=2*chi[inner]/3*(alpha[inner]*K[inner]+3*A0[inner]-3*r[inner]*v[inner])
        R[inner,3]=(beta*pp+1.5*O*alpha*Arr*Arr-O*chi*(upp+2*up_over_r)
            +.5*O*cp*alphap-3*chi*r*up-3*chi*(A0+alpha)+.5*alpha*r*cp)[inner]
        S[inner,3]=(alpha*(K*K/3-3*chi*r*r+kappa1*(1-kappa2)*T))[inner]
        S[inner,4]=(alpha*(K*K/3-3*chi-kappa1*(2+kappa2)*T))[inner]
        if not appendix_c:S[inner,4]-=(3*alpha*w*T)[inner]
        R[inner,6]=(beta*alphap)[inner]
        S[inner,6]=(-F*q[:,3]+r*alpha*beta+r*r-restoring*(alpha+A0)*u)[inner]
    if trumpet_chi is not None:
        X=trumpet_chi['X'];ell=trumpet_chi['ell'];ellp=trumpet_chi['ellp'];alphapp=1+upp
        # chi=r^2 X: cancel the exact leading puncture powers before any
        # 1/r terms are evaluated. Curvature/Khat/Theta/Lambda stay free.
        curvature=alpha*X*(-2/3+r*ell/3+r*r*ellp/3+r*r*ell*ell/6)
        hessian=-2*X/3*(r*r*alphapp+r*alphap+r*r*ell*alphap)
        connection=2*alpha*X/3*(r*r*lp+r*lam+r*r*ell*lam)
        R[inner,0]=(chi*(beta*ell-alpha*Arr))[inner]
        S[inner,0]=(chi*(2*alpha*K/3-2*r*beta))[inner]
        R[inner,2]=(beta*ap+curvature+hessian+connection)[inner]
        S[inner,2]=(-4*alpha*chi/3*(2+r*(ell+lam))+alpha*Arr*K-beta*r*Arr)[inner]
        R[inner,3]=(beta*pp+1.5*O*alpha*Arr*Arr-O*X*(r*r*alphapp+r*alphap)
            +.5*O*chi*ell*alphap-3*chi*r*alphap-2*chi*alpha+.5*alpha*chi*r*ell)[inner]
        curvature_scalar=2*X+2*r*X*ell+2*chi*ellp-.5*chi*ell*ell
        R[inner,4]=(beta*Tp+alpha*chi*(2+r*ell)+O*alpha/2*(curvature_scalar-1.5*Arr*Arr+chi*lp+2*r*X*lam))[inner]
        R[inner,5]=(beta*lp+2*lam*divbeta/3+2*alpha*(ap-1.5*Arr*ell))[inner]
    H=(2*dk*(K-3)/3-6*c+2*O*r*cp
       +O*O*(2*cpp+4*cp_over_r-2.5*cp*cp/chi-1.5*Arr*Arr))
    H[inner]=(2*K*K/3-6*chi+2*O*r*cp+O*O*(2*cpp+4*cp_over_r-2.5*cp*cp/chi-1.5*Arr*Arr))[inner]
    M=O*(ap+3*Arr_over_r-1.5*Arr*cp/chi)-2*(pp+2*Tp)/3+2*r*Arr
    if trumpet_chi is not None:
        H[inner]=(2*K*K/3-6*chi+2*O*chi*(2+r*ell)+O*O*(curvature_scalar-1.5*Arr*Arr))[inner]
        M[inner]=(O*(ap-1.5*Arr*ell)-2*(pp+2*Tp)/3+2*r*Arr)[inner]
    diag=np.zeros((len(r),12));diag[:,0]=H;diag[:,1]=M;diag[:,2]=T;diag[:,3]=lam/2
    diag[:,6]=chi*r*r-w*w;diag[:,7]=alpha;diag[:,8]=chi;diag[:,9]=1.
    diag[:,10]=np.nan
    interior=O>0
    # Canceled invariant mass: exact zero for the unperturbed reference.
    N=(dk*(K-3)/9-c+O*(-K*Arr/3+A0*cp_over_r)+O*O*(Arr*Arr/4-cp*cp/(4*chi)))
    N[inner]=(K*K/9-chi+O*(-K*Arr/3+A0*cp_over_r)+O*O*(Arr*Arr/4-cp*cp/(4*chi)))[inner]
    if trumpet_chi is not None:
        N[inner]=(K*K/9-chi+O*(-K*Arr/3+A0*X*(2+r*ell))
            +O*O*(Arr*Arr/4-X*(1+r*ell+r*r*ell*ell/4)))[inner]
    if timing is not None:timing['mass_numerator']=N
    diag[interior,10]=r[interior]**3*N[interior]/(2*chi[interior]**1.5*O[interior]**3)
    if np.any(inner):
        radius=r[inner]/(O[inner]*np.sqrt(chi[inner]));normal=radius*(K[inner]/3-O[inner]*Arr[inner]/2)
        spatial=A0[inner]/O[inner]-r[inner]*cp[inner]/(2*chi[inner])
        diag[inner,10]=radius/2*(1+normal*normal-spatial*spatial)
    if trumpet_chi is not None:
        radius=1/(O[inner]*np.sqrt(X[inner]));normal=radius*(K[inner]/3-O[inner]*Arr[inner]/2)
        spatial=r[inner]**2/O[inner]-r[inner]*ell[inner]/2
        diag[inner,10]=radius/2*(1+normal*normal-spatial*spatial)
    kang=-O*Arr/2+K/3
    spatial=np.empty_like(r);positive=r>0
    spatial[positive]=np.sqrt(chi[positive])*(A0[positive]/r[positive]-.5*O[positive]*cp[positive]/chi[positive])
    spatial[~positive]=np.inf # Analytic expansion of infinitesimal regular spheres.
    if trumpet_chi is not None:spatial[inner]=(np.sqrt(X)*(r*r-.5*O*r*ell))[inner]
    diag[:,11]=2*(spatial-kang)
    return R,S,diag,beta,dbeta,ddbeta
