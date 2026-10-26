"""Nonlinear radius-weighted massless scalar, including off-constraint Z4c geometry.

f=(R phi)/r=phi/(Omega sqrt(chi)), P=(n_bar(R phi))/r. Both are even at a
regular center. The angular Ricci term is retained; no Einstein constraint is
silently used to change the scalar equation off constraint.
"""
import numpy as np
from g1_rhs import radial_ratio


def matter(ops,q,d,fields,dfields,beta,total_chi=None):
    r=ops.r;O=(1-r*r)/2;chi=1+q[:,0];cp=d[:,0]
    if total_chi is not None:chi=np.asarray(total_chi)
    K=q[:,3]-3+2*q[:,4];Arr=q[:,2];k=K/3-O*Arr/2
    f=fields[:,0];P=fields[:,1]
    phi=O*np.sqrt(chi)*f
    phi_r=np.sqrt(chi)*(O*dfields[:,0]+(-r+.5*O*cp/chi)*f)
    normal_phi=np.sqrt(chi)*(O*P+k*f)
    return phi,phi_r,normal_phi


def add_sources(r,q,R,diagnostic,phi_r,normal_phi,total_chi=None,total_alpha=None):
    O=(1-r*r)/2;chi=1+q[:,0];alpha=(1+r*r)/2+q[:,6]
    if total_chi is not None:chi=np.asarray(total_chi)
    if total_alpha is not None:alpha=np.asarray(total_alpha)
    density=normal_phi*normal_phi+chi*phi_r*phi_r
    R[:,2]-=16*np.pi/3*alpha*chi*phi_r*phi_r
    R[:,3]+=8*np.pi*alpha*O*normal_phi*normal_phi
    R[:,4]-=4*np.pi*alpha*O*density
    R[:,5]+=16*np.pi*alpha*normal_phi*phi_r
    diagnostic[:,0]-=8*np.pi*O*O*density
    diagnostic[:,1]+=8*np.pi*O*normal_phi*phi_r


def angular_ricci(r,q,d,beta,kappa1=4.,kappa2=-.99,appendix_c=True,damping_form='constant',total_chi=None,total_alpha=None):
    O=(1-r*r)/2;chi=1+q[:,0];cp=d[:,0];lam=q[:,5];T=q[:,4]
    alpha=(1+r*r)/2+q[:,6]
    if total_chi is not None:chi=np.asarray(total_chi)
    if total_alpha is not None:alpha=np.asarray(total_alpha)
    Ric=(-O*O*chi*radial_ratio(lam,r,d[0,5])+O*O*cp*lam/3-2*O*r*chi*lam/3+kappa1*(1+kappa2)*T)
    if damping_form=='reference-lapse':Ric+=kappa1*(1+kappa2)*(-q[:,6])*T/alpha
    if not appendix_c:Ric+=2*beta*r*T/alpha
    return Ric


def rhs(ops,q,d,fields,dfields,ddfields,beta,beta_deviation,kappa1=4.,kappa2=-.99,appendix_c=True,geometry_second=None,origin_beta_derivative=None,analytic_endpoint=False,damping_form='constant',total_chi=None,total_alpha=None):
    r=ops.r;O=(1-r*r)/2;A0=(1+r*r)/2
    c=q[:,0];chi=1+c;cp=d[:,0];Arr=q[:,2];T=q[:,4];lam=q[:,5];u=q[:,6]
    alpha=A0+u;alphap=r+d[:,6];K=q[:,3]-3+2*T;dk=q[:,3]+2*T
    if total_chi is not None:chi=np.asarray(total_chi)
    if total_alpha is not None:alpha=np.asarray(total_alpha)
    v=beta_deviation
    f=fields[:,0];P=fields[:,1];fp=dfields[:,0];Pp=dfields[:,1];fpp=ddfields[:,0]
    U=alpha*dk/3-u-r*v
    akr=np.empty_like(r);akr[:-1]=-1+alpha[:-1]*Arr[:-1]+U[:-1]/O[:-1]
    dU,_=ops.derivatives(U,1)
    if analytic_endpoint:
        dU[-1]=(alphap[-1]*dk[-1]/3+alpha[-1]*(d[-1,3]+2*d[-1,4])/3-d[-1,6]
            -v[-1]-(v[-1]+1.5*alpha[-1]*Arr[-1]))
    akr[-1]=-1+alpha[-1]*Arr[-1]-dU[-1]
    # The exact 2D wave potential: 2m/(Omega^2 R^3)-Ricci_angular/Omega^2.
    chi_second0=0.;alpha_second0=0.
    if r[0]==0:
        chi_second0=geometry_second[0,0] if geometry_second is not None else ops.derivatives(cp,-1)[0][0]
        alpha_second0=1+(geometry_second[0,6] if geometry_second is not None else ops.derivatives(d[:,6],-1)[0][0])
    N=dk*(K-3)/9-c+O*(-K*Arr/3+A0*radial_ratio(cp,r,chi_second0))+O*O*(Arr*Arr/4-cp*cp/(4*chi))
    B=N-kappa1*(1+kappa2)*T+2*O*r*chi*lam/3
    if damping_form=='reference-lapse':B-=kappa1*(1+kappa2)*(-u)*T/alpha
    if not appendix_c:B-=2*beta*r*T/alpha
    V=np.empty_like(r);V[:-1]=B[:-1]/(O[:-1]*O[:-1])+chi[:-1]*radial_ratio(lam,r,d[0,5])[:-1]-cp[:-1]*lam[:-1]/3
    # Finite mass and the compatible Theta/Z weights give this analytic limit.
    V[-1]=0.
    result=np.empty_like(fields)
    beta_origin=0.
    if r[0]==0:beta_origin=origin_beta_derivative if origin_beta_derivative is not None else ops.derivatives(beta,-1)[0][0]
    beta_over_r=radial_ratio(beta,r,beta_origin)
    result[:,0]=beta*fp+beta_over_r*f+alpha*P
    mix=.5*alpha*cp+chi*alphap
    fp_over_r=radial_ratio(fp,r,fpp[0])
    mix_limit=.5*alpha[0]*chi_second0+chi[0]*alpha_second0
    mix_over_r=radial_ratio(mix,r,mix_limit)
    result[:,1]=(beta*Pp+(beta_over_r+akr)*P+alpha*chi*(fpp+2*fp_over_r)
                 +mix*fp+(mix_over_r-alpha*V)*f)
    return result,{'potential':V,'potential_numerator':B,'normal_curvature_numerator':U}
