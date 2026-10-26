"""Product-rule scri derivatives of complete g=1 pole numerators.

These retain finite off-constraint traces. No field/constraint reset occurs.
"""
import numpy as np


def derivatives(q,d,dd,beta_deviation,kappa1=4.,kappa2=-.99,appendix_c=True,slicing=1.,restoring=1.,restoring_form='symmetric',damping_form='constant'):
    c=q[0];chi=1+c;Ar=q[2];p=q[3]-3;T=q[4];lam=q[5];u=q[6]
    cp=d[0];ap=d[2];pp=d[3];Tp=d[4];lp=d[5];up=d[6]
    alpha=1+u;alphap=1+up;K=p+2*T;dk=q[3]+2*T;Kp=pp+2*Tp
    v=beta_deviation;vp=v+1.5*alpha*Ar;beta=-1+v;betap=-1+vp
    w=beta/alpha;wp=(betap+beta)/alpha-w*alphap/alpha
    U=alpha*dk/3-u-v;Up=alphap*dk/3+alpha*Kp/3-up-v-vp
    result=np.zeros_like(q)
    result[0]=2*(cp*U+chi*Up)
    result[2]=(-4/3*((alphap+alpha)*(cp+chi*lam)+alpha*(dd[0]+cp*lam+chi*lp))
        +(alphap*Ar+alpha*ap)*(K-w)+alpha*Ar*(Kp-wp))
    Y=dk*(K-3)/3-3*c+kappa1*(1-kappa2)*T
    Yp=2*K*Kp/3-3*cp-6*c+kappa1*(1-kappa2)*Tp
    result[3]=alphap*Y+alpha*Yp
    Z=dk*(K-3)/3-3*c-kappa1*(2+kappa2)*T
    Zp=2*K*Kp/3-3*cp-kappa1*(2+kappa2)*Tp
    if not appendix_c:Z-=3*w*T;Zp-=3*(wp*T+w*Tp)
    result[4]=alphap*Z+alpha*Zp
    result[5]=(-alphap*(2*K/3+kappa1)*lam-alpha*(2*Kp*lam/3+(2*K/3+kappa1)*lp)
        -2/3*(alphap*(2*pp+Tp)+alpha*(2*dd[3]+dd[4]))
        +4*(alphap*Ar+alpha*ap+alpha*Ar))
    result[6]=(-2*alpha*alphap*q[3]-alpha*alpha*pp+(v-u+u*v)
        +(v+vp-u-up+up*v+u*vp)-restoring*((alphap+1)*u+(alpha+1)*up))
    if restoring_form=='multiplicative':result[6]-=2*restoring*u*up
    if damping_form=='reference-lapse':
        result[3]-=kappa1*(1-kappa2)*(up*T+u*Tp)
        result[4]+=kappa1*(2+kappa2)*(up*T+u*Tp)
        result[5]+=kappa1*(up*lam+u*lp)
    return result,Up


def corner_correction(q,d,dd,R,beta_deviation,h,coefficient,derivative_lift,**kwargs):
    """Damp the raw chi/clock-rate corner through a value-zero derivative lift.

    The only added rates are chi and Khat at the penultimate node. Their
    endpoint values stay unchanged. The directional Jacobian of (chi_t,alpha_t)
    with respect to (chi_r,Khat_r) is triangular, so this correction contributes
    exactly -coefficient/h times each corner residual to its own time rate.
    """
    ds,_=derivatives(q,d,dd,beta_deviation,**kwargs);raw=R-ds
    alpha=1+q[6];chi=1+q[0];dk=q[3]+2*q[4];v=beta_deviation
    U=alpha*dk/3-q[6]-v;Jchi=(-1+v)-2*U;Jp=-2*chi*alpha/3
    p_rate=-coefficient/h*raw[6]/(alpha*alpha)
    chi_rate=(-coefficient/h*raw[0]-Jp*p_rate)/Jchi
    return np.array([chi_rate,p_rate])*derivative_lift,raw[[0,6]]
