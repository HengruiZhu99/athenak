"""Explicit lapse-gauge variants; no metric or constraint projection."""
import numpy as np


def lapse_coefficient(r,alpha,slicing=1.,family='linear',scale=.1):
    O=(1-r*r)/2;A=(1+r*r)/2
    if family=='linear':return alpha*alpha+4*slicing*O*O*alpha
    if family=='rational':return alpha*alpha+4*slicing*O*O*alpha*(A+scale)/(alpha+scale)
    if family=='blended-log':return (1-4*O*O)*alpha*alpha+8*slicing*O**3*alpha
    raise ValueError('Unknown lapse principal coefficient')


def rational_correction(r,alpha,u,khat_deviation,slicing=1.,scale=.1):
    O=(1-r*r)/2
    return 4*slicing*O*O*alpha*u/(alpha+scale)*khat_deviation


def af_slicing_strength(slicing=1.,family='linear',scale=.1):
    return slicing*(.5+scale)/scale if family=='rational' else slicing


def blended_correction(r,alpha,khat_deviation,slicing=1.):
    O=(1-r*r)/2
    return 4*O*O*alpha*(alpha+slicing*r*r)*khat_deviation


def collapsed_background_correction(r,u):
    """Replace the additive inner r^2 source by r^2*alpha/Aref.

    A (2Omega)^4 taper leaves the scri normal jets unchanged. This changes
    a lapse source, not an Einstein equation or a constraint variable.
    """
    O=(1-r*r)/2;A=(1+r*r)/2
    return 16*O**4*r*r*u/A


def robin_penalty(r,omega,alpha,chi,beta,alpha_r,F,norm,lapse_power=2.):
    """Weak incoming-gauge Robin data at a finite puncture cutoff.

    Physical/constraint characteristics must already be outflow. For the
    lapse-gradient/Khat wave pair, the incoming excess speed supplies a
    continuous activation. Return a Khat rate; its tracefree-curvature lift
    2/(3Omega) is required by the principal constraint-free gauge mode.
    """
    speed=np.sqrt(F*chi);incoming=max(speed-beta,0.)
    residual=alpha_r-lapse_power*alpha/r
    value=-omega*chi/norm*incoming/speed*residual if incoming>0 else 0.
    return value,residual,incoming
