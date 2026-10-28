"""Nonadvective direct/adapting-reference gauge and its scalar boundary symbol.

L is defined by partial_t U = L partial_r U, so coordinate speeds are -eig(L).
The boundary jet uses the incoming pure gauge vector, not a Lambda-only reset.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hyperbolic_shift_1d'))
from gamma_driver import collar
from principal import symbol as advective_symbol


def coefficient(r, g, beta, lapse_squared, bulk=4.):
    outer=.75*g*(2*lapse_squared+beta**2)
    c=collar(r)
    return (1-c)*bulk+c*outer


def shift_speeds(beta,g,mu):
    root=np.sqrt(beta**2+16*mu/(3*g))
    return (-beta-root)/2,(-beta+root)/2


def separation(beta,g,mu,lapse_squared):
    """Sufficient strict separation from both advected lapse speeds (times g)."""
    return 4*mu/3-g*(lapse_squared+abs(beta)*np.sqrt(lapse_squared))


def symbol(alpha,chi,g,F,eta,w,kappa,tau,mu,beta):
    L=advective_symbol(alpha,chi,g,F,eta,w,kappa,tau,mu)[:9,:9]+beta*np.eye(9)
    L[7]=0.;L[7,5]=mu
    return L


def incoming_vector(chi,g,beta,mu):
    """Exact right eigenvector, normalized by delta Lambda = 1.

    Only chi_r, g_r, Lambda, beta_r change. In particular delta(Lambda-V)=0.
    The evolving reference b is a tangent variable with no incoming component.
    """
    lam=(beta+np.sqrt(beta*beta+16*mu/(3*g)))/2
    r=np.zeros(9);r[[0,1,5,7]]=[-.5*chi*g,g*g,1.,mu/lam]
    return lam,r
