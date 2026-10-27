"""Unit-sphere conformal frame and its scri generator for g=1.

g_B=chi*g_bar, Omega_B=sqrt(chi)*Omega, n_B^t=1/alpha.
No assumption of affine coordinate time is made in this diagnostic.
"""
import numpy as np


def nonaffinity(chi,alpha,chi_r,alpha_r,beta_r,chi_t):
    """n_B^b nabla^B_b n_B^a = kappa n_B^a at null scri.

    The null shift beta=-alpha*sqrt(chi) and its time derivative are used,
    but radial derivatives are independent. The clock has alpha=1 in runs.
    """
    return chi_t/(2*alpha*chi)-chi_r/(2*np.sqrt(chi))-np.sqrt(chi)*alpha_r/alpha-beta_r/alpha
