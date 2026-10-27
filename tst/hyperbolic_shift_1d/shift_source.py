"""Algebraic shift sources; no additional bulk principal derivatives.

For lapse-adapted sources sigma=alpha/A, D_t beta=-r*sigma^2+mu*D
-xi*(beta+r*sigma)/Omega. The restoring limit at scri uses the
derivative of the complete numerator, including sigma'.
"""
import numpy as np


def background(r,alpha,u,du,kind,w=None,wp=None):
    if kind=='reference':return -r,np.zeros_like(r),np.zeros_like(r)
    if kind not in ['lapse-adapted','lapse-inner']:raise ValueError('Unknown shift source')
    A=(1+r*r)/2
    sigma=alpha/A
    delta=u/A
    delta_prime=du/A-u*r/(A*A)
    if kind=='lapse-inner':
        if w is None or wp is None:raise ValueError('Inner source requires its smooth window and derivative')
        sigma=(1-w)+w*sigma
        delta_prime=wp*delta+w*delta_prime
        delta=w*delta
    return -r*sigma*sigma,r*delta,delta+r*delta_prime
