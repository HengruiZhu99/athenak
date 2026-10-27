"""Optional finite-product, lapse-adjusted Z4 damping in the interior.

Replace alpha*kappa1 by (1-w)*4*alpha + w*kappa_inner*A.
Apply the exact difference of the native kappa-dependent pole numerators.
There is no division by alpha, no constraint projection, and no scri change.
"""
import numpy as np


def correction(r,alpha,theta,connection_error,w,kappa_inner,kappa2=-.99):
    if not np.isfinite(kappa_inner) or kappa_inner<0:
        raise ValueError('Inner Z4 damping must be finite and nonnegative')
    result=np.zeros((len(r),8))
    if kappa_inner==0:return result
    delta=w*(kappa_inner*(1+r*r)/2-4*alpha)
    result[:,3]=delta*(1-kappa2)*theta
    result[:,4]=-delta*(2+kappa2)*theta
    result[:,5]=-delta*connection_error
    return result
