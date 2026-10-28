"""Moving-puncture bulk coefficient with a C7 outer gauge collar.

The advective two-variable Gamma driver uses the independent evolved Lambda.
Only the prescribed Minkowski source -r remains in the bulk beta equation.
No lapse or conformal-factor suppression is applied to its bulk coefficient.
"""
import numpy as np


def collar(r, start=.8, end=.9):
    x=np.clip((np.asarray(r)-start)/(end-start),0.,1.)
    a=x**8;b=(1-x)**8
    return a/(a+b)


def parameters(r, separated_mu, restoring, bulk_mu=1.):
    if not np.isfinite(bulk_mu) or bulk_mu<=0:
        raise ValueError('Gamma coefficient must be finite and positive')
    c=collar(r)
    return (1-c)*bulk_mu+c*separated_mu,restoring*c
