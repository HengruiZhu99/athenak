"""Fixed numerical collocation maps; physical coordinates/equations remain r."""
from math import comb
import numpy as np

def mapped_grid(x,strength=0.,kind='sinh'):
    if not np.isfinite(strength) or not 0<=strength<=20:raise ValueError('Grid stretch must be in [0,20]')
    if kind not in ['sinh','inner-c7','inner-exp']:raise ValueError('Unknown grid map')
    if strength==0:return x.copy(),np.ones_like(x)
    if kind=='sinh':return np.sinh(strength*x)/np.sinh(strength),strength*np.cosh(strength*x)/np.sinh(strength)
    if kind=='inner-exp':
        # Exponential spacing on x<=.3, smoothly matched to r=x by x=.5.
        # E(x)<=x on [0,.5], so every term in the derivative is nonnegative.
        z=np.clip((x-.3)/.2,0,1)
        S=sum(comb(15,k)*z**k*(1-z)**(15-k) for k in range(8,16))
        Sp=51480*z**7*(1-z)**7/.2
        argument=2*strength*np.minimum(x,.5)
        E=.5*np.expm1(argument)/np.expm1(strength)
        Ep=strength*np.exp(argument)/np.expm1(strength)
        return (1-S)*E+S*x,(1-S)*Ep+S+Sp*(x-E)
    # Bernstein form avoids cancellation in the C7 smoothstep polynomial.
    # r=x beyond x=.4, preserving the uniform scri and outer-taper grid.
    z=np.clip(x/.4,0,1);S=sum(comb(15,k)*z**k*(1-z)**(15-k) for k in range(8,16))
    Sp=51480*z**7*(1-z)**7
    small=strength/np.sinh(strength)
    return x*(small+(1-small)*S),small+(1-small)*(S+z*Sp)
