"""Fixed numerical collocation maps; physical coordinates/equations remain r."""
from math import comb
import numpy as np

def mapped_grid(x,strength=0.,kind='sinh'):
    if not np.isfinite(strength) or not 0<=strength<=6:raise ValueError('Grid stretch must be in [0,6]')
    if kind not in ['sinh','inner-c7']:raise ValueError('Unknown grid map')
    if strength==0:return x.copy(),np.ones_like(x)
    if kind=='sinh':return np.sinh(strength*x)/np.sinh(strength),strength*np.cosh(strength*x)/np.sinh(strength)
    # Bernstein form avoids cancellation in the C7 smoothstep polynomial.
    # r=x beyond x=.4, preserving the uniform scri and outer-taper grid.
    z=np.clip(x/.4,0,1);S=sum(comb(15,k)*z**k*(1-z)**(15-k) for k in range(8,16))
    Sp=51480*z**7*(1-z)**7
    small=strength/np.sinh(strength)
    return x*(small+(1-small)*S),small+(1-small)*(S+z*Sp)
