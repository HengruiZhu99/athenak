"""Coordinate-only positive metric profiles; independent of gauge source/mass."""
import numpy as np
from wormhole_data import Jet,transition

def trumpet_metric_profile(r,compact=False):
    """chi_ref=r^2/D, returning total/deviation jets and log(chi_ref/r^2) jets.

    Compact profile equals r^2/A^2 below.1 and exactly1 beyond.3. All
    transitions are C-infinity. This is an invertible variable choice only.
    """
    if min(r)<=0:raise ValueError('Trumpet metric profiles require a positive physical cutoff')
    rows=[]
    for value in r:
        if compact and value>=.3:
            rows.append([1.,0.,0.,0.,0.,0.,-2/value,2/value**2]);continue
        x=Jet(float(value),1);O=(1-x*x)/2;A=(1+x*x)/2
        if not compact or value<=.1:D=A*A;s=Jet(1)
        elif value>=.3:D=x*x;s=Jet(0)
        else:s=1-transition((x-.1)/.2);D=x*x+O*O*s
        total=x*x/D;deviation=-O*O*s/D
        ell=-D.d/D.v;ellp=-D.dd/D.v+(D.d/D.v)**2
        rows.append([total.v,total.d,total.dd,deviation.v,deviation.d,deviation.dd,ell,ellp])
    return np.asarray(rows)
