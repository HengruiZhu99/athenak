#!/usr/bin/env python3
"""Invariant horizon mass convergence on analytic Schwarzschild trumpets."""
import json
import numpy as np
from trumpet_data import CMCTrumpet
from horizon import apparent_horizons
records=[]
for mass in [.1,1/3]:
    initial=CMCTrumpet(mass);errors=[];linear=[]
    for n in [128,256,512]:
        r=np.linspace(.09*mass,1,n);_,_,_,meta=initial.data(r)
        chi=np.array([z['chi_total'] for z in meta]);R=np.array([z['R'] for z in meta]);lapse=np.array([z['physical_lapse'] for z in meta])
        H=R-initial.C/R**2
        with np.errstate(invalid='ignore'):expansion=2*(lapse+H)/R
        expansion[-1]=4. # The actual-scri limit.
        result=apparent_horizons(r,chi,expansion)
        assert result['apparent_horizon_count']==1 and result['apparent_horizon_interpolation']=='local-polynomial'
        errors.append(abs(result['apparent_horizon_mass']-mass));linear.append(abs(result['apparent_horizon_mass_linear']-mass))
    orders=np.log2(np.array(errors[:-1])/errors[1:])
    assert errors[-1]<1e-8 and errors[-1]<linear[-1]/100,(errors,linear)
    if errors[-1]>1e-12:assert min(orders)>3.5,orders
    records.append(dict(mass=mass,polynomial_mass_errors=errors,linear_mass_errors=linear,orders=orders.tolist(),
        scope='Read-only extraction accuracy; no numerical evolution or horizon-formation claim'))
print(json.dumps(records))
