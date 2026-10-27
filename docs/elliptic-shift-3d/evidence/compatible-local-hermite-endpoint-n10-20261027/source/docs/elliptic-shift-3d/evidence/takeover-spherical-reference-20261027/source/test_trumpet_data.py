#!/usr/bin/env python3
"""Independent native constraints, stationary ADM rates and trumpet asymptotics."""
import argparse,json
import numpy as np
from trumpet_data import CMCTrumpet
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args();records=[]
for mass in [.1,1/3]:
    initial=CMCTrumpet(mass);r=(np.arange(512)+.5)/511.5
    q,d,dd,meta=initial.data(r);other,*_=CMCTrumpet(mass,128).data(r)
    coordinate_change=float(np.max(abs(q-other)));assert coordinate_change<2e-11,coordinate_change
    b=np.array([z['beta'] for z in meta]);alpha=(1+r*r)/2+q[:,6];chi=np.array([z['chi_total'] for z in meta])
    db=b/r+1.5*alpha*q[:,2];ddb=1.5*((r+d[:,6])*q[:,2]+alpha*d[:,2])+1.5*alpha*q[:,2]/r
    R,S,diag=Kernel(a.library).parts(r,q,d,dd,b,db,ddb,total_chi=chi)
    constraints=np.max(abs(diag[:,:4]),axis=0);assert max(constraints)<1e-9,constraints
    geometric=R[:-1,:6]+S[:-1,:6]/((1-r[:-1]*r[:-1])/2)[:,None]
    stationary=float(np.max(abs(geometric)));assert stationary<2e-10,stationary
    source_errors=[]
    for form in ['symmetric','multiplicative']:
        # The restoring adapter's lapse row does not depend on chi. Its
        # general tensor rows use deviation chi; constraints were audited
        # independently above with exact total chi and zero restoring.
        lR,lS,_=Kernel(a.library).parts(r,q,d,dd,b,db,ddb,lapse_restoring=.7)
        if form=='multiplicative':lS[:,6]-=.7*q[:,6]**2
        G=initial.lapse_source(r,.7,form)
        rate=lR[:-1,6]+(lS[:-1,6]+G[:-1])/((1-r[:-1]**2)/2)
        source_errors.append(float(max(abs(rate))));assert source_errors[-1]<2e-10,source_errors
    mask=(r>.01)&(r<.98);mass_error=float(max(abs(diag[mask,10]-mass)));assert mass_error<2e-9,mass_error
    rr=np.array([1e-3,5e-4,2.5e-4]);rows=[initial.point(float(x))[3] for x in rr]
    areal=np.array([z['R'] for z in rows]);lapse=np.array([z['physical_lapse'] for z in rows])
    powers=np.log(lapse[:-1]/lapse[1:])/np.log(2)
    assert max(abs(powers-initial.gamma))<.01,(powers,initial.gamma)
    assert max(abs(areal/initial.R0-1))<.01 and min(lapse)>0
    assert min(b[:8])>0 # Correct future black-hole inner shift/outflow.
    yAH=1/(2*mass)
    rAH=float(np.exp(np.log1p(-yAH/initial.y0)/initial.gamma-initial.regular_integral(np.array([yAH]))[0]))
    aq,ad,add,am=initial.point(rAH);aa=(1+rAH*rAH)/2+aq[6]
    assert abs(am['R']-2*mass)<1e-12
    light_horizon=-am['beta']+aa*np.sqrt(am['chi_total'])
    assert abs(light_horizon)<1e-12,light_horizon
    records.append(dict(mass=mass,R0=initial.R0,C=initial.C,gamma=initial.gamma,
        constraints=constraints.tolist(),stationary_geometric_RHS=stationary,mass_error_bulk=mass_error,
        coordinate_quadrature_change=coordinate_change,inner_lapse_powers=powers.tolist(),
        analytic_lapse_source_native_stationarity=source_errors,
        inner_areal_radii=areal.tolist(),horizon_r=rAH,horizon_areal_radius=am['R'],horizon_outward_light_speed=light_horizon,
        scope='Already-trumpet geometry only. Live Minkowski-reference lapse may relax; no stationarity subtraction.'))
print(json.dumps(records))
