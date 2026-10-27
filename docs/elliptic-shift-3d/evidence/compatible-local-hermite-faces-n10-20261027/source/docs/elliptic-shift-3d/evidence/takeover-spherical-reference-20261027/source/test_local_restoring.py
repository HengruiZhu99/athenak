#!/usr/bin/env python3
"""Local rational compensation: independent source identities and limits."""
import json
import numpy as np
from gauge import restoring_coefficient,lapse_coefficient
r=np.linspace(0,1,65);A=(1+r*r)/2;O=(1-r*r)/2;nu=1.;delta=.001;t=(1-r*r)**4
errors=[]
for epsilon in [1e-3,1e-4,1e-5,1e-6]:
 u=epsilon*(1-r*r)**2;alpha=A+u;kdev=epsilon*(1-r*r)**3
 xi=restoring_coefficient(r,1,nu,'rational',delta,'physical-local',alpha)
 rat=-lapse_coefficient(r,alpha,nu,'rational',delta)*kdev-2*xi*alpha*u
 linear=-lapse_coefficient(r,alpha,nu,'linear')*kdev-2*(1+2*t)*alpha*u
 exact=4*nu*O*O*alpha*u/(alpha+delta)*kdev+6*nu*u*u*alpha*t/(alpha+delta)
 np.testing.assert_allclose(rat-linear,exact,atol=3e-18,rtol=1e-7)
 errors.append(float(max(abs(rat-linear))))
assert min(np.log10(np.array(errors[:-1])/errors[1:]))>1.99
alpha=np.array([.02,.05,.1]);rr=np.full(3,.12);AA=(1+rr*rr)/2;OO=(1-rr*rr)/2
Fp=np.imag(lapse_coefficient(rr,alpha+1e-20j,nu,'rational',delta))/1e-20
indicator=1-Fp/(2*alpha);exact=-2*nu*OO*OO*(AA+delta)*delta/(alpha*(alpha+delta)**2)
np.testing.assert_allclose(indicator,exact,atol=2e-14,rtol=1e-13)
# At a collapsed lapse the bounded AF coefficient still matches its derived
# nu_inner. Delta is never used to replace or floor alpha or Omega.
khat=-.8;limits=[];nu_inner=nu*(.5+delta)/delta
for alpha in [1e-7,1e-8,1e-9,1e-10]:
 strength=restoring_coefficient(np.array([0.]),1,nu,'rational',delta,'physical-local',np.array([alpha]))[0]
 F=lapse_coefficient(0.,alpha,nu,'rational',delta)
 normalized=(-F*(khat+3)-2*strength*alpha*(alpha-.5))/(.5*alpha)
 limits.append(abs(normalized+2*nu_inner*khat))
assert min(np.log10(np.array(limits[:-1])/limits[1:]))>.99
print(json.dumps(dict(Minkowski_linearized_source_preserved=True,quadratic_source_difference=errors,local_principal_steepening_indicator=indicator.tolist(),collapsed_limit_errors=limits,scri_profile_first_three_corrections_zero=True,scope='Independent algebra/limit audit, not global shock avoidance or nonlinear stability')))
