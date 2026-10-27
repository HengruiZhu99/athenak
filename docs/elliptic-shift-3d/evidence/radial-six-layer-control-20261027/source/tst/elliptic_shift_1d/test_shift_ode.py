#!/usr/bin/env python3
"""Independent dense solve, polynomial and continuum controls for banded shift."""
import argparse,json
from types import SimpleNamespace
import numpy as np
from numerics import RadialOperators
from shift_ode import BandedShift
from evolve import Evolution
from kernel import Kernel
from fast import FastEngine
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args()
records=[]
for grid in ['half','nodal','cutoff']:
 errors=[]
 for n in [32,64,128,256,512]:
    r=(np.arange(n)+.5)/(n-.5) if grid=='half' else np.linspace(0 if grid=='nodal' else .03,1,n)
    op=RadialOperators(r,puncture=grid=='cutoff');solver=BandedShift(op)
    for degree in [0,2,4,6]:
        source=np.zeros_like(r) if degree==0 else degree*r**(degree-1)
        expected=r**degree;got=solver.solve(source,1.)
        np.testing.assert_allclose(got,expected,atol=6e-12,rtol=0)
    source=-.06*r*np.sin(3*r*r);exact=.01*(np.cos(3*r*r)-np.cos(3));got=solver.solve(source,0.)
    errors.append(float(max(abs(got-exact))))
    assert solver.last_backward_error<3e-16
    if n==32:
        # Build dense D through the public derivative routine, not band storage.
        dense,_=op.derivatives(np.eye(n),np.ones(n));skip=solver.skip
        dense=np.vstack([dense[skip:n-1+skip],np.eye(n)[-1]])
        rhs=np.r_[source[skip:n-1+skip],0.]
        independent=np.linalg.solve(dense,rhs)
        np.testing.assert_allclose(got,independent,atol=2e-13,rtol=0)
    records.append(dict(grid=grid,n=n,error=errors[-1],backward_error=solver.last_backward_error,
                        minimum_scaled_pivot=solver.minimum_pivot,factor_growth=solver.factor_growth))
 assert errors[0]/errors[1]>20 and errors[1]/errors[2]>20,errors
 assert errors[-1]<1e-13,errors
for n in [32,64,128,256,512]:
 for grid in ['half','nodal','cutoff']:
    r=(np.arange(n)+.5)/(n-.5) if grid=='half' else np.linspace(0 if grid=='nodal' else .03,1,n)
    op=RadialOperators(r,puncture=grid=='cutoff');solver=BandedShift(op,bias='right')
    source=-.06*r*np.sin(3*r*r);exact=.01*(np.cos(3*r*r)-np.cos(3));got=solver.solve(source,0.)
    if n==32:
        dense=op.advection_derivative(np.eye(n),np.ones(n),np.ones(n));dense[-1]=np.eye(n)[-1]
        rhs=source.copy();rhs[-1]=0.
        np.testing.assert_allclose(got,np.linalg.solve(dense,rhs),atol=2e-13,rtol=0)
    if n==512:assert max(abs(got-exact))<1e-13
    records.append(dict(grid=grid,bias='right',n=n,error=float(max(abs(got-exact))),backward_error=solver.last_backward_error))
# Compare actual nonlinear tensor rates and compiled integration. The free
# constraints are nonzero and must survive this gauge solve independently.
for origin in [False,True]:
 opts=SimpleNamespace(n=65,origin_grid=origin,puncture=False,origin_variables=False,backend='spherical-g1',
    width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,eta=0.,theta='appendix-c',
    slicing=1.,lapse_restoring=1.,endpoint='analytic-g1',gauge_corner_sat=0.,advection='centered',
    scri_sat=4.,regularity_sat=8.,origin_ko=.5,origin_ko_radius=.4,
    metric_variable='log',lapse_variable='log-reference',elliptic_method='discrete-ode',elliptic_refinements=2)
 ev=Evolution(opts);r=ev.r;q=np.zeros((65,8));f=(1-r*r)**4
 for field,amplitude,power in [(0,.002,0),(2,.001,2),(3,.0015,0),(4,.0001,0),(5,.0002,1),(6,.002,0)]:q[:,field]=amplitude*r**power*f
 rate=ev.rhs(q,accepted=True);R,S,c,beta=ev.parts(q);geom=ev.geometry_state;d=ev.geometry_first;dd=ev.geometry_second
 mask=(r>0)&(r<1);alpha=ev.total_alpha;chi=ev.total_chi
 db=-1+np.divide(ev.shift_deviation,r,out=np.full_like(r,ev.origin_shift_slope+1),where=r>0)+1.5*alpha*geom[:,2]
 ar=np.divide(geom[:,2],r,out=np.zeros_like(r),where=r>0)
 ddb=1.5*((r+d[:,6])*geom[:,2]+alpha*d[:,2])+1.5*alpha*ar
 nativeR,nativeS,diag=Kernel(a.library).parts(r[mask],geom[mask],d[mask],dd[mask],beta[mask],db[mask],ddb[mask],lapse_restoring=1.)
 # Canceled and native R/S decompositions differ by regular Omega multiples.
 # Compare their physical sum, without the separately applied KO/penalties.
 np.testing.assert_allclose(R[mask]+S[mask]/ev.omega[mask,None],nativeR+nativeS/ev.omega[mask,None],atol=3e-12,rtol=0)
 np.testing.assert_allclose(c[mask,:4],diag[:,:4],atol=3e-12,rtol=0)
 assert all(max(abs(c[:,i]))>0 for i in range(4))
 engine=FastEngine(ev,8,a.fast_library);got,_=engine.rhs(q);np.testing.assert_allclose(got,rate,atol=2e-11,rtol=0)
 for passes,bias in [(0,'centered'),(4,'centered'),(0,'right'),(2,'right'),(4,'right')]:
    ev.op.shift_ode_solver=BandedShift(ev.op,2,bias);reference_rate=ev.rhs(q)
    ev.op.shift_ode_solver=BandedShift(ev.op,passes,bias);trial=ev.rhs(q)
    # Arithmetic refinement preserves the chosen discrete operator. Changing
    # its bias is a discretization change and has a real truncation difference.
    np.testing.assert_allclose(trial,reference_rate,atol=2e-11,rtol=0)
    other=FastEngine(ev,8,a.fast_library);got,_=other.rhs(q)
    np.testing.assert_allclose(got,trial,atol=2e-11,rtol=0)
    state=q.copy();compiled=q.copy();dt=.01*ev.h
    for _ in range(20):
        k1=ev.rhs(state);k2=ev.rhs(state+dt*k1/2);k3=ev.rhs(state+dt*k2/2);k4=ev.rhs(state+dt*k3)
        state+=dt*(k1+2*k2+2*k3+k4)/6
    done,_,failed=other.advance(compiled,20,dt);assert done==20 and not failed
    np.testing.assert_allclose(compiled,state,atol=2e-13,rtol=0)
    assert other.diagnostics()['max_elliptic_discrete_residual']<1e-12
    other.close()
 engine.close()
print(json.dumps(dict(polynomial_dense_and_native_controls=True,independent_constraints_retained=True,records=records)))
