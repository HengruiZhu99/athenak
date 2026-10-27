"""Native kappa oracle, leading AF constraint mode, scri and lapse invariance."""
import argparse,json
from pathlib import Path
import numpy as np
from constraint_damping import correction
from evolution import Evolution
from kernel import Kernel
from puncture import PunctureEvolution
from elliptic import connection
from telegraph import window

p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
k=Kernel(a.library);rng=np.random.default_rng(20261027);errors=[]
for radius,alpha in [(.03,.002),(.2,.3),(.7,.8),(.85,.9),(.98,1.)]:
 r=np.array([radius]);A=(1+r*r)/2;q=rng.normal(size=(1,8))*.01;d=rng.normal(size=(1,8))*.02;dd=rng.normal(size=(1,8))*.01;q[:,6]=alpha-A
 beta=np.array([-.2*radius]);db=np.array([-.2]);ddb=np.array([.01]);w,_=window(r,width=.24,inner_profile='af-regular')
 effective=(1-w)*4+w*8*A/alpha
 R0,S0,_=k.parts(r,q,d,dd,beta,db,ddb,kappa1=4)
 R1,S1,_=k.parts(r,q,d,dd,beta,db,ddb,kappa1=float(effective[0]))
 got=correction(r,np.array([alpha]),q[:,4],q[:,5]-connection(r,1+q[:,1],d[:,1]),w,8)
 assert np.array_equal(R0,R1)
 errors.append(float(np.max(abs(S1-S0-got))))
assert max(errors)<1e-13
# A smooth AF asymptotic state: V0=4*g1 and delta=Lambda0-V0.
# Native calls approach delta_dot=(2*b1-kappa_inner)*delta.
r=np.logspace(-3,-6,4);q=np.zeros((4,8));d=q.copy();dd=q.copy();O=(1-r*r)/2
g1=.2;b1=.7;b2=.03;lam0=.05
q[:,0]=2*r**4;d[:,0]=8*r**3;dd[:,0]=24*r*r
q[:,1]=g1*r+.1*r*r;d[:,1]=g1+.2*r;dd[:,1]=.2
for f,c in [(2,.3),(3,.4),(4,.1)]:q[:,f]=c*r*r;d[:,f]=2*c*r;dd[:,f]=2*c
q[:,5]=lam0+.01*r;d[:,5]=.01;q[:,6]=.7*r*r;d[:,6]=.4*r;dd[:,6]=.4
beta=b1*r+b2*r*r;db=b1+2*b2*r;ddb=np.full(4,2*b2)
f=k.lib.spherical_parts_positive_totals;f.argtypes=k.lib.spherical_parts.argtypes
R=np.zeros((4,8));S=R.copy();diag=np.zeros((4,12));assert f(4,r,q,d,dd,beta,db,ddb,4.,-.99,1,1.,R,S,diag)==0
V=connection(r,1+q[:,1],d[:,1]);S+=correction(r,q[:,6],q[:,4],q[:,5]-V,np.ones(4),8)
rate=R+S/O[:,None];V0dot=4*(b1*g1+4*b2/3)
expected=(2*b1-8)*(lam0-4*g1);af_errors=abs(rate[:,5]-V0dot-expected)
assert np.all(np.diff(af_errors)<0) and af_errors[-1]<1e-5
e0=PunctureEvolution(129,a.library,grid_stretch=0,shift_gauge='moving-puncture',shift_inner_stencil='odd')
e1=PunctureEvolution(129,a.library,grid_stretch=0,shift_gauge='moving-puncture',shift_inner_stencil='odd',inner_z4_damping=8)
q=e0.initial();q[20:25,4]+=.002;q[25,5]+=.01;r0=e0.rhs(q);r1=e1.rhs(q)
assert np.array_equal(r0[:,[0,1,2,6,7,8]],r1[:,[0,1,2,6,7,8]])
assert np.array_equal(r0[e0.r>=.8],r1[e0.r>=.8])
assert np.array_equal(q[1:,5],e1.materialize(q)[1:,5])
ev=Evolution(65,a.library,restoring=4,shift_gauge='moving-puncture',inner_z4_damping=8);z=np.zeros((65,10));equilibrium=float(np.max(abs(ev.rhs(z))));assert equilibrium<2e-9
z[:,0]=.001*ev.r**2;z[:,1]=-.001*ev.r**2;z[:,6]=.002*ev.r**2;ev.step(z,1e-8)
assert ev.stats['max_relative_null']<2e-15 and ev.stats['max_driver_boundary_defect']<1e-12
result=dict(native_kappa_difference_errors=errors,AF_connection_error_rate=expected,AF_limit_errors=af_errors.tolist(),Minkowski_max_RHS=equilibrium,stage_null=ev.stats['max_relative_null'],unchanged_lapse_telegraph_shift_rates_at_same_state=True,outer_collar_bitwise_unchanged=True,independent_connection_retained=True,
 scope='Interior lower-order Z4 source control, exactly equivalent to a lapse-dependent native kappa1. It is not a bulk constraint projection and is not yet duration-qualified. All old controls keep inner_z4_damping=0.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
