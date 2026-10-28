"""Bulk Gamma-driver identity, exact boundary traces, symbol and grid audits."""
import argparse,json
from pathlib import Path
import numpy as np
from evolution import Evolution
from puncture import PunctureEvolution
from gamma_driver import parameters,collar
from principal import symbol,speeds,eigenspace_audit
from gauge import lapse_coefficient

p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=np.linspace(0,1,1001);mu0=.1+r*r;mu,xi=parameters(r,mu0,4.)
assert np.all(mu[r<=.8]==1) and np.all(xi[r<=.8]==0)
assert np.array_equal(mu[r>=.9],mu0[r>=.9]) and np.all(xi[r>=.9]==4)
e=Evolution(65,a.library,restoring=4,shift_gauge='moving-puncture')
z=np.zeros((65,10));equilibrium=float(np.max(abs(e.rhs(z))));assert equilibrium<2e-9
stage_checks=[]
for amp in [-.01,.001,.01]:
 e=Evolution(33,a.library,restoring=4,shift_gauge='moving-puncture')
 q=np.zeros((33,10));q[:,0]=amp*e.r**2;q[:,1]=.4*amp*e.r**2;q[:,6]=.3*amp*e.r**2
 e.step(q,1e-8)
 assert e.stats['max_relative_null']<2e-15 and e.stats['max_driver_boundary_defect']<1e-12
 stage_checks.append(dict(amplitude=amp,relative_null=e.stats['max_relative_null'],driver_defect=e.stats['max_driver_boundary_defect']))
# At the same state, the physical Einstein/telegraph/lapse equations are identical.
old=PunctureEvolution(129,a.library,grid_stretch=0.)
new=PunctureEvolution(129,a.library,grid_stretch=0.,shift_gauge='moving-puncture')
q=old.initial();q[35:55,1]+=.0001;q[47,5]+=.0002
ro=old.rhs(q);rn=new.rhs(q)
assert np.array_equal(ro[:,:7],rn[:,:7]) and np.array_equal(ro[:,8],rn[:,8])
assert new.materialize(q)[47,5]==q[47,5]
# Independent bulk beta identity with all numerical stabilization disabled.
raw=PunctureEvolution(129,a.library,ko=0,grid_stretch=0.,shift_gauge='moving-puncture')
q=raw.initial();q[:,9]=.002*raw.r
rr=raw.rhs(q);chart=q[:,:8].copy();chart[1:,6]=np.log(q[1:,6]/raw.chart.refs['alpha'][1:,0])
beta=q[:,7]-raw.r;adv=raw.op.advection_derivative(chart,np.ones(8),beta)
bulk=(raw.r>0)&(raw.r<=.8)
expected=beta*(-1+adv[:,7])-raw.r+q[:,9]
assert np.array_equal(rr[bulk,7],expected[bulk])
assert raw.materialize(q)[0,9]==0 and rr[0,9]==0
center={}
for m in [.75,1.]:
 P=symbol(.5,1,1,.75,1,1,.005,.02,m)
 center[str(m)]=eigenspace_audit(P,-speeds(.5,1,1,.75,1,1,.005,.02,m))
assert not center['0.75']['complete'] and center['1.0']['complete']
scri=eigenspace_audit(symbol(1,1,1,1,1,0,.005,.02,1.5),-speeds(1,1,1,1,1,0,.005,.02,1.5))
assert scri['complete']
grid_records=[]
for stretch in [16.,0.]:
 ev=PunctureEvolution(257,a.library,grid_stretch=stretch,shift_gauge='moving-puncture')
 ev.rhs(ev.initial());v=ev.last
 F=lapse_coefficient(ev.r,v['alpha']);eta=1.0001*v['alpha']/(v['alpha']+1e-4*ev.A)
 gap=4*v['shift_mu']/3-v['chi']*(F+v['g']*eta*.25*ev.w**2)
 grid_records.append(dict(stretch=stretch,first_radius=ev.r[1],mapped_max_speed=ev.stats['max_speed'],initial_CFL_at_005=ev.stats['max_speed']*.05,initial_min_shift_lapse_squared_gap_times_g=float(min(gap[1:]))))
 assert np.all(gap[1:]>0)
result=dict(Minkowski_max_RHS=equilibrium,stage_checks=stage_checks,bulk_identity_exact=True,unchanged_native_lapse_telegraph_rates=True,independent_connection_retained=True,center_symbol=center,scri_symbol=scri,grid_records=grid_records,
 scope='Scalar frozen symbol and implementation identities; no nonlinear or uniform puncture hyperbolicity claim. Literal coefficient .75 is rejected for the retained lapse at the regular Minkowski center.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
