#!/usr/bin/env python3
"""Exact retained-state copy, causal rejection, native rates and strict restart.

The coarse parent is an implementation fixture with large constraint errors;
this test never admits it as a physical wormhole-to-trumpet experiment.
"""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from puncture_chart import PunctureChart
from fast import FastEngine
from kernel import Kernel
from trajectory import read_trajectory
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);p.add_argument('--output',type=Path);a=p.parse_args();temporary=None
if a.output is None:temporary=tempfile.TemporaryDirectory(prefix='elliptic-domain-transfer-');a.output=Path(temporary.name)/'runs'
a.output.mkdir(parents=True,exist_ok=False)
common=[sys.executable,str(Path(__file__).with_name('evolve.py')),'--library',a.library,'--fast-library',a.fast_library,'--backend','spherical-g1','--puncture','--puncture-chart','--initial-data','wormhole','--mass',str(1/3),'--lapse-restoring','1','--lapse-restoring-form','multiplicative','--lapse-restoring-profile','physical-log','--lapse-background-source','collapsed-inner','--slicing','.25','--advection','upwind','--scri-sat','4','--regularity-sat','8','--cfl','.05','--report-dt','2']
def run(name,extra,expect=True):
 with (a.output/(name+'.log')).open('w') as f:r=subprocess.run(common+extra+['--output',str(a.output/name)],stdout=f,stderr=subprocess.STDOUT)
 if expect:assert r.returncode==0,(name,(a.output/(name+'.log')).read_text())
 else:assert r.returncode!=0,name
 return json.loads((a.output/name/'manifest.json').read_text())
run('parent',['--n','129','--puncture-ko','.5','--t','4'])
source=a.output/'parent/profile-0003.npz';bad=a.output/'parent/profile-0002.npz';cut=4;dt=.05/128
base=['--n','125','--inner-radius',str(4/128),'--transfer-af-profile',str(source)]
full=run('full',base+['--t',str(4+40*dt)])
run('first',base+['--t',str(4+20*dt)])
run('resumed',base+['--t',str(4+40*dt),'--resume',str(a.output/'first/checkpoint.npz')])
with np.load(source,allow_pickle=False) as src,np.load(a.output/'full/profile-0001.npz',allow_pickle=False) as dst:
 assert np.array_equal(src['q'][cut:],dst['q']);assert np.array_equal(src['r'][cut:],dst['r'])
 state=dst['q'].copy()
with np.load(a.output/'full/checkpoint.npz',allow_pickle=False) as x,np.load(a.output/'resumed/checkpoint.npz',allow_pickle=False) as y:
 for key in ['q','q_physical','chi_total','alpha_total','r','t','step']:assert np.array_equal(x[key],y[key]),key
# The actual finite-cutoff left point uses the physical tensor equations,
# rather than the asymptotic coefficient ODE formerly at index zero.
opts=SimpleNamespace(**full['arguments']);ev=PunctureChart(opts);expected=ev.rhs(state,accepted=True)
engine=FastEngine(ev,8,a.fast_library);got,_=engine.rhs(state);engine.close()
np.testing.assert_allclose(got,expected,atol=2e-10,rtol=2e-11)
filtered_opts=SimpleNamespace(**vars(opts));filtered_opts.puncture_ko=.5
filtered=PunctureChart(filtered_opts);filtered_rate=filtered.rhs(state)
filtered_engine=FastEngine(filtered,8,a.fast_library);filtered_got,_=filtered_engine.rhs(state)
np.testing.assert_allclose(filtered_got,filtered_rate,atol=2e-10,rtol=2e-11);filtered_engine.close()
r=ev.r;geom=ev.geometry_state;d=ev.geometry_first;dd=ev.geometry_second;alpha=ev.total_alpha;chi=ev.total_chi
_,_,_,beta=ev.parts(state);db=beta/r+1.5*alpha*geom[:,2];ddb=1.5*((r+d[:,6])*geom[:,2]+alpha*d[:,2])+1.5*alpha*geom[:,2]/r
R,S,diag=Kernel(a.library).parts(r[:-1],geom[:-1],d[:-1],dd[:-1],beta[:-1],db[:-1],ddb[:-1],slicing=.25,total_chi=chi[:-1])
xi=1-.25*(1-r*r)**4;O=(1-r*r)/2;A=(1+r*r)/2
S[:,6]-=2*xi[:-1]*alpha[:-1]*geom[:-1,6];S[:,6]+=(1-r[:-1]**2)**4*r[:-1]**2*geom[:-1,6]/A[:-1]
native=R+S/O[:-1,None]
chart_d,_=ev.op.derivatives(state,np.ones(8));chart_adv=ev.op.advection_derivative(state,np.ones(8),beta)
centered=expected-beta[:,None]*(chart_adv-chart_d);centered[-1,6]=0
physical=centered.copy();physical[:,0]*=chi;physical[:,6]*=alpha
for field in [2,3,4]:physical[:,field]*=ev.refs['w'][:,0]
error=float(max(abs(physical[:-1]-native).ravel()));assert error<2e-10,error
assert all(max(abs(diag[:,field]))>0 for field in range(4))
assert abs(expected[0,0]-2*ev.b1)>1e-4
run('incoming',['--n','125','--inner-radius',str(4/128),'--transfer-af-profile',str(bad),'--t','4.01'],expect=False)
assert 'incoming' in (a.output/'incoming.log').read_text().lower()
assert json.loads((a.output/'incoming/manifest.json').read_text())['status']=='domain-transfer-rejected'
with np.load(bad,allow_pickle=False) as z:incoming=z['q'][cut:].copy()
try:ev.rhs(incoming)
except FloatingPointError as e:assert 'incoming' in str(e)
else:raise AssertionError('Python stage guard accepted incoming characteristics')
engine=FastEngine(ev,8,a.fast_library)
try:engine.rhs(incoming)
except FloatingPointError as e:assert 'incoming' in str(e)
else:raise AssertionError('Compiled stage guard accepted incoming characteristics')
engine.close()
try:read_trajectory(a.output/'full')
except ValueError as e:assert 'explicit' in str(e)
else:raise AssertionError('Generic analysis silently dropped the original AF prefix')
_,retained_rows,_,_=read_trajectory(a.output/'resumed',allow_domain_transfer=True)
assert retained_rows[0]['t']==4. and retained_rows[-1]['t']==4+40*dt
original=source.read_bytes()
try:
 source.write_bytes(original+b'tampered')
 try:read_trajectory(a.output/'full',allow_domain_transfer=True)
 except ValueError as e:assert 'hash changed' in str(e)
 else:raise AssertionError('Altered source profile accepted')
finally:source.write_bytes(original)
record=dict(generic_analysis_requires_explicit_domain_scope=True,source_profile_hash_checked=True,retained_stored_fields_and_grid_bitwise=True,strict_restart_bitwise=True,native_finite_left_point_error=error,
 compiled_Python_finite_cutoff_agreement=True,independent_constraints_retained=True,incoming_cutoff_rejected=True,Python_and_compiled_stage_guards_verified=True,
 parent_grid_points=129,retained_grid_points=125,receipt=full['domain_transfer_parent'],
 scope='Implementation fixture only. Coarse parent has large constraints and is not a scientifically admitted prefix or wormhole result.')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({key:value for key,value in record.items() if key!='receipt'}))
if temporary:temporary.cleanup()
