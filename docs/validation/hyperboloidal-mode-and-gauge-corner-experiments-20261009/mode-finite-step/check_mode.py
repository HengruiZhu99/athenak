"""Exact cached final-only RK3 action on exported approximate C0norm modes.

No eigensolve, evolution, physics change, or modification of prior evidence.
The cached matrix has its original finite-difference Jacobian approximation.
"""
from pathlib import Path
import hashlib, json, math, os, platform, struct, subprocess, time
import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'build-layer-research/boundary/full-tensor-propagator/full22-v2'
EXPORT=ROOT/'build-layer-research/continuum/discrete-mode-identification'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
PIN={
 EXPORT/'candidate-vectors.npz':'fc88b20d4953f5088aed97d04dce41ad0af039fc794a401a21cceb53280e1eee',
 EXPORT/'candidate-metadata.json':'e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1',
 OLD/'spatialnorm-projected-J20.npz':'767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6',
 OLD/'spatialnorm-cache0.0001-J22.npz':'1f7a5297ea5a6308a707222d558f75316ae9615a6046abd581ebd8e7a90b6e65',
 OLD/'spatialnorm-cache0.0001-metadata.json':'98b7b7f459cce564398769cff48545cb6e89eb3708d01b2018a81b3a393d4e26',
 OLD/'server-spatialnorm':'bc4f4c62e4fa3e286da4c19f11fa79ba73a8ed3658f4fa42646d7dca7bf5a943',
}
for p,h in PIN.items(): assert sha(p)==h,(str(p),sha(p),h)
inputs={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in PIN}
for name in ['spatialnorm-cache0.0001-lift.bin','spatialnorm-cache0.0001-restrict.bin','full22_server.cpp','projected_base.hpp','build-provenance.json','validate22.py','spatialnorm-cache0.0001-validation.json']:
 p=OLD/name; inputs[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
meta=json.loads((OLD/'spatialnorm-cache0.0001-metadata.json').read_text())
candidate_meta=json.loads((EXPORT/'candidate-metadata.json').read_text())
vectors=dict(np.load(EXPORT/'candidate-vectors.npz'))
A=load_npz(OLD/'spatialnorm-cache0.0001-J22.npz')
J=load_npz(OLD/'spatialnorm-projected-J20.npz')
N=meta['points']; d20=20*N; d22=22*N
assert N==1640 and A.shape==(d22,d22) and J.shape==(d20,d20)
L=np.fromfile(OLD/'spatialnorm-cache0.0001-lift.bin',dtype='<f8').reshape(N,22,20)
P=np.fromfile(OLD/'spatialnorm-cache0.0001-restrict.bin',dtype='<f8').reshape(N,20,22)
def lift(v): return np.einsum('pij,pj->pi',L,v.reshape(N,20)).ravel()
def restrict(v): return np.einsum('pij,pj->pi',P,v.reshape(N,22)).ravel()
def pair(z): return [float(z.real),float(z.imag)]
def rayleigh(v,y): return np.vdot(v,y)/np.vdot(v,v)
def modal(v,delta,dt):
 dm=rayleigh(v,delta); mu=1+dm
 residual=delta-dm*v
 return {'mu':pair(mu),'abs_mu':float(abs(mu)),'mu_minus_one':pair(dm),
  'effective_growth_log_abs_mu_over_dt':float(np.log1p(dm).real/dt),
  'phase_radians':float(np.log1p(dm).imag),'effective_frequency_phase_over_dt':float(np.log1p(dm).imag/dt),
  'residual_state_per_unit_input_l2':float(np.linalg.norm(residual)/np.linalg.norm(v)),
  'residual_state_per_unit_input_linf':float(np.max(np.abs(residual))/np.linalg.norm(v)),
  'residual_state_per_unit_input_per_dt':float(np.linalg.norm(residual)/np.linalg.norm(v)/dt)}

started=time.monotonic(); dt0=.03*meta['min_omega']; saved={}; rows=[]
for cm in candidate_meta['candidates']:
 key=cm['key']; v=vectors[key]; assert v.shape==(d20,)
 assert abs(np.linalg.norm(v)-1)<2e-14
 lam=complex(*cm['lambda']); x=lift(v); x0=restrict(x); k1=A@x; k2=A@k1; k3=A@k2
 j1=J@v; j2=J@j1; j3=J@j2
 gen=J@v-lam*v
 row={'key':key,'lambda':pair(lam),'input_norm':float(np.linalg.norm(v)),
  'recomputed_generator_residual_state_per_time':float(np.linalg.norm(gen)),
  'reported_generator_residual':cm['actual_J_residual_generator_units'],
  'generator_rayleigh_lambda':pair(rayleigh(v,j1)),
  'P_J22_L_vs_J20_action_l2':float(np.linalg.norm(restrict(k1)-j1)),
  'P_L_identity_action_l2':float(np.linalg.norm(x0-v)),
  'normal_J22_L_norm':float(np.linalg.norm(k1-lift(restrict(k1)))),
  'normal_feedback_second_coefficient_l2':float(np.linalg.norm(restrict(k2)-j2)),
  'normal_feedback_third_coefficient_l2':float(np.linalg.norm(restrict(k3)-j3)),
  'steps':[]}
 for fac in [1.,.5,.25]:
  dt=dt0*fac
  delta=x0-v+dt*restrict(k1)+dt*dt/2*restrict(k2)+dt**3/6*restrict(k3)
  delta20=dt*j1+dt*dt/2*j2+dt**3/6*j3
  state=v+delta; saved[f'{key}_native_dt{fac}']=state
  state_direct=restrict(x+dt*k1+dt*dt/2*k2+dt**3/6*k3)
  m=modal(v,delta,dt); m.update({'dt':dt,'factor_nominal_dt':fac,
   'increment_l2':float(np.linalg.norm(delta)),
   'polynomial_reassociation_l2':float(np.linalg.norm(state-state_direct)),
   'state_norm_amplification':float(np.linalg.norm(state)/np.linalg.norm(v)),
   'residual_against_exp_dt_lambda_state':float(np.linalg.norm(delta-np.expm1(dt*lam)*v)),
   'residual_against_R3_dt_lambda_state':float(np.linalg.norm(delta-(dt*lam+(dt*lam)**2/2+(dt*lam)**3/6)*v)),
   'scalar_R3_mu':pair(1+dt*lam+(dt*lam)**2/2+(dt*lam)**3/6),
   'scalar_exp_mu':pair(np.exp(dt*lam)),
   'native22_vs_projected20_state_l2':float(np.linalg.norm(delta-delta20)),
   'native22_vs_projected20_per_dt':float(np.linalg.norm(delta-delta20)/dt),
   'native22_minus_projected20_scalar_mu':pair(rayleigh(v,delta-delta20)),
   'projected20':modal(v,delta20,dt)})
  row['steps'].append(m)
 rows.append(row)
 print(key,'generator residual',row['recomputed_generator_residual_state_per_time'],
       'steps',[(q['factor_nominal_dt'],q['effective_growth_log_abs_mu_over_dt'],q['residual_state_per_unit_input_l2']) for q in row['steps']],flush=True)

# Independent actual nonlinear native one-step centered derivative for candidate0.
# Fresh startup exports are kept separate; no cached/frozen prefix is overwritten.
fresh=HERE/'native-step-cache'; stderr=HERE/'native-step-server.stderr'
cmd=[str(OLD/'server-spatialnorm'),'16','2.2','0.0001',str(fresh)]
with stderr.open('w') as err:
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
 line=proc.stdout.readline()
 if not line: raise RuntimeError(stderr.read_text())
 direct_meta=json.loads(line); assert direct_meta==meta or direct_meta['xyz_omega_volume_ginv_chi']==meta['xyz_omega_volume_ginv_chi']
 (HERE/'native-step-metadata.json').write_text(json.dumps(direct_meta,indent=2)+'\n')
 def apply_real(v,eps,dt):
  proc.stdin.write(b's'+struct.pack('dd',eps,dt)+np.asarray(v,dtype='<f8').tobytes());proc.stdin.flush()
  b=bytearray()
  while len(b)<8*d20:
   chunk=proc.stdout.read(8*d20-len(b))
   if not chunk:raise RuntimeError('native step terminated: '+stderr.read_text())
   b.extend(chunk)
  return np.frombuffer(b,dtype='<f8').copy()
 native=[]; v=vectors['candidate0']
 for step in rows[0]['steps']:
  dt=step['dt']; fast=saved[f"candidate0_native_dt{step['factor_nominal_dt']}"]
  item={'dt':dt,'factor_nominal_dt':step['factor_nominal_dt'],'eps_sweep':[]}
  for eps in [1e-3,1e-4,1e-5]:
   y=apply_real(v.real,eps,dt)+1j*apply_real(v.imag,eps,dt)
   error=np.linalg.norm(y-fast)
   q=modal(v,y-v,dt);q.update({'eps_maxfree_component':eps,'state_l2_vs_cached_exact_map':float(error),
    'state_linf_vs_cached_exact_map':float(np.max(np.abs(y-fast))),'state_error_vs_cache_per_dt':float(error/dt)})
   item['eps_sweep'].append(q)
   saved[f"candidate0_fd_dt{step['factor_nominal_dt']}_eps{eps}"]=y
  native.append(item)
 proc.stdin.close(); status=proc.wait(); assert status==0
export_checks={}
for suffix in ['data','indices','indptr','lift','restrict']:
 old=OLD/f'spatialnorm-cache0.0001-{suffix}.bin';new=Path(str(fresh)+f'-{suffix}.bin')
 export_checks[suffix]={'old_sha256':sha(old),'fresh_sha256':sha(new),'byte_identical':sha(old)==sha(new)}
 assert export_checks[suffix]['byte_identical']
np.savez_compressed(HERE/'mode-actions.npz',**saved)
result={'scope':'Approximate-mode action of exact cached final-only native RK3 map; not eigenvalue certification or nonlinear/continuum stability.',
 'map':'P_ref (I + dt J22 + dt^2 J22^2/2 + dt^3 J22^3/6) Lift_ref',
 'continuous_generator':'J20=P_ref J22 Lift_ref; algebraic projection at every infinitesimal RHS',
 'modal_scalar':'mu=1+v* (B_dt v-v)/(v*v); phase uses principal log1p(mu-1)',
 'residual_units':'Euclidean state norm per unit Euclidean input; divided by dt is separately reported; no normalization by ||J||.',
 'direct_oracle_scope':'Centered finite-amplitude derivative of actual nonlinear final-only SSPRK3, real/imag separately; includes finite-difference and cached-J approximation errors.',
 'nominal_pole_dt':dt0,'omega_min':meta['min_omega'],'spacing':meta['spacing'],'points':N,
 'inputs':inputs,'candidate_metadata_status':candidate_meta['status'],'candidates':rows,
 'actual_native_one_step_candidate0':native,'fresh_cache_byte_identity':export_checks,
 'native_command':cmd,'native_server_exit':status,'native_stderr_sha256':sha(stderr),
 'source_sha256':sha(Path(__file__)),'seconds':time.monotonic()-started,
 'head_at_launch':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy_version},
 'environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','PYTHONPATH']}}
for p,h in PIN.items(): assert sha(p)==h
(HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('DONE',result['seconds'],flush=True)
