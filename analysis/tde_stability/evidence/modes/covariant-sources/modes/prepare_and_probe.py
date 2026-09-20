from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[1]/'mode-analysis'
sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE,profile,discrepancy
BIN=ROOT.parent/'athena-covariant-modehook'
assert hashlib.sha256(BIN.read_bytes()).hexdigest()=='ef1a5473e3bfb5722240cf96a65284a6a5a78c8987509d6de0719a572f84dad1'
text=(OLD/'input.athinput').read_text().replace('<z4c>','<z4c>\nccz4_covariant_sources = false\ndamp_lapse_scaled = false').replace('cfl_number = 0.3','cfl_number = 0.15')
INPUT=ROOT/'input.athinput';INPUT.write_text(text)
opts=[('covariant_alpha01',('z4c/ccz4_covariant_sources=true',)),('covariant_constant01',('z4c/ccz4_covariant_sources=true','z4c/damp_lapse_scaled=true')),('covariant_constant03',('z4c/ccz4_covariant_sources=true','z4c/damp_lapse_scaled=true','z4c/damp_kappa1=.3'))]
validation={};results=[]
v=np.fromfile(OLD/'arnoldi_s40_m32_eps0.001-mode0.bin').reshape(SHAPE)*1e-3
for name,overrides in opts:
 op=Operator(str(ROOT/name),binary=BIN,input_file=INPUT,dt=.0375,overrides=overrides)
 z=op.advance(steps=20,label='zero20')
 validation[name]={'zero20_allbitszero':bool(np.all(z.view(np.uint64)==0))}
 f=op.advance(v,1,label='composition_step1');f2=op.advance(f,1,label='composition_step2');direct=op.advance(v,2,label='composition_direct2')
 validation[name]['composition']=discrepancy(direct,f2)
 if not validation[name]['zero20_allbitszero'] or not validation[name]['composition']['bitwise_equal']:raise RuntimeError(validation[name])
(ROOT/'hook-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print('Hook validation PASS',flush=True)
for name,overrides in opts:
 op=Operator(str(ROOT/name),binary=BIN,input_file=INPUT,dt=.0375,overrides=overrides)
 for n in [0,1]:
  v=np.fromfile(OLD/f'arnoldi_s40_m32_eps0.001-mode{n}.bin').reshape(SHAPE)
  r=op.response(v,80,1e-3,label=f'oldmode{n}_eps0.001')
  low=op.response(v,80,3e-4,label=f'oldmode{n}_eps0.0003')
  rec={'case':name,'old_mode':n,'interval_M':3,'amplitude_convergence':discrepancy(r,low),'initial_profile':profile(v),'response_profile':profile(r)}
  for domain,sl in [('full',(...,)),('active',ACTIVE)]:
   gain=float(np.vdot(v[sl],r[sl])/np.vdot(v[sl],v[sl]))
   rec[domain]={'projected_gain':gain,'projected_log_rate_per_M':float(np.log(abs(gain))/3),'norm_gain':float(np.linalg.norm(r[sl])/np.linalg.norm(v[sl])),'relative_shape_change':float(np.linalg.norm(r[sl]-gain*v[sl])/np.linalg.norm(r[sl]))}
  r.tofile(ROOT/f'{name}-oldmode{n}-response.bin')
  results.append(rec)
  (ROOT/'old-mode-responses.json').write_text(json.dumps(results,indent=2)+'\n')
  print(json.dumps({k:v for k,v in rec.items() if k not in ['initial_profile','response_profile']}),flush=True)
(ROOT/'manifest.json').write_text(json.dumps({'binary':str(BIN),'sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'input':str(INPUT),'dt_M':.0375,'steps_per_response':80,'interval_M':3,'old_modes':[str(OLD/f'arnoldi_s40_m32_eps0.001-mode{n}.bin') for n in [0,1]],'scope':'Old-vector responses are not new eigenvalues. Single CPU process16^3block; no job changes.'},indent=2)+'\n')
