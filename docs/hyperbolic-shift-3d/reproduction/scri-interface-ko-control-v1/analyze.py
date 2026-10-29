import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d'))
import scri_v6_direction as v6
import scri_v7_localization as v7
import scri_candidate_replay as c
import scri_directional_replay as prior
start=time.monotonic();lib=ROOT/'output/hyperbolic-core-shell/scri-v6-hybrid-interface-build-v1/candidate.dylib'
m=json.loads(lib.with_name('source.json').read_text())
assert prior.sha(lib)==m['library_sha256']
for name,sha in m['sources'].items():
 assert prior.sha(lib.parent/'source-tree'/name)==sha
 assert hashlib.sha256(subprocess.check_output(['git','show',v7.FROZEN+':'+name])).hexdigest()==sha
x,meta6=v6.checkpoint(ROOT/'output/hyperbolic-core-shell/aurora/stability-donor-seven-minkowski-v1/output/minkowski.checkpoint')
y,meta7=v7.checkpoint(ROOT/'output/hyperbolic-core-shell/aurora/stability-inner-transition-minkowski-v1/output/minkowski.checkpoint',ROOT/'output/hyperbolic-core-shell/aurora-staging/inner-transition-duration-analysis/identity.json')
x/=prior.norm(x);y/=prior.norm(y)
a=c.Probe(lib,16,2,start+350);b=c.Probe(lib,16,4,start+350)
def action(p,q,eps):
 scale=eps/np.max(np.abs(q));return (p.evaluate(scale*q)-p.evaluate(-scale*q))/(2*scale)
import ctypes
newlib=ROOT/'output/hyperbolic-core-shell/scri-v7-ko-control-build-v1/candidate.dylib'
newmanifest=json.loads(newlib.with_name('source.json').read_text());assert prior.sha(newlib)==newmanifest['library_sha256']
changed=[]
for name,sha in newmanifest['sources'].items():
 assert prior.sha(newlib.parent/'source-tree'/name)==sha
 if sha!=m['sources'][name]:changed.append(name)
assert changed==['tst/hyperbolic_shift_3d/scri_candidate_probe.cpp']
strong=c.Probe(newlib,16,4,start+350)
setter=strong.lib.scri_candidate_bulk_ko_control;setter.argtypes=[ctypes.c_void_p,ctypes.c_double];setter.restype=ctypes.c_int
try:
 parity=0
 for vector in [x,y]:
  scale=1e-5/np.max(np.abs(vector))
  for sign in [-1,1]:
   assert c.exact(b.evaluate(sign*scale*vector),strong.evaluate(sign*scale*vector));parity+=1
 assert setter(strong.handle,0)==1 and setter(strong.handle,float('nan'))==1 and setter(None,1)==1
 assert setter(strong.handle,1)==0
 raw=[x,action(a,x,1e-5),y,action(b,y,1e-5)];basis=[]
 for z in raw:
  q=z.copy()
  for repeat in range(2):
   for v in basis:q-=prior.dot(v,q)*v
  basis.append(q/prior.norm(q))
 rows=[]
 for eps in [1e-5,1e-6]:
  row=dict(amplitude=eps)
  for label,p in [('baseline_KO_0_5',b),('doubled_KO_1',strong)]:
   W=[action(p,q,eps) for q in basis];H=np.array([[prior.dot(q,w) for w in W] for q in basis]);values,vectors=np.linalg.eig(H);ritz=[]
   for j,z in enumerate(values):
    coef=vectors[:,j];u=sum(q*c for q,c in zip(basis,coef));ju=sum(w*c for w,c in zip(W,coef));r=prior.norm(ju-z*u)
    ritz.append(dict(real=float(z.real),imag=float(z.imag),absolute_residual=r/prior.norm(u),relative_residual=r/prior.norm(ju)))
   row[label]=dict(matrix=H.tolist(),ritz=ritz)
  rows.append(row)
 assert setter(strong.handle,.5)==0
 z=x*1e-5/np.max(np.abs(x));assert c.exact(b.evaluate(z),strong.evaluate(z))
 old=json.loads((ROOT/'output/hyperbolic-core-shell/aurora-staging/interface-span-analysis-v2/result.json').read_text())
 replay=max(float(np.max(np.abs(np.array(row['baseline_KO_0_5']['matrix'])-np.array(ref['V7']['matrix'])))) for row,ref in zip(rows,old['rows']));assert replay<1e-7
 result=dict(scope='Read-only existing bulk KO coefficient .5 to1 on the observed V6/V7 joint span; no evolution, timestep or production change and no global stability conclusion',rows=rows,baseline_full_array_bitwise_parity_evaluations=parity+1,old_V7_matrix_max_difference=replay,invalid_coefficient_rejections_passed=True,library_sha256=prior.sha(newlib),source_manifest_sha256=prior.sha(newlib.with_name('source.json')),frozen_library_sha256=prior.sha(lib),changed_sources=changed,script_sha256=prior.sha(__file__),imports={str(Path(z.__file__).relative_to(ROOT)):prior.sha(z.__file__) for z in [v6,v7,c,prior]},checkpoints=[meta6,meta7],calls=[a.calls,b.calls,strong.calls],seconds=time.monotonic()-start)
 Path(sys.argv[1]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(seconds=result['seconds'],doubled=rows[0]['doubled_KO_1']['ritz'])))
finally:a.close();b.close();strong.close()
