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
try:
 # Fixed before seeing results: union of the two observed direction/RHS planes,
 # compare endpoints and their arithmetic midpoint, with no weight scan.
 raw=[x,action(a,x,1e-5),y,action(b,y,1e-5)];basis=[];conditioning=[]
 for z in raw:
  q=z.copy();original=prior.norm(q)
  for repeat in range(2):
   for v in basis:q-=prior.dot(v,q)*v
  remainder=prior.norm(q);conditioning.append(remainder/original)
  if remainder/original<1e-10:raise ValueError('Dependent observed direction span')
  basis.append(q/remainder)
 Q=np.column_stack(basis);rows=[]
 for eps in [1e-5,1e-6]:
  A=np.column_stack([action(a,q,eps) for q in basis]);B=np.column_stack([action(b,q,eps) for q in basis]);row=dict(amplitude=eps)
  for label,W in [('V6',A),('V7',B),('midpoint_tangent_operator',(A+B)/2)]:
   H=np.array([[prior.dot(q,W[:,j]) for j in range(4)] for q in basis]);R=np.column_stack([W[:,j]-sum(q*H[i,j] for i,q in enumerate(basis)) for j in range(4)]);values,vectors=np.linalg.eig(H);res=[]
   for j in range(4):
    w=vectors[:,j];res.append(float(prior.norm(sum(R[:,k]*w[k] for k in range(4)))/prior.norm(sum(W[:,k]*w[k] for k in range(4)))))
   row[label]=dict(matrix=H.tolist(),ritz=[dict(real=float(z.real),imag=float(z.imag),relative_residual=res[i]) for i,z in enumerate(values)],column_relative_residuals=(np.linalg.norm(R,axis=0)/np.linalg.norm(W,axis=0)).tolist())
  rows.append(row)
 out=dict(scope='Read-only four-dimensional union of observed V6/V7 tangent planes; midpoint is an operator diagnostic, not an implemented interpolation or evolution. No global stability claim.',rows=rows,seed_inner_product=prior.dot(x,y),basis_remainder_ratios=conditioning,orthogonality_error=float(np.max(np.abs(np.array([[prior.dot(q,r) for q in basis] for r in basis])-np.eye(4)))),library_sha256=prior.sha(lib),source_manifest_sha256=prior.sha(lib.with_name('source.json')),script_sha256=prior.sha(__file__),imports={str(Path(z.__file__).relative_to(ROOT)):prior.sha(z.__file__) for z in [v6,v7,c,prior]},checkpoints=[meta6,meta7],calls=[a.calls,b.calls],seconds=time.monotonic()-start)
 Path(sys.argv[1]).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(seconds=out['seconds'],midpoint=rows[0]['midpoint_tangent_operator']['ritz'])))
finally:a.close();b.close()
