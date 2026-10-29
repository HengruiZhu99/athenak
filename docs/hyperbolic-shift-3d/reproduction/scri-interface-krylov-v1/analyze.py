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
 raw=[x,action(a,x,1e-5),y,action(b,y,1e-5)];Q=[]
 def orthogonal(z):
  q=z.copy()
  for repeat in range(2):
   for v in Q:q-=prior.dot(v,q)*v
  return q
 for z in raw:
  q=orthogonal(z);Q.append(q/prior.norm(q))
 W=[];max_basis=32
 for j in range(max_basis):
  if j>=len(Q):break
  w=(action(a,Q[j],1e-5)+action(b,Q[j],1e-5))/2;W.append(w)
  if len(Q)<max_basis:
   q=orthogonal(w);r=prior.norm(q)
   if r>1e-12*prior.norm(w):Q.append(q/r)
  print(json.dumps(dict(completed_actions=j+1,basis=len(Q),seconds=time.monotonic()-start)),flush=True)
 Q=Q[:len(W)]
 H=np.array([[prior.dot(q,w) for w in W] for q in Q])
 old=json.loads((ROOT/'output/hyperbolic-core-shell/aurora-staging/interface-span-analysis-v2/result.json').read_text())
 replay=float(np.max(np.abs(H[:4,:4]-np.array(old['rows'][0]['midpoint_tangent_operator']['matrix']))))
 assert replay<1e-7
 def combination(vectors,coeff):return sum(v*z for v,z in zip(vectors,coeff))
 rows=[]
 for k in [4,8,12,20,len(Q)]:
  if k>len(Q):continue
  values,vectors=np.linalg.eig(H[:k,:k]);order=np.argsort(-values.real);ritz=[]
  for j in order[:4]:
   value=values[j];coeff=vectors[:,j];u=combination(Q[:k],coeff);ju=combination(W[:k],coeff)
   r=prior.norm(ju-value*u)
   ritz.append(dict(real=float(value.real),imag=float(value.imag),absolute_residual=r/prior.norm(u),relative_residual=r/prior.norm(ju)))
  rows.append(dict(dimension=k,rightmost_projected=ritz))
 value=values[order[0]];coeff=vectors[:,order[0]];u=combination(Q,coeff)
 independent=[]
 for eps in [1e-5,1e-6]:
  Ju=np.zeros(len(u),dtype=complex)
  for part,phase in [(u.real,1),(u.imag,1j)]:
   if prior.norm(part)>1e-14:Ju+=phase*(action(a,part,eps)+action(b,part,eps))/2
  independent.append(dict(amplitude=eps,real=float(value.real),imag=float(value.imag),absolute_residual=prior.norm(Ju-value*u)/prior.norm(u),relative_residual=prior.norm(Ju-value*u)/prior.norm(Ju)))
 result=dict(scope='Read-only fixed midpoint tangent operator, block-seeded Krylov projection; no implemented interpolation, evolution, or global stability claim',rows=rows,independent_actions=independent,matrix=H.tolist(),old_four_dimensional_matrix_max_difference=replay,basis_dimension=len(Q),orthogonality_error=max(abs(prior.dot(q,r)-(i==j)) for i,q in enumerate(Q) for j,r in enumerate(Q)),library_sha256=prior.sha(lib),source_manifest_sha256=prior.sha(lib.with_name('source.json')),script_sha256=prior.sha(__file__),imports={str(Path(z.__file__).relative_to(ROOT)):prior.sha(z.__file__) for z in [v6,v7,c,prior]},checkpoints=[meta6,meta7],calls=[a.calls,b.calls],seconds=time.monotonic()-start)
 Path(sys.argv[1]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(seconds=result['seconds'],rightmost=rows[-1],independent=independent)))
finally:a.close();b.close()
