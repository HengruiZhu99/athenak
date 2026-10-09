"""Independent saved J0 degree-control arithmetic; no assembly, queries or generator spectra."""
from pathlib import Path
import hashlib,json,math,warnings
import numpy as np
warnings.simplefilter('error');np.seterr(all='raise')
R=Path(__file__).resolve().parents[2];P=R/'boundary/total-j-finite-rb-degree-control-20261009';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mm(a,b):return np.einsum('ik,kj->ij',a,b,optimize=False)
def err(a,b):return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(a),np.linalg.norm(b)))
def read(p):return json.loads(p.read_text())
inputs=P/'immutable-J0-finite-rb-degree-control-20261009/J0-degree-review-inputs.json'
assert sha(inputs)=='40852becf2d8cf3094d695d0984ef300ea64aa9afce81508d95111cd1d59afe7'
manifest=read(inputs);pins={};results=[]
def pin(path,expected):
 path=Path(path);assert sha(path)==expected;name=str(path.resolve())
 assert name not in pins or pins[name]==expected;pins[name]=expected
for degree in manifest['degree_controls']:
 N=degree['N'];assert N in (12,16)
 for item in degree['inputs']:pin(item['path'],item['sha256']);assert Path(item['path']).stat().st_size==item['bytes']
 primary=P/f'J0-N{N}-rb.98-segmentedQ64-a12x24-primary001';family=P/f'J0-N{N}-forcing-family-replay001'
 report=read(family/'report.json');assert report['passed'] and report['family_count']==33 and len(report['cases'])==33
 for item in report['input_pins']:pin(item['path'],item['sha256'])
 pin(family/'forcing-family.npz',report['array_sha256'])
 with np.load(primary/'operator.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(family/'forcing-family.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
 assert all(np.isfinite(v).all() and v.dtype==np.float64 for v in [*a.values(),*f.values()])
 X=f['family_X'];assert X.shape==(8*N,33)
 assert {(r['channel'],r['polynomial_rho_degree']) for r in report['cases'][:-1]}=={(c,d) for c in range(8) for d in range(4)} and report['cases'][-1]['mixed']
 rhs=mm(a['Kweak']+a['SATload'],X)+f['pointwise_manufactured_load']+f['incoming_boundary_load'];L=np.linalg.cholesky(a['E'])
 solved=np.linalg.solve(L.T,np.linalg.solve(L,rhs));delta=solved-X
 maxc=max(err(solved[:,i],X[:,i]) for i in range(33));maxe=max(float(np.linalg.norm(np.einsum('ij,j->i',L.T,delta[:,i],optimize=False))/max(1.,np.linalg.norm(np.einsum('ij,j->i',L.T,X[:,i],optimize=False)))) for i in range(33))
 assert maxc<2e-9 and maxe<2e-9
 mixed=[err(f['pointwise_manufactured_load'][:,-1],a['manufactured_load']),err(f['incoming_boundary_load'][:,-1],a['manufactured_boundary_load']),err(X[:,-1],a['manufactured_X'])];assert max(mixed)<5e-11
 comparisons={}
 for kind,name in [('radial','quadrature'),('angular','angular')]:
  q=read(P/f'J0-N{N}-{name}-comparison.json');assert q['passed'] and q['threshold']==2e-8
  arrays=[]
  for path,digest in zip(q['input_paths'],q['input_matrix_sha256']):
   file=Path(path)/'operator.npz';pin(file,digest)
   with np.load(file,allow_pickle=False) as z:arrays.append({k:z[k] for k in q['rows']})
  measured={k:err(arrays[0][k],arrays[1][k]) for k in q['rows']};assert max(measured.values())<2e-8
  assert all(abs(measured[k]-q['rows'][k]['scaled'])<1e-12 for k in measured)
  comparisons[kind]=measured
 mass=read(primary/'auxiliary-readback/report.json');exact=read(P/f'polynomial-mass-N{N}/exact.json')
 assert mass['passed']
 results.append({'N':N,'family_fields':33,'independent_Cholesky_max_coefficient_scaled':maxc,'independent_Cholesky_max_energy_scaled':maxe,'mixed_max_scaled':max(mixed),'comparisons':comparisons,'mass_report_sha256':sha(primary/'auxiliary-readback/report.json'),'exact_mass_sha256':sha(P/f'polynomial-mass-N{N}/exact.json'),'matrix_independent_readback_sha256':sha(OUT/f'N{N}-matrix-readback.json'),'matrix_status':read(OUT/f'N{N}-matrix-readback.json')['status']})
for name,digest in pins.items():assert sha(name)==digest
record={'passed':True,'kind':__doc__,'source_sha256':sha(__file__),'input_manifest_sha256':sha(inputs),'checked_input_pins':pins,'sources_unchanged':True,'results':results,'source_scope':'Degree assembler differs from independently reviewed N8 BLAS source only in degree admission, copied basename and hash-bound degree-independent reference cache reuse. Forcing numerical construction unchanged; imported degree source binding and admission pins updated.','source_review':'Direct pointwise forcing and incoming loads are retained; independent saved load solves are not claimed independent source equations. Exact mass proof uses unchanged polynomial coefficient integration generalized to declared N.','generator_spectrum_or_propagation':False,'continuum_or_stability_acceptance':False}
f=OUT/'receipt.json'
with f.open('x') as h:h.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'receipt_sha256':sha(f),'checked_pins':len(pins),'results':results},indent=2))
