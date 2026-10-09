"""Read-only dependency capture for a held derivation; no mathematical execution."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prod='27c19d20696ea6dd4704032c51dfd026218f64f2'
N8=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009/immutable-total-J-finite-rb-N8-matrix-control-20261009'
local=ROOT/'build-layer-research/boundary/total-j-local-angular-20261009/immutable-C0-spatialnorm-total-J-local-angular-20261009'
indices={
 N8/'index.json':'d083ac45cd9f471898837fe7a1a54d965ad40c954db2f8d2c65ea183d72338e0',
 local/'index.json':'b4131aa02f093b7744d3de1b600e829b7213a59779672978bd84c71f6912c513',
 ROOT/'build-layer-research/continuum/total-j-harmonic-basis/immutable-Cartesian-total-J-basis-20261009/index.json':'414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e',
 ROOT/'build-layer-research/boundary/total-j-flat-core-envelope-20261009/immutable-total-J-flat-core-envelope-20261009/index.json':'b0fde1e0eb95d6660a9fa3d190eda69207e6153c88da35366b038369ac3aa3d4'}
for p,h in indices.items():assert sha(p)==h,(p,sha(p))
items=[]
def capture(source,target,production=False):
 target=P/'inputs'/target;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
 if production:
  rel=str(source.relative_to(ROOT));b=subprocess.run(['git','show',prod+':'+rel],cwd=ROOT,capture_output=True,check=True).stdout
  assert b==source.read_bytes(),source
 shutil.copyfile(source,target);assert sha(target)==sha(source)
 items.append({'path':str(target.relative_to(P)),'origin':str(source),'sha256':sha(target),'bytes':target.stat().st_size,
  'verified_byte_identical_production_commit':prod if production else None})
for name in ('athenak_bridge.hpp','conformal_constraints.hpp','conformal_rhs.hpp','layer_reference.hpp','layer_gauge.hpp','spherical_tensor.hpp'):
 capture(ROOT/'src/z4c/hyperboloidal'/name,'production/'+name,True)
for name in ('actual_bridge.cpp','baseline_dual_spatial.hpp','generic_gauge.hpp','configuration_rows.hpp','radial_normalization.hpp'):
 capture(N8/'implementation'/name,'frozen-baseline/'+name)
capture(local/'core_oracle.cpp','frozen-baseline/core_oracle.cpp')
private=ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
for name in ('native_injection.hpp','spatial_norm_control.hpp'):
 capture(private/name,'frozen-baseline/'+name)
for i,p in enumerate(indices):capture(p,f'upstream-index-{i+1}.json')
docs=[]
for name in ('DERIVATION.md','HELD-RECIPE.md','capture_preparation.py'):
 p=P/name;docs.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size})
r={'kind':'Held source-only Einstein-sector pure-coordinate derivation and dependency capture',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'preparation_HEAD':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
 'production_source_commit':prod,'documents':docs,'inputs':items,
 'reference':{'S':1.,'curvature_radius':.5,'geometry_r0':.05,'geometry_r1':.95,'gauge_r0':.45,'gauge_r1':.85},
 'gauge':{'physical_trace_lapse':True,'preferred_source':False,'norm':True,'xi':2.,'eta_norm':6.,'C':2./3.,'kappa':10.,'kappa2':0.},
 'coordinates':'T=tau(rho), X^i=x^i*zeta(rho); independent Tdot=v and Xdot=x*w',
 'new_reference_jets_required':True,'old_missing_reference_jets_used_as_zero':False,
 'scientific_compile_or_kernel_query':False,'symbolic_or_numeric_scientific_gate_executed':False,
 'operator_spectrum_propagation_or_boundary_admitted':False,
 'review_status':'held for root/independent source-math review; no local numerical gate admitted'}
out=P/'source-only-preparation.json';assert not out.exists();out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({'preparation':str(out),'sha256':sha(out),'captured_inputs':len(items),'documents':docs},indent=2))
