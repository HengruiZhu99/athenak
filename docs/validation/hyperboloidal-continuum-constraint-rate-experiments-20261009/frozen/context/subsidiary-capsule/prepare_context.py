from pathlib import Path
import json,hashlib,re,shutil,subprocess
ROOT=Path('/Users/hz0693/research/hyperboloidal');P=ROOT/'build-layer-research/continuum/finite-rb-subsidiary-context';P.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
F=ROOT/'build-layer-research/continuum/constraint-propagation/immutable-constraint-propagation-20261009';A=ROOT/'build-layer-research/boundary/total-j-local-angular-20261009/immutable-C0-spatialnorm-total-J-local-angular-20261009';B=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-held-20261009';Q=ROOT/'build-layer-research/continuum/discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp'
files={
 'original/subsidiary.hpp':F/'subsidiary.hpp','original/constraint_tangent.cpp':F/'constraint_tangent.cpp','original/manifest.json':F/'manifest.json','original/receipt.json':F/'receipt.json','original/DERIVATION.md':F/'DERIVATION.md',
 'current-context/baseline_dual.hpp':A/'baseline_dual.hpp','current-context/dual_helpers.hpp':Q,'current-context/bridge.cpp':A/'bridge.cpp','current-context/angular-index.json':A/'index.json',
 'held-context/FINAL-ADDENDUM.md':B/'FINAL-ADDENDUM.md','held-context/final-addendum-receipt.json':B/'final-addendum-receipt.json','held-context/source-pins.json':B/'source-pins.json'}
original_manifest=json.loads((F/'manifest.json').read_text());angular=json.loads((A/'index.json').read_text());angular_files={d['path']:d for d in angular['files']}
for n in ['subsidiary.hpp','constraint_tangent.cpp','receipt.json','DERIVATION.md']:assert sha(F/n)==original_manifest['files'][n]['sha256']
for n in ['baseline_dual.hpp','bridge.cpp']:assert sha(A/n)==angular_files[n]['sha256']
assert sha(Q)==angular_files['compiled-sources/external-research/continuum/discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp']['sha256']
receipt=json.loads((B/'final-addendum-receipt.json').read_text());assert sha(F/'manifest.json')==receipt['subsidiary_manifest']['sha256'];assert sha(F/'subsidiary.hpp')==receipt['subsidiary_helper']['sha256']
# Capture only the lexical project header closure; this is not a compiler-dependency run.
headers={};queue=[ROOT/'src/z4c/hyperboloidal/layer_reference.hpp']
while queue:
 p=queue.pop()
 if p in headers:continue
 headers[p]=sha(p)
 for name in re.findall(r'^\s*#\s*include\s+"([^"]+)"',p.read_text(),re.M):
  q=ROOT/'src'/name
  if not q.exists():q=p.parent/name
  assert q.exists(),(p,name)
  queue.append(q)
for p in headers:files['project-headers/'+str(p.relative_to(ROOT/'src'))]=p
for n,p in files.items():q=P/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def function(s,signature):
 i=s.index(signature);k=s.index('{',i)+1;d=1
 while d:d+=(s[k]=='{')-(s[k]=='}');k+=1
 return s[i:k]
a=(F/'constraint_tangent.cpp').read_text();b=Q.read_text();same=[]
for signature in ['Jet Lift(','hyp::OmegaJet<D> Omega(','std::array<double,8> Constraints(']:
 x=function(a,signature);y=function(b,signature);assert x==y
 same.append({'signature':signature,'byte_identical':True,'sha256':hashlib.sha256(x.encode()).hexdigest(),'bytes':len(x.encode())})
context={'status':'source_only_dependency_context_candidate; no scientific batch or caller implementation','launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compiled_production_implementation_context':'27c19d20696ea6dd4704032c51dfd026218f64f2','existing_boundary_context_is_already_transitively_pinned':True,'source_originals':{n:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for n,p in files.items()},'helper_manifest_sha256':sha(F/'manifest.json'),'helper_sha256':sha(F/'subsidiary.hpp'),'binding_functions_equal':same,'lexical_project_header_count':len(headers),'system_header_scope':'Kokkos and standard library headers are not copied; exact original compiler/include context remains in original receipt. No compiler dependency scan was performed.','required_constraint_order':['H_physical','M_cov_x','M_cov_y','M_cov_z','Z_cov_x','Z_cov_y','Z_cov_z','Theta_physical'],'required_call':'Subsidiary(p,10,qjet)','required_reference':{'S':1,'a':.5,'geometry_r0':.05,'geometry_r1':.95,'stationary':True,'Einstein_background_constraints_zero':True,'Theta_ref':0,'kappa_input':10,'kappa2':0},'coefficient_normalization':'kap=alpha*kappa1; actual ConformalRHS uses10/live_alpha, kappa2=0. p is fixed stationary reference, not live perturbed state.','no_hidden_Omega_or_frame_scaling':True,'actual_finite_rb_caller_reviewed':False,'reason_actual_caller_not_reviewed':'Boundary confirms no actual finite-rb kernel caller exists yet; current-context bridge is only the frozen local angular source context.','mismatch_in_declared_sources_or_scaling_found':False,'scientific_compile_or_batch_performed':False,'new_helper_or_boundary_implementation':False}
(P/'context.json').write_text(json.dumps(context,indent=2)+'\n');print(P,len(files),len(headers))
