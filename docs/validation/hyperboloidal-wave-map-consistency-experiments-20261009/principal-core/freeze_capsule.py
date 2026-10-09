import pathlib,hashlib,json,shutil,subprocess
P=pathlib.Path(__file__).resolve().parent;R=P.parents[2];Q=P/'immutable-principal-core-wave-map-20261009';Q.mkdir()
for n in ['reference_wave_map.hpp','dual_helpers.hpp','principal.cpp','core.cpp','check_principal.py','run_gate.py','PRECOMPILE.md','release-recipe.json','REPORT.md','tensor-parts-arithmetic-repair.patch','verify_capsule.py','freeze_capsule.py']:shutil.copyfile(P/n,Q/n)
shutil.copytree(P/'attempts',Q/'attempts');D=Q/'held-context';D.mkdir();H=R/'build-layer-research/continuum/reference-wave-map-principal-coordinate-held-20261009'
for n in ['PLAN.md','held-recipe.json','root-A-B-release-C-held.json','C-BOUNDED-INERTIAL-REVISION.md','C-bounded-inertial-held.json']:shutil.copyfile(H/n,D/n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
A=Q/'attempts/1791564714189871000';r=json.loads((A/'receipt.json').read_text());assert r['source_after']=={p:sha(R/p)for p in r['source_after']}
files=[]
for p in sorted(Q.rglob('*')):
 if not p.is_file():continue
 large=p.name in ['principal-release','principal-debug','core-release','core-debug']or p.suffix in ['.npz','.npy']or p.name in ['principal-release.json','principal-debug.json','run-principal-release.stdout','run-principal-debug.stdout']
 files.append({'path':str(p.relative_to(Q)),'sha256':sha(p),'bytes':p.stat().st_size,'role':'large_payload'if large else'source_or_receipt'})
idx={'status':'A actual principal20 and B exact core PASS only; C HELD','launch_HEAD_at_freeze':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'production_implementation':r['production_implementation'],'accepted_attempt':'attempts/1791564714189871000','accepted_receipt_sha256':sha(A/'receipt.json'),'helper_sha256':sha(Q/'reference_wave_map.hpp'),'files':files,'file_count':len(files),'bytes':sum(f['bytes']for f in files),'external_source_inputs':r['source_before'],'external_compiler_dependencies':{k:v for k,v in r.items()if k.endswith('_dependencies')},'scope':'No finite-k, eigenvalues, C queries, operator, propagation or evolution. Failed assembled-subtraction attempt preserved.'}
(Q/'index.json').write_text(json.dumps(idx,indent=2)+'\n');print(json.dumps({'path':str(Q),'index_sha256':sha(Q/'index.json'),'file_count':len(files),'bytes':idx['bytes']},indent=2))
