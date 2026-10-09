import hashlib,json,pathlib,shutil,subprocess
P=pathlib.Path(__file__).resolve().parent;R=P.parents[2];Q=P/'immutable-local-reference-wave-map-20261009';Q.mkdir()
for n in ['reference_wave_map.hpp','test_support.hpp','wide_arithmetic.hpp','dual_helpers.hpp','nonlinear_values.hpp','local_gate.cpp','run_local.py','PLAN.md','release-recipe.json','REPORT.md','verify_capsule.py','freeze_capsule.py']:shutil.copyfile(P/n,Q/n)
shutil.copytree(P/'attempts',Q/'attempts')
D=Q/'reviewed-context';D.mkdir()
for n in ['DERIVATION.md','SCRI-IDENTITY.md']:shutil.copyfile(R/'build-layer-research/reference-wave-map-proposal-20261009'/n,D/n)
shutil.copyfile(R/'build-layer-research/continuum/reference-wave-map-independent-review-20261009/review.json',D/'independent-derivation-review.json')
shutil.copyfile(R/'build-layer-research/continuum/reference-wave-map-independent-review-20261009/additive-scri-and-coordinate-factoring-review.json',D/'additive-scri-review.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
A=Q/'attempts/1791563579962029000';receipt=json.loads((A/'receipt.json').read_text());recipe=json.loads((A/'release-recipe.json').read_text())
assert receipt['source_after']=={n:sha(R/n)for n in receipt['source_after']}
assert json.loads((A/'release.json').read_text())==json.loads((Q/'attempts/1791563435278429000/release.json').read_text())
files=[]
for p in sorted(Q.rglob('*')):
 if not p.is_file():continue
 role='large_payload'if p.name in ['local-release','local-debug']or p.suffix in ['.npz','.npy']else'source_or_receipt'
 files.append({'path':str(p.relative_to(Q)),'sha256':sha(p),'bytes':p.stat().st_size,'role':role})
idx={'status':'local physical reference wave-map algebra/gauge/dual PASS only','launch_HEAD_at_freeze':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'accepted_attempt':'attempts/1791563579962029000','accepted_receipt_sha256':sha(A/'receipt.json'),'helper_sha256':sha(Q/'reference_wave_map.hpp'),'production_implementation':receipt['production_implementation'],'files':files,'file_count':len(files),'bytes':sum(f['bytes']for f in files),'external_source_inputs':recipe['inputs'],'external_compiler_dependencies':{m:receipt[m+'_compiler_dependencies']for m in ['release','debug']},'scope':'No operators/spectra/propagation/evolution. All failed attempts preserved. Binary payload may be omitted by explicit compact policy.'}
(Q/'index.json').write_text(json.dumps(idx,indent=2)+'\n');print(json.dumps({'path':str(Q),'index_sha256':sha(Q/'index.json'),'files':len(files),'bytes':idx['bytes']},indent=2))
