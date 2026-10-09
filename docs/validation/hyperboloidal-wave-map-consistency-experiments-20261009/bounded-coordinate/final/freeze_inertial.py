from pathlib import Path
import hashlib,json,shutil,subprocess
Q=Path(__file__).resolve().parent;P=Q.parent;R=P.parents[2]
F=P/'immutable-bounded-inertial-coordinate-local-20261009'
assert not F.exists(),F
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Recheck the declared active source sets before copying or creating destination.
for name in ['inertial-family-held-003','inertial-identity-held-006']:
 folder=P/name;recipe=json.loads((folder/'local-recipe.json').read_text())
 for rec in recipe['sources']:
  src=folder/rec['path'];assert sha(src)==rec['sha256'],src
s=json.loads((Q/'summary.json').read_text())
for rec in s['pins'].values():assert sha(rec['path'])==rec['sha256'],rec['path']
items=[]
for name in ['inertial-family-held-003','inertial-identity-held-004','inertial-identity-held-005','inertial-identity-held-006']:
 folder=P/name
 for f in sorted(folder.rglob('*')):
  if f.is_file()and'__pycache__'not in f.parts:items.append((f,str(f.relative_to(P))))
for f in sorted(Q.rglob('*')):
 if f.is_file()and'__pycache__'not in f.parts:items.append((f,'final/'+str(f.relative_to(Q))))
# Copy additive reviews and context without altering or recapturing old freezes.
extra=[
 (P/'immutable-Einstein-coordinate-local-attempts-20261009/index.json','context/old-broad-failed-index.json'),
 (P/'FACTORING-IDENTITY.md','context/FACTORING-IDENTITY.md'),
 (R/'build-layer-research/continuum/immutable-independent-higher-reference-jets-20261009/index.json','context/independent-higher-reference-index.json'),
 (R/'build-layer-research/continuum/reference-wave-map-independent-review-20261009/additive-scri-and-coordinate-factoring-review.json','context/additive-independent-factoring-review.json')]
root_review=R/'build-layer-research/reference-CPP-composition-root-review-20261009'
for f in sorted(root_review.rglob('*')):
 if f.is_file()and'__pycache__'not in f.parts:extra.append((f,'context/root-CPP-reference-composition/'+str(f.relative_to(root_review))))
items+=extra
# Construct metadata before mkdir; no source mutations occur during capture.
records=[]
for src,rel in items:
 assert src.is_file(),src
 rec={'path':rel,'sha256':sha(src),'bytes':src.stat().st_size,'original_path':str(src)}
 if src.name=='probe'or src.stat().st_size>1048576:
  rec['metadata_only_reason']='as-built executable retained locally'if src.name=='probe'else'large scientific payload retained locally'
 records.append((src,rec))
assert len({x[1]['path']for x in records})==len(records)
F.mkdir();files=[];large=[];finite_json=0
for src,rec in records:
 if 'metadata_only_reason'in rec:large.append(rec);continue
 dst=F/rec['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 assert sha(dst)==rec['sha256']and sha(src)==rec['sha256'],src
 if dst.suffix=='.json':json.dumps(json.loads(dst.read_text()),allow_nan=False);finite_json+=1
 files.append(rec)
for rec in large:assert sha(rec['original_path'])==rec['sha256'],rec['original_path']
for rec in files:assert sha(F/rec['path'])==rec['sha256'],rec['path']
index={'kind':'frozen bounded inertial-coordinate finite-Omega C0 point-action and stable equivalent identity gates','launch_HEAD_at_capture':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runtime_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':files,'large_payloads_metadata_only':large,'small_file_count':len(files),'small_bytes':sum(x['bytes']for x in files),'finite_json_count':finite_json,'all_copied_and_external_retained_sha256_reverified':True,'passed_stable_equivalent_identity_and_local_scientific_gates':True,'initial_direct_identity_and_old_broad_failures_preserved':True,'input_smoothness_does_not_imply_bounded_gauge_source':True,'Cdot_operator_spectrum_evolution_scri_acceptance':False,'report_sha256':sha(F/'final/REPORT.md'),'summary_sha256':sha(F/'final/summary.json'),'readback_receipt_sha256':sha(F/'final/saved-output-readback001/receipt.json')}
(F/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(F),'index_sha256':sha(F/'index.json'),'files':len(files),'bytes':index['small_bytes'],'finite_json':finite_json,'large_records':len(large),'report_sha256':index['report_sha256'],'summary_sha256':index['summary_sha256']},indent=2))
