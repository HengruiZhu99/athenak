"""One-shot compact freeze of completed factored Q/null native preflights."""
import hashlib,importlib.util,json,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'immutable-native-Q-null-preflight-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists()
spec=importlib.util.spec_from_file_location('qnull_auditor',HERE/'audit_native.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.load_helper()
review=ROOT/'build-layer-research/continuum/independent-q-null-native-review/immutable-independent-Q-null-native-review-20261009'
assert sha(review/'index.json')=='7217f64a021420608213b49b83d6799e24c07aed22672569a95fe549cebd6f9d'
for name,entry in json.loads((review/'index.json').read_text())['files'].items():
 assert sha(review/name)==(entry if isinstance(entry,str) else entry['sha256'])
sources={HERE/name:name for name in ['build_native.py','audit_native.py','launch_preflight.py','freeze_preflight.py','native.athinput','RECIPE.md']}
large=set();summary={};verified=audit.verify_build('physical-inner')
for stage in ['reference','short']:
 p=HERE/'physical-inner/audit'/(stage+'.json');r=json.loads(p.read_text());assert r['status']=='PASS' and r['build_verification']==verified
 assert r['auditor_sha256']==sha(HERE/'audit_native.py') and len(r['private_snapshots'])==3
 lp=HERE/'physical-inner'/(stage+'-launch.json');launch=json.loads(lp.read_text())
 assert launch['source_verification']==verified and launch['launch_script_sha256']==sha(HERE/'launch_preflight.py')
 for name,row in r['all_run_files'].items():assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes']
 summary[stage]={'audit_sha256':sha(p),'comparison':r['original_HST_diagnostics'],'maximum_binary64_drift':max(x['full_precision_drift_from_initial_max'] for x in r['private_snapshots'])}
 for f in (HERE/'physical-inner'/stage).rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts:continue
  if f.suffix in {'.rst','.bin'} or f.name=='athena-validation':large.add(f)
  else:sources[f]=str(f.relative_to(HERE))
 sources[p]=str(p.relative_to(HERE));sources[lp]=str(lp.relative_to(HERE))
assert json.loads((HERE/'physical-inner/short-launch.json').read_text())['reference_audit_sha256']==sha(HERE/'physical-inner/audit/reference.json')
for folder in [HERE/'native-build',HERE/'snapshot-audit']:
 for f in folder.rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts:continue
  if f.suffix in {'.o','.d','.raw','.npz','.npy'} or f.name in {'athena-q-null','check_snapshot'}:large.add(f)
  else:sources[f]=str(f.relative_to(HERE))
u=json.loads((HERE/'snapshot-audit/snapshot-results.json').read_text());assert u['status']=='PASS' and len(u['rows'])==6
c=json.loads((HERE/'snapshot-audit/compile-receipt.json').read_text())
for group in ['all_repository_dependencies_sha256','libraries_sha256']:
 for name,digest in c[group].items():assert sha(ROOT/name)==digest
assert all(row['returncode']==0 and row['stderr']=='' for row in u['rows'])
for f in review.rglob('*'):
 if f.is_file():sources[f]='independent-native-review/'+str(f.relative_to(review))
sources[audit.FAMILY/'native-build-receipt.json']='dependencies/original-norm-native-build-receipt.json'
OUT.mkdir();files={}
for source,name in sorted(sources.items()):
 target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 assert target.read_bytes()==source.read_bytes()
 files[name]={'source':str(source.relative_to(ROOT)),'sha256':sha(target),'bytes':target.stat().st_size}
p=OUT/'REPORT.json';p.write_text(json.dumps({'scope':'Completed finiteOmega reference and angular .02 native integrity preflights only; short H/M/Z worse. No stable pulse, closed scri, puncture or BH acceptance.','freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','local_gate_sha256':audit.TRACE_INDEX,'independent_native_review_index_sha256':sha(review/'index.json'),'results':summary,'snapshot_utility_sha256':sha(HERE/'snapshot-audit/snapshot-results.json'),'unchanged':'PhysicalP geometric evolution/storage, C0 damping, Cart loader/stencils/ng3 ghosts/KO/final projection. Existing pole diagnostic excludes new gauge beta pole.','precision':'All25 active binary64RST/BINcast checks pass. Historical physical_metric keys measurePenrose gtilde/chi; SPD equivalent physical forOmega>0.','utility_scope':'Values-only sigma5-minus0 beta pole; reference derivative jets held equal. Not full live derivative sources or future falloff. Original namespace-only utility compile failure preserved.'},indent=2,allow_nan=False)+'\n')
files[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
index={'immutable':True,'scope':'Bounded completed native preflight; no stable formulation acceptance','files':files,'large_outputs_metadata_only':{str(f.relative_to(ROOT)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(large)}}
(OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'large_hashes':len(large),'index_sha256':sha(OUT/'index.json')}))
