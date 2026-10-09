"""One-shot compact freeze of both completed inner-trace native preflights."""
import hashlib,importlib.util,json,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'immutable-native-inner-trace-preflights-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists()
spec=importlib.util.spec_from_file_location('trace_auditor',HERE/'audit_trace_native.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.load_helper()
review=HERE/'independent-source-review/immutable-trace-native-source-review-20261009'
assert sha(review/'index.json')=='386e57e45b93f0600df111dd57e60b99c49deeeb286f0a23c8ccac44bbca30d3'
review_index=json.loads((review/'index.json').read_text())
for name,entry in review_index['files'].items():
 value=entry if isinstance(entry,str) else entry['sha256'];assert sha(review/name)==value
sources={HERE/name:name for name in ['build_trace_native.py','audit_trace_native.py','launch_preflight.py','freeze_preflight.py','native.athinput']}
large=set();results={}
for mode in ['trace','combined']:
 verified=audit.verify_build(mode);summary={}
 for stage in ['reference','short']:
  p=HERE/mode/'audit'/(stage+'.json');r=json.loads(p.read_text());assert r['status']=='PASS' and r['build_verification']==verified
  assert r['auditor_sha256']==sha(HERE/'audit_trace_native.py') and len(r['private_snapshots'])==3
  lp=HERE/mode/(stage+'-launch.json');launch=json.loads(lp.read_text())
  assert launch['source_verification']==verified and launch['launch_script_sha256']==sha(HERE/'launch_preflight.py')
  for path,record in r['all_run_files'].items():assert sha(ROOT/path)==record['sha256'] and (ROOT/path).stat().st_size==record['bytes']
  summary[stage]={'audit_sha256':sha(p),'comparison':r['original_HST_diagnostics'],
    'maximum_drift_from_initial':max(x['full_precision_drift_from_initial_max'] for x in r['private_snapshots'])}
  for f in (HERE/mode/stage).rglob('*'):
   if not f.is_file() or '__pycache__' in f.parts:continue
   if f.suffix in {'.rst','.bin'} or f.name=='athena-validation':large.add(f)
   else:sources[f]=str(f.relative_to(HERE))
  sources[p]=str(p.relative_to(HERE));sources[lp]=str(lp.relative_to(HERE))
  for prefix in ['audit','launch']:
   f=HERE/(prefix+'-'+mode+'-'+stage+'.log');sources[f]=str(f.relative_to(HERE))
 assert json.loads((HERE/mode/'short-launch.json').read_text())['reference_audit_sha256']==sha(HERE/mode/'audit/reference.json')
 bp=HERE/(mode+'-build')
 for f in bp.rglob('*'):
  if not f.is_file():continue
  if f.suffix in {'.o','.d'} or f.name=='athena-inner-'+mode:large.add(f)
  else:sources[f]=str(f.relative_to(HERE))
 results[mode]=summary
for p in review.rglob('*'):
 if p.is_file():sources[p]='independent-source-review/'+str(p.relative_to(review))
base=audit.FAMILY/'native-build-receipt.json';sources[base]='dependencies/original-norm-native-build-receipt.json'
OUT.mkdir();files={}
for source,name in sorted(sources.items()):
 target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 assert target.read_bytes()==source.read_bytes()
 files[name]={'source':str(source.relative_to(ROOT)),'sha256':sha(target),'bytes':target.stat().st_size}
rp=OUT/'REPORT.json';rp.write_text(json.dumps({'scope':'Two bounded regular-lapse variants: native reference/short integrity only; no stable pulse/puncture/scri/BH acceptance.',
 'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'local_gate_sha256':audit.TRACE_INDEX,'results':results,'source_integration_independent_review_sha256':sha(review/'index.json'),
 'unchanged':'Complete original C0 geometry, pole diagnostic, physical-P gauge pole, spatial-norm shift, derivatives, ghosts, KO, final-only projection.',
 'combined_scope':'Direct full regular-lapse replacement before assembly; no additive collapsed-lapse correction.',
 'precision':'All25 active binary64 RST fields; BIN matches binary32 casts. Historical physical_metric keys mean Penrose metric; SPD equivalent for Omega>0.'},indent=2,allow_nan=False)+'\n')
files[rp.name]={'sha256':sha(rp),'bytes':rp.stat().st_size}
index={'immutable':True,'scope':'Finite-Omega preflight only','small_files':files,
 'large_files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(large)}}
(OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'large_hashes':len(large),'index_sha256':sha(OUT/'index.json')}))
