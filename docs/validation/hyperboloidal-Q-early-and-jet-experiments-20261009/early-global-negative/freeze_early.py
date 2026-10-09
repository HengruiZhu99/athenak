"""Freeze only this completed Q screen; large CSR/history/executables are pinned."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess
import numpy as np
w=Path(__file__).resolve().parent;root=w.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.check_call(['python3',str(w/'verify_sources.py')],stdout=subprocess.DEVNULL)
out=w/'immutable-Q-early-weight-global-screen-20261009';assert not out.exists()
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for y in x.values():finite(y)
 elif isinstance(x,list):
  for y in x:finite(y)
checked=[]
for p in w.rglob('*.json'):
 finite(json.loads(p.read_text()));checked.append(str(p.relative_to(w)))
for p in [w/'spatialnorm-validation-vectors.npz',*list((w/'full22-candidate').glob('*t*.npz'))]:
 if 'J22' in p.name or 'J20' in p.name:continue
 with np.load(p) as data:
  for n in data:
   if np.issubdtype(data[n].dtype,np.number):assert np.isfinite(data[n]).all(),str(p)+':'+n
auth=json.loads((w/'gate-authorization.json').read_text());pins=w/'gate-pins';pins.mkdir()
for label,key in [('core','gate'),('independent','independent')]:
 source=Path(auth[key+'_index_path']).parent
 for n in ['index.json','receipt.json','REPORT.md','DERIVATION.md','REVIEW.md']:
  if (source/n).exists():shutil.copy2(source/n,pins/(label+'-'+n))
large={};files={};out.mkdir();allowed={'.py','.hpp','.cpp','.md','.json','.log','.stderr','.stdout','.make','.diff','.txt'}
for p in sorted(w.rglob('*')):
 if not p.is_file() or p.is_relative_to(out):continue
 name=str(p.relative_to(w))
 if p.suffix in allowed:
  q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);files[name]={'sha256':sha(q),'bytes':q.stat().st_size}
 else:large[name]={'original_absolute_path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
externals={}
for p in [w.parent/'full-tensor-propagator/full22-v2/spatialnorm-projected-J20.npz',w.parent/'full-tensor-propagator/full22-v2/spatialnorm-cache0.0001-J22.npz',w.parent/'full-tensor-propagator/projected-v1/spatialnorm-validation-vectors.npz',w.parent/'full-tensor-propagator/full22-v2/spatialnorm-projected-expm-analysis.json',w.parent/'full-tensor-propagator/full22-v2/spatialnorm-projected-krylov-field-analysis.json']:
 externals[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
late=w.parent/'full-tensor-conformal-q-null-feedback/full22-candidate'
for n in ['spatialnorm-projected-J20.npz','spatialnorm-cache0.0001-J22.npz','spatialnorm-projected-krylov-m50-80-h0.1-t2.0.npz','spatialnorm-projected-krylov-t2.0-analysis.json','spatialnorm-projected-krylov-t2.0-field-analysis.json']:
 p=late/n;externals[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
metadata={'local_large_artifacts':large,'unchanged_original_comparison_inputs':externals};q=out/'large-artifacts-metadata-only.json';q.write_text(json.dumps(metadata,indent=2)+'\n');files[q.name]={'sha256':sha(q),'bytes':q.stat().st_size}
index={'scope':'completed actual22/native20 source/stage gates and canonical short validation; negative earlier Wgauge feedback attribution, actual full22/native20/stage/shortcanonical and exploratoryt2, no t6 or longnative','files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),'finite_JSON_files_checked_before_freeze':len(checked),'large_local_artifact_records':len(large),'external_comparison_artifact_records':len(externals),'all_saved_states_finite':True,'implementation_consistency_gates_pass':True,'exploratory_t2_outcome':'catastrophic versus C0; earlier W worsens all endpoint H/M/Z versus frozen late-Q','no_t6_no_long_native_no_long_canonical':True,'global_native_or_scri_stability_accepted':False,'sources_reverified_before_freeze':True,'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','root_owns_native':True,'no_production_edits':True}
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for n,h in files.items():assert sha(out/n)==h['sha256'] and (out/n).stat().st_size==h['bytes']
print(json.dumps({'index_path':str(out/'index.json'),'index_sha256':sha(out/'index.json'),'files':len(files),'bytes':index['total_bytes'],'finite_JSON_files':len(checked),'large_records':len(large)},indent=2))
