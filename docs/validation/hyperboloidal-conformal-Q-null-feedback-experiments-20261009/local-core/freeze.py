"""Freeze completed local source gates; never overwrite a frozen directory."""
from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'receipt.json').read_text());c=json.loads((P/'check-report.json').read_text())
assert r['passed_local_pole_source_principal_corner_gates'] and c['passed_local_pole_source_principal_corner_gates']
assert r['sources_unchanged'] and r['source_before']==r['source_after'] and len(r['source_before'])==382
assert len(r['commands'])==14 and all(x['returncode']==0 and (P/x['stderr']).read_text()==''for x in r['commands'])
for name,digest in r['source_before'].items():assert sha(ROOT/name)==digest,name
for name in ['full20','nonlinear']:assert (P/(name+'.json')).read_bytes()==(P/(name+'-debug.json')).read_bytes()
F=P/'immutable-conformal-Q-null-feedback-local-20261009';assert not F.exists()
paths=[x for x in P.rglob('*')if x.is_file() and x.suffix in ['.cpp','.hpp','.py','.json','.stdout','.stderr','.md','.txt'] and not any(part.endswith('.dSYM')for part in x.parts)]
F.mkdir();files={}
for x in sorted(paths):
 name=x.relative_to(P);dest=F/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(x,dest);files[str(name)]=sha(dest)
large={name:{'sha256':sha(P/name),'bytes':(P/name).stat().st_size,'original_repo_path':str((P/name).relative_to(ROOT))}for name in r['binary_sha256']}
index={'scope':'Local actual gauge source/reference/principal/initial-corner and reconstructed leading20-pole gate only. No finite-frequency, global/native, full Taylor hierarchy, energy, scri closure or BH acceptance.','files':files,'large_outputs_outside_snapshot':large,'input_count':382,'tracked_src_root_CMake_count':365,'command_count':14,'sources_rechecked_on_freeze':True,'passed_local_pole_source_principal_corner_gates':True,'native_or_global_accepted':False,'launch_HEAD':r['launch_HEAD'],'production_implementation':r['production_implementation'],'recommended_target':{'a':.5,'S':1,'kappa_input':10,'scri_lapse_damping':2,'sigma':5,'source_r0':.85,'source_r1':.95,'physical_inner':True,'physical_trace_lapse_input':False,'preferred_source_input':True},'helper_sha256':sha(F/'q_null_feedback.hpp'),'factored_base_sha256':sha(F/'factored_base.hpp'),'receipt_sha256':sha(F/'receipt.json'),'file_count':len(files),'total_bytes':sum((F/name).stat().st_size for name in files)}
(F/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for name,digest in files.items():assert sha(F/name)==digest
print(json.dumps({'path':str(F.relative_to(ROOT)),'index_sha256':sha(F/'index.json'),'receipt_sha256':index['receipt_sha256'],'helper_sha256':index['helper_sha256'],'factored_base_sha256':index['factored_base_sha256'],'files':len(files),'bytes':index['total_bytes'],'commands':14,'inputs':382},indent=2))
