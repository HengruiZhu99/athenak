"""Freeze a fresh long live-only window without mutating archived t2 evidence."""
from pathlib import Path
import hashlib,json,shutil,subprocess,math
w=Path(__file__).resolve().parent;root=w.parents[2];live=w.parent/'full-tensor-live-damping';v=live/'full22-candidate';oldfreeze=live/'immutable-live-damping-global-screen-20261009';out=w/'immutable-live-damping-long-window-20261009';assert not out.exists();out.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
summary=read(w/'summary.json');assert summary['shared_t0_through2_max_relative_state_error']<1e-10 and summary['all_saved_vectors_finite'] and not summary['guard_hit']
index=read(oldfreeze/'index.json');assert sha(oldfreeze/'index.json')=='4e350a8c77f1951f05dedf45ea58c3110b10010442b03031862e3f144f9317df'
for name,row in index['files'].items():assert sha(oldfreeze/name)==row['sha256']
prov=read(live/'build-provenance.json');checks={}
for name,row in prov['builds'].items():
 assert all(sha(Path(p))==h for p,h in row['compiler_dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in row['link_archive_hashes'].items());folder=v if name.startswith('full22') else live;assert sha(folder/'server-spatialnorm')==row['executable_sha256'];checks[name]={'compiler_dependencies':len(row['compiler_dependency_hashes']),'all_dependencies_archives_executable_unchanged':True,'executable_sha256':row['executable_sha256']}
for name,exe in [('field-diagnostic-build.json','diagnostic-fields'),('reference-build.json','reference-coefficients')]:
 row=read(v/name);assert all(sha(Path(p))==h for p,h in row['dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in row['link_archive_hashes'].items());assert sha(v/exe)==row['executable_sha256']
identity={'runtime_implementation':prov['runtime_implementation'],'original_live_build_HEAD':prov['launch_HEAD'],'extension_launch_HEAD':read(w/'PREPARATION.json')['launch_HEAD'],'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'live_t2_index_sha256':sha(oldfreeze/'index.json'),'all_archived_live_t2_files_unchanged':True,'compiled_oracle_identity_checks':checks,'no_new_compile_native_or_canonical_evolution':True,'helper_sha256':sha(v/'live_damping_profile.hpp'),'matrix_sha256':sha(v/'spatialnorm-projected-J20.npz'),'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''};assert identity['matrix_sha256']=='0f20ca347d2494b92b1f18b2fb016c231515aa00d7e87c8ef365fc6188a56680';assert identity['helper_sha256']=='69bbbc137486eb3398f94ed50a8b04583c375da049372b82b19330d8432fc153' and identity['no_production_edits']
(w/'source-identity-verification.json').write_text(json.dumps(identity,indent=2)+'\n')
for p in w.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.log','.stderr','.md']:shutil.copy2(p,out/p.name)
for p,name in [(live/'build-provenance.json','original-live-build-provenance.json'),(oldfreeze/'source-identity-verification.json','original-live-source-identity.json')]:shutil.copy2(p,out/name)
large={str(p):{'sha256':sha(p),'bytes':p.stat().st_size,'copied':False} for p in w.glob('*.npz')}
for p in [v/'spatialnorm-projected-J20.npz',v/'spatialnorm-projected-krylov-m50-80-h0.1-t2.0.npz',live/'spatialnorm-validation-vectors.npz']:large[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'copied':False}
(out/'large-output-metadata.json').write_text(json.dumps(large,indent=2)+'\n')
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for y in x.values():finite(y)
 elif isinstance(x,list):
  for y in x:finite(y)
for p in out.glob('*.json'):finite(read(p))
files={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()};(out/'index.json').write_text(json.dumps({'scope':'fresh live-only N16 projected-continuous exploratory t6, archivedt2unchanged, no longcanonical/native/energyacceptance','files':files},indent=2)+'\n');print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'index_sha256':sha(out/'index.json'),'summary_sha256':sha(out/'summary.json'),'REPORT_sha256':sha(out/'REPORT.md')},indent=2))
