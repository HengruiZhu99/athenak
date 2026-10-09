"""Freeze the fresh fullC1 norm t6 extension, keeping original evidence intact."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];old=w.parent/'full-tensor-covariant-c1';native=old/'full22-candidate';frozen=old/'immutable-C1-global-screen-20261009';out=w/'immutable-fullC1-long-window-20261009';assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert sha(frozen/'manifest.json')=='5a44232529dd90e71f493895a6742817d73ee016066ab0cdead159dd19cdee4b'
manifest=read(frozen/'manifest.json')
for name,row in manifest['files'].items():assert sha(frozen/name)==row['sha256'],name
prep=read(w/'PREPARATION.json')
for name,row in prep['pins'].items():assert sha(root/name)==row['sha256'],name
for name,h in prep['scripts'].items():assert sha(w/name)==h,name
prov=read(old/'build-provenance.json');checks={}
for name,row in prov['builds'].items():
 for path,h in row['compiler_dependency_hashes'].items():assert sha(Path(path))==h,path
 for path,h in row['link_archive_hashes'].items():assert sha(Path(path))==h,path
 cmd=row['command'];exe=Path(cmd[cmd.index('-o')+1]);assert sha(exe)==row['executable_sha256']
 checks[name]={'dependency_count':len(row['compiler_dependency_hashes']),'archives':len(row['link_archive_hashes']),'executable_sha256':row['executable_sha256'],'unchanged':True}
field=read(frozen/'field-diagnostic-build.json')
for path,h in field['dependency_hashes'].items():assert sha(Path(path))==h,path
assert sha(native/'diagnostic-fields')==field['executable_sha256']
summary=read(w/'summary.json');assert summary['shared_t0_through2_max_relative_state_error']<1e-10 and summary['all_saved_vectors_finite'] and not summary['guard_hit']
assert sha(native/'spatialnorm-projected-J20.npz')=='1093d7c07a71019cd69e578d52ca0c47635dc190b32ccb371683fd162132241e'
assert subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''
identity={'scope':'New fullC1 norm projected-continuous t6 action; no new compilation or native/longcanonical evolution',
 'runtime_implementation':prov['runtime_implementation'],'original_launch_HEAD':prov['launch_HEAD'],'extension_launch_HEAD':prep['launch_HEAD'],
 'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'original_C1_manifest_sha256':sha(frozen/'manifest.json'),'original_C1_archived_files_unchanged':len(manifest['files']),
 'compiled_oracle_checks':checks,'field_oracle_dependencies_unchanged':len(field['dependency_hashes']),
 'field_oracle_executable_sha256':field['executable_sha256'],'C1_math_sha256':prov['C1_math_sha256'],
 'explicit_overlay_hashes':prov['explicit_overlay_hashes'],'no_production_edits':True,'no_long_canonical_native_or_stability_acceptance':True}
(w/'source-identity-verification.json').write_text(json.dumps(identity,indent=2)+'\n')
out.mkdir()
for p in w.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.log','.stderr','.md']:shutil.copy2(p,out/p.name)
for p,name in [(old/'build-provenance.json','original-C1-build-provenance.json'),(frozen/'field-diagnostic-build.json','original-field-diagnostic-build.json'),(frozen/'source-identity-verification.json','original-C1-source-identity.json')]:shutil.copy2(p,out/name)
large={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size,'copied':False} for p in w.glob('*.npz')}
for p in [native/'spatialnorm-projected-J20.npz',native/'spatialnorm-projected-krylov-m50-80-h0.1-t2.0.npz',old/'spatialnorm-validation-vectors.npz']:
 large[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size,'copied':False}
(out/'large-artifacts-metadata-only.json').write_text(json.dumps(large,indent=2)+'\n')
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
for p in out.rglob('*.json'):finite(read(p))
files={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
(out/'index.json').write_text(json.dumps({'scope':summary['scope'],'files':files},indent=2)+'\n')
for name,row in files.items():assert sha(out/name)==row['sha256']
print(json.dumps({'files':len(files),'bytes':sum(r['bytes'] for r in files.values()),'finite_json':len(list(out.rglob('*.json'))),'index_sha256':sha(out/'index.json'),'summary_sha256':sha(out/'summary.json'),'REPORT_sha256':sha(out/'REPORT.md')},indent=2))
