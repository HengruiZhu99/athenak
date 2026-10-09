"""Freeze new long-window evidence; original C0 histories and operators remain unchanged."""
from pathlib import Path
import hashlib,json,shutil,subprocess,math
here=Path(__file__).resolve().parent;root=here.parents[2];old=here.parent/'full-tensor-propagator/full22-v2';prior=here.parent/'full-tensor-global-final'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=here/'immutable-C0-long-window-20261009';assert not out.exists();out.mkdir()
build=json.loads((old/'build-provenance.json').read_text());diag=json.loads((prior/'source-identity-verification.json').read_text())
verification={'original_compilation_HEAD':build['launch_HEAD'],'runtime_implementation':build['implementation_reference'],'long_launch_HEAD':'1f58f7b1d18dbcf0f313f65cceb7e7e3853f7125','freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'no_new_compile_or_native_evolution_or_canonical_reevolution':True,'builds':{},'diagnostics':{},'original_scratch_sources':{},'original_archive':{}}
for g,row in build['builds'].items():
 changed=[p for p,h in row['compiler_reported_dependency_hashes'].items() if sha(Path(p))!=h]
 archives=[p for p,h in row['archive_hashes'].items() if sha(Path(p))!=h]
 exe=old/f'server-{g}';native=old.parent/f'server-{g}';frozen=old.parent/'projected-v1'/native.name
 r={'compiler_dependency_count':len(row['compiler_reported_dependency_hashes']),'changed_dependencies':changed,'changed_link_archives':archives,'full22_executable_sha256':sha(exe),'full22_executable_matches_build_receipt':sha(exe)==row['executable_sha256'],'native20_executable_sha256':sha(native),'native20_matches_frozen_original':sha(native)==sha(frozen)}
 assert not changed and not archives and r['full22_executable_matches_build_receipt'] and r['native20_matches_frozen_original'];verification['builds'][g]=r
for name,row in diag['diagnostic_builds'].items():
 changed=[p for p,h in row['dependency_hashes'].items() if sha(Path(p))!=h];archives=[p for p,h in row['archive_hashes'].items() if sha(Path(p))!=h];exe=old/name
 r={'compiler_dependency_count':len(row['dependency_hashes']),'changed_dependencies':changed,'changed_link_archives':archives,'executable_sha256':sha(exe),'executable_matches_frozen_build_receipt':sha(exe)==row['executable_sha256']}
 assert not changed and not archives and r['executable_matches_frozen_build_receipt'];verification['diagnostics'][name]=r
for name,h in build['scratch_sources'].items():
 actual=sha(old/name);assert actual==h;verification['original_scratch_sources'][name]=actual
manifest=json.loads((prior/'manifest.json').read_text())
files=manifest['files']
for name,row in files.items():
 h=row if isinstance(row,str) else row['sha256'];assert sha(prior/name)==h
verification['original_archive']={'manifest_sha256':sha(prior/'manifest.json'),'all_files_unchanged':True,'file_count':len(files)}
(root/'build-layer-research/boundary/full-tensor-C0-long-window-20261009/source-identity-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
for p in here.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.log','.stderr','.md']:
  shutil.copy2(p,out/p.name)
for name in ['build-provenance.json','canonical-vs-krylov-all-states.json']:
 shutil.copy2(old/name,out/('original-'+name))
shutil.copy2(prior/'source-identity-verification.json',out/'original-source-identity-verification.json')
large={}
for p in sorted(here.glob('*.npz')):
 large[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'copied':False}
for g in ['production','spatialnorm']:
 for p in [old/f'{g}-projected-J20.npz',old.parent/f'{g}-validation-vectors.npz',old/f'{g}-projected-expm-t2.0.npz']:
  assert p.exists(),p
  large[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'copied':False}
(out/'large-output-metadata.json').write_text(json.dumps(large,indent=2)+'\n')
files={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
index={'scope':'exploratory original C0 N16 projected-continuous t6, no long independent canonical/native acceptance','files':files,'large_outputs_metadata_only':True,'source_identity_verified':True,'no_guard_hit':True}
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for y in x.values():finite(y)
 elif isinstance(x,list):
  for y in x:finite(y)
count=0
for name,row in files.items():
 p=out/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
 if p.suffix=='.json':finite(json.loads(p.read_text()));count+=1
print(json.dumps({'index_sha256':sha(out/'index.json'),'files':len(files),'bytes':sum(r['bytes'] for r in files.values()),'finite_json_count':count,'verification_passed':True},indent=2))
