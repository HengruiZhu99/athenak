"""Verify and freeze the finite-step approximate-mode action, once only."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess

w=Path(__file__).resolve().parent;root=w.parents[2]
old=w.parent/'full-tensor-propagator/full22-v2'
exports=root/'build-layer-research/continuum/discrete-mode-identification'
out=w/'immutable-mode-final-step-20261009';assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
r=read(w/'results.json');assert r['native_server_exit']==0
assert all(x['byte_identical'] for x in r['fresh_cache_byte_identity'].values())
for name,entry in r['inputs'].items():assert sha(root/name)==entry['sha256'],name
assert sha(w/'check_mode.py')==r['source_sha256']
globalfreeze=w.parent/'full-tensor-global-final'
assert sha(globalfreeze/'manifest.json')=='4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2'
manifest=read(globalfreeze/'manifest.json')
for name,entry in manifest['files'].items():assert sha(globalfreeze/name)==entry['sha256'],name
prov=read(old/'build-provenance.json');build=prov['builds']['spatialnorm']
for name,h in build['compiler_reported_dependency_hashes'].items():assert sha(Path(name))==h,name
for name,h in build['archive_hashes'].items():assert sha(Path(name))==h,name
assert sha(old/'server-spatialnorm')==build['executable_sha256']
for name,h in prov['scratch_sources'].items():assert sha(old/name)==h,name
for name,entry in prov['native_stage_lifecycle_sources'].items():assert sha(root/name)==entry['current_sha256'],name
assert subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''
for row in r['candidates']:
 assert abs(row['recomputed_generator_residual_state_per_time']-row['reported_generator_residual'])<1e-18
 assert row['P_J22_L_vs_J20_action_l2']<1e-10
 assert all(s['abs_mu']>1 for s in row['steps'])
 assert row['steps'][0]['residual_state_per_unit_input_l2']>row['steps'][1]['residual_state_per_unit_input_l2']>row['steps'][2]['residual_state_per_unit_input_l2']
native_error=max(x['state_l2_vs_cached_exact_map'] for row in r['actual_native_one_step_candidate0'] for x in row['eps_sweep'] if x['eps_maxfree_component']==1e-4)
assert native_error<2e-11
identity={'scope':'Original finite-Omega N16 C0norm operator; no new C++ compile, long evolution or physics change',
 'runtime_implementation':prov['implementation_reference'],'original_build_HEAD':prov['launch_HEAD'],
 'current_freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'compiler_dependencies_reverified':len(build['compiler_reported_dependency_hashes']),
 'link_archives_reverified':len(build['archive_hashes']),'all_compiled_dependencies_archives_executable_unchanged':True,
 'all_original_global_archive_files_unchanged':True,'original_global_archive_files':len(manifest['files']),
 'original_global_manifest_sha256':sha(globalfreeze/'manifest.json'),
 'candidate_vectors_sha256':sha(exports/'candidate-vectors.npz'),'candidate_metadata_sha256':sha(exports/'candidate-metadata.json'),
 'all_new_cache_exports_byte_identical':True,'native_eps1e4_max_state_l2_error':native_error,
 'no_production_edits':True,'no_certified_eigenvalue_or_continuum_claim':True,
 'reviewed_trace_report_sha256':sha(root/'build-layer-research/inner-trace-native/rejected-controls-report.md')}
(w/'source-identity-verification.json').write_text(json.dumps(identity,indent=2)+'\n')
out.mkdir()
for p in w.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.log','.stderr','.md']:shutil.copy2(p,out/p.name)
for name in ['full22_server.cpp','projected_base.hpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','build-provenance.json','validate22.py','spatialnorm-cache0.0001-validation.json']:
 dest=out/'original-compiled-source-and-receipts'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(old/name,dest)
for name in ['candidate-metadata.json','candidate-vectors.npz']:
 dest=out/'approximate-mode-inputs'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(exports/name,dest)
large={}
for p in [*w.glob('*.bin'),*w.glob('*.npz'),old/'spatialnorm-cache0.0001-J22.npz',old/'spatialnorm-projected-J20.npz',old/'spatialnorm-cache0.0001-lift.bin',old/'spatialnorm-cache0.0001-restrict.bin',old/'server-spatialnorm']:
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
index={'scope':'Approximate N16 C0norm continuous mode tested against exact cached final-only RK3 action and independent actual one-step finite-difference oracle. No eigenvalue/stability certification.', 'files':files}
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for name,entry in files.items():assert sha(out/name)==entry['sha256']
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'finite_json':len(list(out.rglob('*.json'))),'index_sha256':sha(out/'index.json'),'REPORT_sha256':sha(out/'REPORT.md'),'results_sha256':sha(out/'results.json')},indent=2))
