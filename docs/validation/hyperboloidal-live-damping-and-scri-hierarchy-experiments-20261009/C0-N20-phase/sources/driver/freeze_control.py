"""Freeze actual C0 N20 control with exact original compiled oracle provenance."""
from pathlib import Path
import hashlib,json,shutil,subprocess,math
w=Path(__file__).resolve().parent;v=w/'full22';n=w/'native20';root=w.parents[2];old=w.parent/'full-tensor-propagator';oldv=old/'full22-v2';out=w/'immutable-C0-N20-phase-control-20261009';assert not out.exists();out.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def save(name,value):p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def cp(p,name):q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for folder,label in [(w,'driver'),(n,'native20'),(v,'full22')]:
 for p in folder.iterdir():
  if p.is_file() and p.suffix in ['.py','.cpp','.hpp','.log','.stderr']:cp(p,Path('sources')/label/p.name)
  elif p.is_file() and p.suffix=='.json':
   if p.name.endswith('metadata.json'):continue
   if p.name=='spatialnorm-cache0.0001-validation.json':
    d=read(p);coords=d['metadata'].pop('xyz_omega_volume_ginv_chi');d['metadata']['coordinates_metadata_only']={'source_path':str(p),'sha256':sha(p),'shape':[len(coords),len(coords[0])]};save(Path('receipts')/label/p.name,d)
   else:cp(p,Path('receipts')/label/p.name)
b=read(oldv/'build-provenance.json')['builds']['spatialnorm'];assert sha(oldv/'server-spatialnorm')==b['executable_sha256'];assert all(sha(Path(p))==h for p,h in b['compiler_reported_dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in b['archive_hashes'].items())
assert sha(old/'server-spatialnorm')==sha(old/'projected-v1/server-spatialnorm')=='495647e847aa77cca2c51615ed1fd0e007d71cdbb8b5ed310b37e332c7812bf0'
for name in ['full22_server.cpp','projected_base.hpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp']:assert sha(v/name)==sha(oldv/name)
for name in ['tangent_server.cpp','old-jv-source.cpp']:assert sha(n/name)==sha(old/name)
f=read(v/'field-diagnostic-build.json');assert all(sha(Path(p))==h for p,h in f['dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in f['link_archive_hashes'].items());assert sha(v/'diagnostic-fields')==f['executable_sha256']
identity={'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','original_compile_launch_HEAD':read(oldv/'build-provenance.json')['launch_HEAD'],'new_control_launch_HEAD':'1959930034066ecfd43483452e723cbb197e0ad6','freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'original_full22_executable_sha256':sha(oldv/'server-spatialnorm'),'original_native20_executable_sha256':sha(old/'server-spatialnorm'),'original_full22_compiler_dependency_count':len(b['compiler_reported_dependency_hashes']),'all_original_compiled_dependencies_and_archives_rehashed_unchanged':True,'new_own_field_callback_compile_only_grid_constructor_changed':True,'actual_operator_reuses_original_pinned_compiled_oracles_readonly':True,'no_native_evolution_or_long_canonical_run':True,'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''};assert identity['no_production_edits'];save('source-identity-verification.json',identity);cp(oldv/'build-provenance.json','original-compile-provenance.json')
gate=read(v/'spatialnorm-pre-pilot-gate.json');assert gate['all_consistency_checks_pass'];checks=[]
for t in [2.]:
 p=v/f'spatialnorm-projected-krylov-m50-80-h0.1-t{t}.json';run=read(p);a=read(v/f'spatialnorm-projected-krylov-t{t}-analysis.json');f=read(v/f'spatialnorm-projected-krylov-t{t}-field-analysis.json');pair=max(s['coarse_fine_max_relative_difference'] for c in run['columns'] for s in c['steps']);defect=max(q['relative_state_l2_per_time'] for c in run['columns'] for s in c['steps'] for q in [s['accepted_curve_defect']]+s['output_curve_defects']);eps=max(q['eps1e-5_vs3e-5_relative_constraints_l2'] for h in a['histories'] for q in h['constraint_amplitude_convergence']);assert pair<=1e-10 and a['diagnostic_server_exit']==f['server_exit']==0 and eps<1e-7;checks.append({'time':t,'run_sha256':sha(p),'seconds':run['seconds'],'Arnoldi_matvecs':run['matvecs'],'direct_residual_matvecs':run['residual_matvecs'],'local_pair_max':pair,'actual_curve_defect_over_state_per_time_max':defect,'constraint_epsilon_check_max':eps,'guard_hit':False})
save('propagation-checks.json',checks)
# Exact constant-similarity canonical short action is independently checked.
canonical=read(v/'short-canonical-validation.json');assert canonical['random_matvec_similarity_relative_l2']<1e-13 and max(q for r in canonical['checks'] for q in r['relative_state_l2'])<1e-10
save('short-canonical-verification.json',canonical)
large={}
for folder in [n,v]:
 for p in folder.iterdir():
  if p.is_file() and (p.suffix in ['.npz','.bin'] or p.name.endswith('metadata.json') or p.name=='diagnostic-fields'):large[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'copied':False}
save('large-artifacts-metadata-only.json',large);cp(w/'summary.json','summary.json');cp(w/'REPORT.md','REPORT.md')
# Original N16 full archive and new t6 archive stay immutable.
for folder,index in [(w.parent/'full-tensor-global-final','manifest.json'),(w.parent/'full-tensor-C0-long-window-20261009/immutable-C0-long-window-20261009','index.json')]:
 for name,row in read(folder/index)['files'].items():assert sha(folder/name)==row['sha256']
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for q in x.values():finite(q)
 elif isinstance(x,list):
  for q in x:finite(q)
for p in out.rglob('*.json'):finite(read(p))
files={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()};save('index.json',{'scope':'original C0 spatialnorm actual N20 tinyspan phase-control global screen; fixed pointwise seed; no continuum/physical stability or order claim','files':files});print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'index_sha256':sha(out/'index.json'),'summary_sha256':sha(out/'summary.json'),'REPORT_sha256':sha(out/'REPORT.md')},indent=2))
