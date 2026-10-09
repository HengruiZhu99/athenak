"""Freeze isolated live kappa2 actual22/stage/canonical/cheap-global evidence."""
from pathlib import Path
import difflib,hashlib,json,math,shutil,subprocess,zipfile
import numpy as np
w=Path(__file__).resolve().parent;v=w/'full22-candidate';root=w.parents[2];out=w/'immutable-live-damping-global-screen-20261009';assert not out.exists();out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def save(name,value):p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def cp(p,name):q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
auth=read(w/'gate-authorization.json');gate=Path(auth['gate_index_path']);assert sha(gate)==auth['gate_index_sha256'];index=read(gate)
for name,h in index['files'].items():assert sha(gate.parent/name)==h
receipt=read(Path(auth['gate_receipt_path']));assert receipt['passed_finite_Omega_local_numerical_gate'] and receipt['sources_unchanged'] and len(receipt['source_after'])==384 and len(receipt['commands'])==17
assert all(c['returncode']==0 for c in receipt['commands'])
assert all(sha(Path(p))==h for p,h in receipt['source_after'].items())
for folder,label in [(w,'native20'),(v,'full22')]:
 for p in sorted(folder.iterdir()):
  if p.is_file() and p.suffix in ['.cpp','.hpp','.py','.diff','.log','.stderr'] or p.is_file() and (p.name.startswith('build-') or p.name.startswith('dependencies-')):
   cp(p,Path('sources')/label/p.name)
for p in (w/'overlay').rglob('*'):
 if p.is_file():cp(p,Path('sources/overlay')/p.relative_to(w/'overlay'))
for p in w.glob('*.json'):cp(p,Path('receipts/native20')/p.name)
for p in v.glob('*.json'):
 if p.name.endswith('metadata.json') or p.name=='reference-coefficients.json':continue
 if p.name=='spatialnorm-cache0.0001-validation.json':
  d=read(p);coords=d['metadata'].pop('xyz_omega_volume_ginv_chi');d['metadata']['coordinates_metadata_only']={'shape':[len(coords),len(coords[0])],'source_path':str(p),'sha256':sha(p)};save(Path('receipts/full22')/p.name,d)
 else:cp(p,Path('receipts/full22')/p.name)
# Native root header differs only in placement of one identical helper include.
a=w/'overlay/z4c/hyperboloidal/cartesian_patch.hpp';b=root/'build-layer-research/live-damping-native/native-build/include/z4c/hyperboloidal/cartesian_patch.hpp';include='#include "live_damping_profile.hpp"\n';sa=a.read_text();sb=b.read_text();assert sa.replace(include,'')==sb.replace(include,'')
r=sa.replace(include,'');pairs={
 'ConformalRHS(u,omega,damping/u.alpha.value,ResearchLiveKappa2Profile(u,omega,p.radius,damping))':'ConformalRHS(u,omega,damping/u.alpha.value,Real(0))',
 'ConformalRHS(background,omega0,damping,ResearchLiveKappa2Profile(background,omega0,p.radius,damping))':'ConformalRHS(background,omega0,damping,Real(0))',
 'damping/u.alpha.value,ResearchLiveKappa2Profile(u,CartesianOmega(u,p),p.radius,damping));':'damping/u.alpha.value,Real(0));',
 'damping,ResearchLiveKappa2Profile(background,CartesianOmega(background,p),p.radius,damping)).pole;':'damping,Real(0)).pole;'}
for before,after in pairs.items():assert r.count(before)==1;r=r.replace(before,after)
assert r==(root/'src/z4c/hyperboloidal/cartesian_patch.hpp').read_text()
rootindex=root/'build-layer-research/live-damping-native/immutable-native-live-damping-preflight-20261009/index.json';assert sha(rootindex)=='a0db07fe7aad1a52afc9a609d1a0d5666f5e9e358eb51ba9ce96508cc20b252c'
save('independent-native-source-comparison.json',{'tangent_cartesian_sha256':sha(a),'root_native_cartesian_sha256':sha(b),'difference_only_identical_include_placement':True,'include_and_four_args_removed_restores_public_header_bitwise':True,'root_native_index_path':str(rootindex),'root_native_index_sha256':sha(rootindex),'no_shell_pole_diagnostic_omission':True})
prov=read(w/'build-provenance.json');buildchecks={}
for name,b in prov['builds'].items():
 bad=[p for p,h in b['compiler_dependency_hashes'].items() if sha(Path(p))!=h];libs=[p for p,h in b['link_archive_hashes'].items() if sha(Path(p))!=h];folder=v if name.startswith('full22') else w;assert sha(folder/'server-spatialnorm')==b['executable_sha256'];assert not bad and not libs and not b['production_header_mismatches_vs27c'];buildchecks[name]={'dependency_count':len(b['compiler_dependency_hashes']),'changed_dependencies':bad,'changed_archives':libs,'executable_sha256':b['executable_sha256']}
for name,exe in [('field-diagnostic-build.json','diagnostic-fields'),('reference-build.json','reference-coefficients')]:
 b=read(v/name);assert all(sha(Path(p))==h for p,h in b['dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in b['link_archive_hashes'].items());assert sha(v/exe)==b['executable_sha256']
old=w.parent/'full-tensor-propagator';oracles={g:sha(old/f'server-{g}') for g in ['production','spatialnorm']};assert all(h==sha(old/'projected-v1'/f'server-{g}') for g,h in oracles.items())
oldarchive=w.parent/'full-tensor-global-final';oldmanifest=read(oldarchive/'manifest.json')
for name,row in oldmanifest['files'].items():assert sha(oldarchive/name)==row['sha256']
identity={'launch_HEAD':prov['launch_HEAD'],'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'runtime_implementation':prov['runtime_implementation'],'compiler_version':prov['compiler_version'],'build_rehash_checks':buildchecks,'helper_sha256':sha(v/'live_damping_profile.hpp'),'scientific_gate_index_path':str(gate),'scientific_gate_index_sha256':sha(gate),'all384_local_inputs_rehashed':True,'all17_gate_commands_exit0':True,'original_oracles_unchanged':oracles,'original_C0_global_archive_all_files_unchanged':True,'original_C0_manifest_sha256':sha(oldarchive/'manifest.json'),'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''};assert identity['helper_sha256']=='69bbbc137486eb3398f94ed50a8b04583c375da049372b82b19330d8432fc153' and identity['no_production_edits'];save('source-identity-verification.json',identity)
gate=read(v/'spatialnorm-pre-pilot-gate.json');short=read(v/'spatialnorm-krylov-pilot-independent-validation.json');long=read(v/'spatialnorm-projected-krylov-m50-80-h0.1-t2.0.json');analysis=read(v/'spatialnorm-projected-krylov-t2.0-analysis.json');fields=read(v/'spatialnorm-projected-krylov-t2.0-field-analysis.json');attribution=read(v/'actual-matrix-live-attribution.json')
shorterr=max(q for row in short['checks'] for q in row['relative_state_l2']);assert gate['all_consistency_checks_pass'] and shorterr<1e-10 and all(q['passed'] for q in attribution['matrices'].values())
pair=max(s['coarse_fine_max_relative_difference'] for col in long['columns'] for s in col['steps']);defect=max(q['relative_state_l2_per_time'] for col in long['columns'] for s in col['steps'] for q in [s['accepted_curve_defect']]+s['output_curve_defects']);constraint_eps=max(q['eps1e-5_vs3e-5_relative_constraints_l2'] for h in analysis['histories'] for q in h['constraint_amplitude_convergence']);assert pair<=1e-10 and constraint_eps<1e-7 and analysis['diagnostic_server_exit']==fields['server_exit']==0
summary={'scope':'isolated live V(.15,.3) kappa2 C0 spatialnorm actual22/native-stage/short-canonical and exploratory t2; small mixed screen, no adoption','N':16,'span':2.2,'points':1640,'free20':32800,'raw22':36080,'h':.1375,'min_Omega':.0026953124999994555,'nominal_pole_dt':8.085937499998367e-5,'S':1,'a':.5,'geometry':[.05,.95],'gauge':[.45,.85],'kappa':10,'KO':.1,'symmetric_ghost_degree':2,'actual_stage_checks':gate['checks'],'matrix_attribution':attribution,'short_canonical_max_relative_error':shorterr,'short_canonical_seconds':short['total_seconds'],'long_seconds':long['seconds'],'long_Arnoldi_matvecs':long['matvecs'],'long_direct_residual_matvecs':long['residual_matvecs'],'long_local_pair_max':pair,'long_actual_curve_defect_relative_state_per_time_max':defect,'constraint_epsilon_check_max':constraint_eps,'comparison':read(v/'exploratory-t2.0-vs-frozen-C0.json'),'long_canonical_or_native_or_stability_acceptance':False,'guard_hit':False,'minor_diagnostic_failure_preserved':'check_attribution first run indexed numpy.where tuple incorrectly; traceback retained, corrected driver rerun passed before claims; no matrix/oracle overwritten'};save('summary.json',summary)
large={}
for folder in [w,v]:
 for p in sorted(folder.iterdir()):
  if p.is_file() and (p.suffix in ['.npz','.bin'] or p.name.startswith('server-') or p.name in ['diagnostic-fields','reference-coefficients','reference-coefficients.json'] or p.name.endswith('metadata.json')):large[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'copied':False}
save('large-artifacts-metadata-only.json',large)
cp(w/'REPORT.md','REPORT.md')
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for q in x.values():finite(q)
 elif isinstance(x,list):
  for q in x:finite(q)
for p in out.rglob('*.json'):finite(read(p))
catalog={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()};save('index.json',{'scope':summary['scope'],'files':catalog})
print(json.dumps({'files':len(catalog),'bytes':sum(x['bytes'] for x in catalog.values()),'index_sha256':sha(out/'index.json'),'summary_sha256':sha(out/'summary.json'),'report_sha256':sha(out/'REPORT.md')},indent=2))
