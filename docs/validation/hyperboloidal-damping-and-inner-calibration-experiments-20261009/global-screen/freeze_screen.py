"""Freeze prescribed C0 damping-profile native-stage/short-canonical/long-exploratory screen."""
from pathlib import Path
import hashlib,json,math,shlex,shutil,subprocess,zipfile
import numpy as np
here=Path(__file__).resolve().parent;work=here.parent;v2=work/'full22-candidate';root=work.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(n,d):
 p=here/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def cp(p,n):
 q=here/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def artifact(p):
 d={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'copied':False}
 if p.suffix=='.npz':
  a={}
  with zipfile.ZipFile(p) as z:
   for n in z.namelist():
    with z.open(n) as f:
     version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,version)
     a[n]={'shape':list(shape),'dtype':str(dtype),'fortran_order':order}
  d['arrays']=a
 return d
for folder,label in [(work,'native20'),(v2,'full22')]:
 for p in sorted(folder.iterdir()):
  if p.is_file() and (p.suffix in ['.cpp','.hpp','.py','.diff'] or p.name.startswith('build-') or p.name.startswith('dependencies-')):
   cp(p,Path('sources')/label/p.name)
for p in (work/'overlay').rglob('*'):
 if p.is_file():cp(p,Path('sources/overlay')/p.relative_to(work/'overlay'))
for n in ['build-provenance.json','gate-authorization.json','SOURCE_PREPARATION_HOLD.json','SOURCE_BINDING_HOLD.json','run-commands.json']:
 cp(work/n,Path('receipts')/n)
for n in ['pilot-vs-frozen-C0.json','exploratory-t2.0-vs-frozen-C0.json','actual-matrix-profile-attribution.json']:
 cp(v2/n,Path('receipts')/n)
summary={'scope':'C0 prescribed kappa2(analytic Omega); same actual native Cartesian stencils/ghosts; short verified and long exploratory screen, no adoption','parameters':{'N':16,'span':2.2,'h':.1375,'points':1640,'free20':32800,'raw22':36080,'S':1,'a':.5,'r0':.05,'r1':.95,'kappa':10,'sym_ghost_degree':2,'KO':.1,'min_Omega':.0026953124999994555,'nominal_pole_dt':8.085937499998367e-5,'outward_crossing_time':.7457643839234269},'native_or_global_stability_accepted':False,'long_canonical_comparison_completed':False,'gauges':{}}
large={}
for g in ['production','spatialnorm']:
 p=v2/f'{g}-cache0.0001-validation.json';d=read(p);coords=d['metadata'].pop('xyz_omega_volume_ginv_chi');d['metadata']['coordinates_metadata_only']={'shape':[len(coords),len(coords[0])],'original_receipt_path':str(p),'original_receipt_sha256':sha(p)};save(Path('receipts')/p.name,d)
 for n in [f'{g}-pre-pilot-gate.json',f'{g}-projected-krylov-m50-80-h0.1-t0.05.json',f'{g}-projected-krylov-m50-80-h0.1-t2.0.json',f'{g}-krylov-pilot-independent-validation.json']:
  cp(v2/n,Path('receipts')/n)
 for t in ['0.05','2.0']:
  for kind in ['analysis','field-analysis']:cp(v2/f'{g}-projected-krylov-t{t}-{kind}.json',Path('receipts')/f'{g}-projected-krylov-t{t}-{kind}.json')
 for suffix in ['validation.log','cache0.0001-validation.stderr','krylov-pilot.log','pilot-independent.log','pilot-history-analysis.log','pilot-fields-analysis.log','krylov-t2.log','t2-history-analysis.log','t2-fields-analysis.log']:
  cp(v2/f'{g}-{suffix}',Path('logs')/f'{g}-{suffix}')
 gate=read(v2/f'{g}-pre-pilot-gate.json');short=read(v2/f'{g}-krylov-pilot-independent-validation.json');long=read(v2/f'{g}-projected-krylov-m50-80-h0.1-t2.0.json');analysis=read(v2/f'{g}-projected-krylov-t2.0-analysis.json');fields=read(v2/f'{g}-projected-krylov-t2.0-field-analysis.json')
 assert gate['all_consistency_checks_pass']
 shorterr=max(v for row in short['checks'] for v in row['relative_state_l2']);assert shorterr<1e-10
 pair=max(s['coarse_fine_max_relative_difference'] for col in long['columns'] for s in col['steps']);assert pair<=1e-10
 constraint_eps=max(x['eps1e-5_vs3e-5_relative_constraints_l2'] for h in analysis['histories'] for x in h['constraint_amplitude_convergence']);assert constraint_eps<1e-7
 assert analysis['diagnostic_server_exit']==fields['server_exit']==0
 summary['gauges'][g]={'stage_consistency_checks_pass':True,'short_canonical_max_relative_state_error':shorterr,'short_canonical_total_seconds':short['total_seconds'],'exploratory_long_seconds':long['seconds'],'exploratory_long_matvecs':long['matvecs'],'max_long_coarse_fine_relative_difference':pair,'long_error_scope':'empirical local Arnoldi truncation difference; no independent long canonical forward-error validation','max_native_constraint_amplitude_check_relative_error':constraint_eps,'long_vs_frozen_C0':read(v2/'exploratory-t2.0-vs-frozen-C0.json')['gauges'][g]['rows']}
 paths=[work/f'server-{g}',v2/f'server-{g}',work/f'{g}-validation-vectors.npz',v2/f'{g}-projected-J20.npz',v2/f'{g}-cache0.0001-metadata.json']
 paths += [v2/f'{g}-cache0.0001-{s}' for s in ['J22.npz','indptr.bin','indices.bin','data.bin','lift.bin','restrict.bin']]
 paths += [v2/f'{g}-projected-krylov-m50-80-h0.1-t{t}.npz' for t in ['0.05','2.0']]
 paths += [v2/f'{g}-projected-krylov-t{t}-native-diagnostics.npz' for t in ['0.05','2.0']]
 paths += [v2/f'{g}-krylov-pilot-canonical-vectors.npz']
 large[g]=[artifact(p) for p in paths]
large['other']=[artifact(v2/'diagnostic-fields')]
prov=read(work/'build-provenance.json');checks={}
for name,b in prov['builds'].items():
 bad=[p for p,s in b['compiler_dependency_hashes'].items() if sha(Path(p))!=s];archives=[p for p,s in b['link_archive_hashes'].items() if sha(Path(p))!=s]
 checks[name]={'dependencies':len(b['compiler_dependency_hashes']),'changed_dependencies':bad,'changed_archives':archives,'production_header_mismatches_vs27c':b['production_header_mismatches_vs27c']};assert not bad and not archives and not b['production_header_mismatches_vs27c']
fieldbuild=read(v2/'field-diagnostic-build.json')
assert all(sha(Path(p))==h for p,h in fieldbuild['dependency_hashes'].items())
assert all(sha(Path(p))==h for p,h in fieldbuild['link_archive_hashes'].items())
assert sha(v2/'diagnostic-fields')==fieldbuild['executable_sha256']
cp(v2/'field-diagnostic-build.json','field-diagnostic-build.json')
old=work.parent/'full-tensor-propagator';oracles={g:{'working_sha256':sha(old/f'server-{g}'),'preserved_sha256':sha(old/'projected-v1'/f'server-{g}')} for g in ['production','spatialnorm']};assert all(x['working_sha256']==x['preserved_sha256'] for x in oracles.values())
gateindex=root/'build-layer-research/continuum/damping-profile-control/immutable-C0-profile-local-v2-20261009/index.json'
identity={'launch_HEAD':prov['launch_HEAD'],'runtime_implementation':prov['runtime_implementation'],'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'dependency_identity_checks':checks,'explicit_overlay_hashes':prov['explicit_overlay_hashes'],'profile_helper_sha256':sha(v2/'damping_profile.hpp'),'frozen_local_gate_index':{'path':str(gateindex),'sha256':sha(gateindex)},'unchanged_original_oracle_hashes':oracles,'unchanged_prior_global_manifest_sha256':sha(work.parent/'full-tensor-global-final/manifest.json'),'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''}
assert sha(gateindex)=='c9180b5bbedb2a0069a54853768a96f8b0f69736c30a8bc9c277b77f11fc39ea'
assert identity['no_production_edits'];assert identity['profile_helper_sha256']=='64ba382f509e81fe347b96e915d3933ee7188c97fd6a054524b1aa187a891532';save('source-identity-verification.json',identity);save('summary.json',summary);save('large-artifacts-metadata-only.json',large)
assert (here/'REPORT.md').is_file(), 'write final scoped report before freezing'
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(y) for y in x.values())
 if isinstance(x,list):return all(finite(y) for y in x)
 return True
assert all(finite(read(p)) for p in here.rglob('*.json'))
catalog={str(p.relative_to(here)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(here.rglob('*')) if p.is_file() and p.name!='manifest.json'}
save('manifest.json',{'scope':'actual full22/native short gate and exploratory t2 C0 prescribed damping-profile screen; no long canonical/no adoption','files':catalog})
print(json.dumps({'files':len(catalog),'bytes':sum(x['bytes'] for x in catalog.values()),'REPORT_sha256':sha(here/'REPORT.md'),'summary_sha256':sha(here/'summary.json'),'manifest_sha256':sha(here/'manifest.json')},indent=2))
