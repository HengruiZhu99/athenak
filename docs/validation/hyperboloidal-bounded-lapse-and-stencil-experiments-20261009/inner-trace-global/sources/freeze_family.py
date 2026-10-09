"""Freeze two isolated actual trace-gauge negative screens and exact provenance."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];out=w/'immutable-inner-trace-global-screens-v2-20261009';assert not out.exists();out.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def save(n,d):p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def cp(p,n):q=out/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
summary=read(w/'summary.json');identities={};large={};nativeindex=root/'build-layer-research/inner-trace-native/immutable-native-inner-trace-preflights-20261009/index.json';assert sha(nativeindex)=='6ec8049f02091e620bfa2bf95d79dc0e23490ef45169d3ea3dfd8c2d276cc706'
for label,flag,native in [('trace-only','false','trace-build'),('combined','true','combined-build')]:
 here=w/label;v=here/'full22-candidate';auth=read(here/'gate-authorization.json');gate=Path(auth['gate_index_path']);assert sha(gate)==auth['gate_index_sha256']=='801b78e9cdc7755ddda8b0ce6ea375efb02a023f7cfec420ccd8543490c150f8'
 for n,r in read(gate)['files'].items():assert sha(gate.parent/n)==r['sha256']
 receipt=read(Path(auth['gate_receipt_path']));assert receipt['status']=='PASS' and receipt['sources_unchanged'] and len(receipt['source_after'])==378 and len(receipt['commands'])==12 and all(c['returncode']==0 for c in receipt['commands'])
 for n,h in receipt['source_after'].items():p=Path(n);p=p if p.is_absolute() else root/p;assert sha(p)==h
 for folder,sub in [(here,'native20'),(v,'full22')]:
  for p in sorted(folder.iterdir()):
   if p.is_file() and (p.suffix in ['.py','.cpp','.hpp','.diff','.log','.stderr'] or p.name.startswith('dependencies-')):cp(p,Path(label)/'sources'/sub/p.name)
   elif p.is_file() and p.suffix=='.json':
    if p.name.endswith('metadata.json') or p.name=='reference-coefficients.json':continue
    if p.name=='spatialnorm-cache0.0001-validation.json':
     d=read(p);coords=d['metadata'].pop('xyz_omega_volume_ginv_chi');d['metadata']['coordinates_metadata_only']={'source_path':str(p),'source_sha256':sha(p),'shape':[len(coords),len(coords[0])]};save(Path(label)/'receipts'/sub/p.name,d)
    else:cp(p,Path(label)/'receipts'/sub/p.name)
 for p in (here/'overlay').rglob('*'):
  if p.is_file():cp(p,Path(label)/'sources/overlay'/p.relative_to(here/'overlay'))
 prov=read(here/'build-provenance.json');checks={}
 for name,b in prov['builds'].items():
  assert all(sha(Path(p))==h for p,h in b['compiler_dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in b['link_archive_hashes'].items());folder=v if name.startswith('full22') else here;assert sha(folder/'server-spatialnorm')==b['executable_sha256'];assert not b['production_header_mismatches_vs27c'];checks[name]={'compiler_dependency_count':len(b['compiler_dependency_hashes']),'dependencies_archives_executable_unchanged':True,'executable_sha256':b['executable_sha256']}
 for name,exe in [('field-diagnostic-build.json','diagnostic-fields'),('reference-build.json','reference-coefficients')]:
  b=read(v/name);assert all(sha(Path(p))==h for p,h in b['dependency_hashes'].items());assert all(sha(Path(p))==h for p,h in b['link_archive_hashes'].items());assert sha(v/exe)==b['executable_sha256']
 assert sha(v/'inner_conformal_trace.hpp')=='f2a1011eef65d2860e6d74963be98dcdf4184a65a33600475d00086c30d8922e'
 # Independent root header agrees modulo identical include placement.
 p=here/'overlay/z4c/hyperboloidal/cartesian_patch.hpp';q=root/'build-layer-research/inner-trace-native'/native/'include/z4c/hyperboloidal/cartesian_patch.hpp';include='#include "inner_conformal_trace.hpp"\n';a=p.read_text();b=q.read_text();assert a.replace(include,'')==b.replace(include,'');mark=f'ResearchInnerTraceGauge(p,u,lg,InteriorLayerGauge(p,u,lg),{flag})';assert a.count(mark)==1;restored=a.replace(include,'').replace(mark,'InteriorLayerGauge(p,u,lg)');assert restored==(root/'src/z4c/hyperboloidal/cartesian_patch.hpp').read_text()
 r=summary['variants'][label];assert all(x['pass'] for x in r['stage_checks']);assert all(x['passed'] for x in r['actual_matrix_attribution']['matrices'].values());assert max(x for c in r['short_canonical']['checks'] for x in c['relative_state_l2'])<1e-10 and r['local_pair_max']<=1e-10
 a=read(v/'spatialnorm-projected-krylov-t2.0-analysis.json');f=read(v/'spatialnorm-projected-krylov-t2.0-field-analysis.json');assert a['diagnostic_server_exit']==f['server_exit']==0;eps=max(x['eps1e-5_vs3e-5_relative_constraints_l2'] for h in a['histories'] for x in h['constraint_amplitude_convergence']);assert eps<1e-7
 identities[label]={'launch_HEAD':prov['launch_HEAD'],'runtime_implementation':prov['runtime_implementation'],'compiled_dependency_checks':checks,'local_gate_all378inputs12commands_reverified':True,'helper_sha256':sha(v/'inner_conformal_trace.hpp'),'root_native_index_sha256':sha(nativeindex),'root_native_header_sha256':sha(q),'tangent_header_sha256':sha(p),'root_header_difference_only_include_placement':True,'include_and_one_wrapper_removed_restores_public_header_bitwise':True,'combined_API_flag':flag,'native_constraint_epsilon_check_max':eps,'no_t6_longcanonical_native_evolution_by_this_agent':True}
 large[label]={str(p):{'bytes':p.stat().st_size,'sha256':sha(p),'copied':False} for folder in [here,v] for p in folder.iterdir() if p.is_file() and (p.suffix in ['.npz','.bin'] or p.name.startswith('server-') or p.name in ['diagnostic-fields','reference-coefficients','reference-coefficients.json'] or p.name.endswith('metadata.json'))}
old=w.parent/'full-tensor-propagator';originaloracles={g:sha(old/f'server-{g}') for g in ['production','spatialnorm']};assert all(h==sha(old/'projected-v1'/f'server-{g}') for g,h in originaloracles.items())
for folder,index in [(w.parent/'full-tensor-global-final','manifest.json'),(w.parent/'full-tensor-inner-lapse-advection/immutable-inner-lapse-global-screen-20261009','manifest.json')]:
 for name,row in read(folder/index)['files'].items():assert sha(folder/name)==row['sha256']
identity={'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'variants':identities,'original_native_oracles_unchanged':originaloracles,'original_C0_and_rejected_lapse_archives_all_files_unchanged':True,'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''};assert identity['no_production_edits'];save('source-identity-verification.json',identity);save('large-artifacts-metadata-only.json',large);cp(w/'summary.json','summary.json');cp(w/'REPORT.md','REPORT.md')
for p in w.glob('*.py'):cp(p,Path('sources')/p.name)
cp(w/'comparison.log','comparison.log');cp(w/'report-correction.json','report-correction.json')
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for q in x.values():finite(q)
 elif isinstance(x,list):
  for q in x:finite(q)
for p in out.rglob('*.json'):finite(read(p))
files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()};save('index.json',{'scope':'two separate actual22/native-stage/shortcanonical and exploratory t2 trace gauge negative screens; no t6/adoption/continuumstabilityclaim','files':files});print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'index_sha256':sha(out/'index.json'),'summary_sha256':sha(out/'summary.json'),'REPORT_sha256':sha(out/'REPORT.md')},indent=2))
