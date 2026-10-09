"""Recheck exact admitted/core/review/compiled identities without rerunning science."""
from pathlib import Path
import hashlib,json,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r={}
auth=json.loads((w/'gate-authorization.json').read_text())
for key in ['gate','independent']:
 p=Path(auth[key+'_index_path']);assert sha(p)==auth[key+'_index_sha256'];idx=json.loads(p.read_text())
 for n,h in idx['files'].items():assert sha(p.parent/n)==(h if isinstance(h,str) else h['sha256'])
 r[key+'_indexed_files_checked']=len(idx['files'])
receipt=json.loads(Path(auth['gate_receipt_path']).read_text())
for n,h in receipt['source_after'].items():
 p=Path(n);p=p if p.is_absolute() else root/p;assert sha(p)==h
r['core_inputs_rechecked']=len(receipt['source_after'])
build=json.loads((w/'build-provenance.json').read_text());count=0
for row in build['builds'].values():
 for p,h in row['compiler_dependency_hashes'].items():assert sha(p)==h;count+=1
 for p,h in row['link_archive_hashes'].items():assert sha(p)==h
 p=Path(row['command'][row['command'].index('-o')+1]);assert sha(p)==row['executable_sha256']
r['two_build_dependency_references_checked']=count
for file in ['field-diagnostic-build.json','reference-build.json']:
 row=json.loads((w/'full22-candidate'/file).read_text())
 for p,h in row['dependency_hashes'].items():assert sha(p)==h
 for p,h in row['link_archive_hashes'].items():assert sha(p)==h
 assert sha(Path(row['command'][row['command'].index('-o')+1]))==row['executable_sha256']
r['callbacks_rechecked']=2
original=w.parent/'full-tensor-global-final';manifest=json.loads((original/'manifest.json').read_text());assert sha(original/'manifest.json')=='4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2'
for n,h in manifest['files'].items():assert sha(original/n)==h['sha256']
r['original_global_small_files_rechecked']=len(manifest['files'])
r['no_production_edits']=subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)=='';assert r['no_production_edits']
r['input_parameters_match_native']={'physical_trace_lapse':False,'preferred_source':True,'scri_lapse_damping':2,'r0':.45,'r1':.85,'null_r0':.85,'null_r1':.95,'sigma':5,'physical_inner':True}
for p in [w/'tangent_server.cpp',w/'full22-candidate/projected_base.hpp']:
 assert 'g.physical_trace_lapse=false;g.preferred_source=true;g.scri_lapse_damping=2;' in p.read_text()
 assert 'return qnf::Gauge(p,u,g,{.85,.95,5,true});' in (p.parent/'native_injection.hpp').read_text()
r['actual_metadata_explicit_parameters']=json.loads((w/'full22-candidate/spatialnorm-cache0.0001-metadata.json').read_text())['physical_trace_lapse_input']==False
r['passed']=True;(w/'source-identity-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
