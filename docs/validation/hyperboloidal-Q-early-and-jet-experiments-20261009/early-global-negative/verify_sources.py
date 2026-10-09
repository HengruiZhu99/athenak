"""Recheck early gates/builds and unchanged original/late science, without reruns."""
from pathlib import Path
import hashlib,json,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r={};auth=json.loads((w/'gate-authorization.json').read_text())
for key in ['gate','independent']:
 p=Path(auth[key+'_index_path']);assert sha(p)==auth[key+'_index_sha256'];idx=json.loads(p.read_text())
 for n,h in idx['files'].items():assert sha(p.parent/n)==(h if isinstance(h,str) else h['sha256'])
 r[key+'_indexed_files_checked']=len(idx['files'])
receipt=json.loads(Path(auth['gate_receipt_path']).read_text())
for n,h in receipt['source_after'].items():
 p=Path(n);p=p if p.is_absolute() else root/p;assert sha(p)==h
r['core_inputs_rechecked']=len(receipt['source_after']);assert sha(Path(receipt['commands'][-1]['command'][1]))==receipt['postprocessing_source_sha256'];r['separately_pinned_postprocessor_rechecked']=True
count=0
for row in json.loads((w/'build-provenance.json').read_text())['builds'].values():
 for p,h in row['compiler_dependency_hashes'].items():assert sha(p)==h;count+=1
 for p,h in row['link_archive_hashes'].items():assert sha(p)==h
 assert sha(Path(row['command'][row['command'].index('-o')+1]))==row['executable_sha256']
r['two_build_dependency_references_checked']=count
for file in ['field-diagnostic-build.json','reference-build.json']:
 row=json.loads((w/'full22-candidate'/file).read_text())
 for p,h in row['dependency_hashes'].items():assert sha(p)==h
 for p,h in row['link_archive_hashes'].items():assert sha(p)==h
 assert sha(Path(row['command'][row['command'].index('-o')+1]))==row['executable_sha256']
r['callbacks_rechecked']=2
late=w.parent/'full-tensor-conformal-q-null-feedback';snapshot=late/'immutable-Q-null-global-screen-20261009';assert sha(snapshot/'index.json')=='d12e8e4da86f0918cfc7ad741c61a4cff3214494dc808f90df9eb061f0918214';index=json.loads((snapshot/'index.json').read_text())
working_checked=0;working_omissions=[]
for n,h in index['files'].items():
 assert sha(snapshot/n)==h['sha256']
 if n!='freeze.log' and (late/n).exists():assert sha(late/n)==h['sha256'];working_checked+=1
 else:working_omissions.append(n)
r['late_frozen_small_files_checked']=len(index['files']);r['late_working_small_files_checked']=working_checked;r['late_working_omissions_collector_stdout_or_snapshot_generated_only']=working_omissions
for row in json.loads((snapshot/'build-provenance.json').read_text())['builds'].values():
 for p,h in row['compiler_dependency_hashes'].items():assert sha(p)==h
 assert sha(Path(row['command'][row['command'].index('-o')+1]))==row['executable_sha256']
original=w.parent/'full-tensor-global-final';manifest=json.loads((original/'manifest.json').read_text());assert sha(original/'manifest.json')=='4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2'
for n,h in manifest['files'].items():assert sha(original/n)==h['sha256']
r['original_global_small_files_rechecked']=len(manifest['files'])
r['no_production_edits']=subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)=='';assert r['no_production_edits']
r['explicit_parameters_match_native_and_authorization']=auth['explicit_parameters']
meta=json.loads((w/'full22-candidate/spatialnorm-cache0.0001-metadata.json').read_text());assert meta['physical_trace_lapse_input']==False and meta['preferred_source_input']==True and meta['scri_lapse_damping']==2 and meta['null_feedback_weight_mode']==0
r['passed']=True;(w/'source-identity-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
