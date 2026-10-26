#!/usr/bin/env python3
"""Audit native-only endpoint/action receipts and preserved rejected research."""
import hashlib,json,math,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];E=root/'docs/validation/goal1-minkowski-20261018';out=E/'native-second-order-v1';work=root/'build-goal1-minkowski-20261018/native-second-order-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
sources={};binaries={}
for name,stem in [('endpoint-build.json','native_endpoint'),('action-build.json','native_action')]:
 b=read(out/name);assert b['exit_code']==0;binary=Path(b['command'][b['command'].index('-o')+1]);assert sha(binary)==b['binary_sha256'];binaries[str(binary)]=sha(binary)
 for path,h in b['source_sha256'].items():assert sha(path)==h;sources[path]=h
 deps=(work/(stem+'.d')).read_text();forbidden=['goal1_velocity_','goal1_native_reduced_source','goal1_native_reduced_audit','goal1_holonomic_momentum','goal1_typed_momentum','goal1_native_source_ad','research_canonical']
 assert not any(s in deps for s in forbidden),(stem,'rejected method imported')
for filename,stdout in [('endpoint-run.json','native-endpoint.jsonl'),('action-run.json','native-action.jsonl')]:r=read(out/filename);assert r['exit_code']==0 and sha(out/stdout)==r['stdout_sha256']
rows=[json.loads(s)for s in(out/'native-endpoint.jsonl').read_text().splitlines()if s.startswith('{')];assert len(rows)==12;report=read(out/'native-endpoint-review.json');assert report['all_native_endpoint_checks_pass']and report['no_wave_momenta_or_evolved_gradients']
for key,value in report['maxima'].items():assert all(math.isfinite(r[key])for r in rows)and max(r[key]for r in rows)==value
assert all(r['valid']and r['invalid_trace_pole_rejected']for r in rows)
targets=work/'native-endpoint-double.txt';run=read(out/'endpoint-run.json');assert sha(targets)==run['fixture_sha256']
for row,line in zip(rows,targets.read_text().splitlines()):
 a=[float(v)for v in line.split()];assert len(a)==251 and int(a[0])==row['case'];target=a[190:212];error=max(abs(u-v)/max(1,abs(u),abs(v))for u,v in zip(row['native_rhs'],target));assert error==row['native22_rhs_scaled']
assert max(r['omitted_metric_gradient_time_pole_absolute']for r in rows)>1e-3
action=read(out/'native-action-review.json');assert action['added_evolved_fields']==0 and not action['forbidden_reduction_header_dependencies'];assert not action['scientific_error_budget_pass_claimed'];assert action['whole_domain']['native_rhs_normalized_rms']>.1
raw=[json.loads(s)for s in(out/'native-action.jsonl').read_text().splitlines()if s.startswith('{')];assert len(raw)==1 and raw[0]['regions'][0]==action['whole_domain']
activation=read(E/'activation-inventory.json')['source_sha256'];assert len(activation)==401
for path,h in activation.items():assert sha(root/path)==h,(path,'activated numerical source changed')
for path,h in read(E/'baseline-preservation.json')['baseline_sha256'].items():assert sha(root/path)==h,(path,'production/reference changed')
oldpins=read(out/'rejected-reduction-evidence-pins.json')
for path,h in oldpins.items():assert sha(root/path)==h,(path,'rejected evidence changed')
archive=root/'build-goal1-minkowski-20261018/rejected-first-order-reduction-v1';registry=read(archive/'rejected-registry.json')
for path,h in registry['source_sha256'].items():assert sha(path)==h and sha(archive/'source'/Path(path).name)==h,(path,'rejected source changed')
previous={}
for phase in ['full-canonical-v3','physical-interface-v1']:
 old=read(E/phase/'full-candidate-source-index.json');base=root/'build-goal1-minkowski-20261018'/phase/'source'
 for path,h in old.items():assert sha(base/path)==h,(phase,path)
 previous[phase]=len(old)
libraries=read(E/'holonomic-momentum-v1/reused-library-preservation.json')['reused_libraries_rehashed_equal']
for path,h in libraries.items():assert sha(path)==h,(path,'library changed')
assert sha(E/'acceptance-protocol-v0.json')=='5f5167fb994af8473a09dec1b5351754c1eb1ff0e443c353e35c5288d5eda6c9'
for p in [Path(__file__),work/'native_endpoint.d',work/'native_action.d']:sources[str(p)]=sha(p)
snapshot=out/'tested-source';snapshot.mkdir(exist_ok=True)
for path,h in sources.items():
 p=Path(path);target=snapshot/p.name
 if target.exists():assert sha(target)==h
 else:shutil.copy2(p,target)
(out/'tested-source-index.json').write_text(json.dumps(sources,indent=2)+'\n')
result={'scope':__doc__,'all_native_endpoint_receipts_recomputed_pass':True,'native_action_large_errors_retained':True,'native_added_evolved_fields':0,'active_binary_reduction_dependencies':[],'activated_numerical_sources_unchanged':len(activation),'production_binary_and_reference_helpers_unchanged':True,'rejected_research_evidence_files_rehashed_equal':len(oldpins),'rejected_research_source_files_rehashed_equal':len(registry['source_sha256']),'previous_native_candidate_source_counts_unchanged':previous,'tested_binary_sha256':binaries,'reused_libraries_rehashed_equal':libraries,'all_started_handles_terminal':True,'new_native_evolution_trajectories':0,'acceptance_protocol_v0_unchanged_draft':True,'acceptance_matrix_started':False,'remote_work_started':False,'goal1_active':True,'goal2_activated':False,'source_sha256':sha(__file__)};(out/'preservation-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if not isinstance(v,dict)},indent=2))
