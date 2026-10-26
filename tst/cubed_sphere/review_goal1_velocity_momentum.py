#!/usr/bin/env python3
"""Recompute completed velocity-map/source gates and audit immutable inputs."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];E=root/'docs/validation/goal1-minkowski-20261018';out=E/'velocity-momentum-v1';W=root/'build-goal1-minkowski-20261018'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
sources={};binaries={};payloads={}
for filename in ['build.json','endpoint-build.json','source-build.json']:
 b=read(out/filename);assert b['exit_code']==0;binary=Path(b['command'][b['command'].index('-o')+1]);assert sha(binary)==b['binary_sha256'];binaries[str(binary)]=sha(binary)
 for file,h in b['source_sha256'].items():assert sha(file)==h,(file,'tested source changed');sources[file]=h
def data(prefix=''):
 rows=[]
 for run in read(out/(prefix+'runs.json')):
  assert run['exit_code']==0;f=out/(prefix+run['label']+'.jsonl');assert sha(f)==run['stdout_sha256'];part=[json.loads(l)for l in f.read_text().splitlines()if l.startswith('{')];assert len({r['ordinal']for r in part})==len(part);rows+=part
  if run['input_payloads']:
   for file,key in [(run['command'][1],'checkpoint_sha256'),(run['command'][2],'reference_sha256')]:assert sha(file)==run['input_payloads'][key];payloads[file]=sha(file)
 assert len(rows)==460;return rows
for prefix,file in [('', 'finite-review.json'),('source-','source-review.json')]:
 rows=data(prefix);r=read(out/file)
 for key,value in r['maxima'].items():assert max(p[key]for p in rows)==value,(file,key)
 if prefix:
  selected=[p for p in rows if 'independent_q_gradient_scaled'in p];assert len(selected)==r['independent_points']==138
  for key,value in r['independent_maxima'].items():assert max(p[key]for p in selected)==value
  assert r['all_native_source_checks_pass'];assert max(p['raw_curl_omission_absolute']for p in rows)==r['raw_curl_omission_max']
 else:assert r['all_finite_map_checks_pass']
oracle=read(out/'independent-review.json');assert oracle['all_independent_map_checks_pass']and oracle['points']==8;assert sha(root/'tst/cubed_sphere/check_goal1_typed_ad.py')==oracle['reuse_old_independent_helpers_sha256']
ep=read(out/'endpoint-review.json');assert ep['all_endpoint_map_checks_pass']and ep['points']==6;run=read(out/'endpoint-run.json');assert run['exit_code']==0;assert sha(run['reused_fixture'])==run['fixture_sha256'];assert sha(out/'endpoint.jsonl')==run['stdout_sha256']
rows=[json.loads(l)for l in(out/'endpoint.jsonl').read_text().splitlines()if l.startswith('{')]
for key,value in ep['maxima'].items():assert max(r[key]for r in rows)==value
assert all(r['nonzero_pole_rejected']and r['second_tangential_incompatibility_rejected']for r in rows)
activation=read(E/'activation-inventory.json')['source_sha256'];assert len(activation)==401
for file,h in activation.items():assert sha(root/file)==h,(file,'activated source changed')
for file,h in read(E/'baseline-preservation.json')['baseline_sha256'].items():assert sha(root/file)==h,(file,'production/reference changed')
previous={}
for phase in ['full-canonical-v3','physical-interface-v1']:
 index=read(E/phase/'full-candidate-source-index.json');base=W/phase/'source'
 for file,h in index.items():assert sha(base/file)==h,(phase,file)
 previous[phase]=len(index)
for phase in ['holonomic-chart-v1','holonomic-momentum-v1','native-reduced-source-v1']:
 for file,h in read(E/phase/'tested-source-index.json').items():assert sha(file)==h,(file,'earlier helper changed')
libraries=read(E/'holonomic-momentum-v1/reused-library-preservation.json')['reused_libraries_rehashed_equal']
for file,h in libraries.items():assert sha(file)==h,(file,'library changed')
extra=['check_goal1_velocity_momentum.py','goal1_typed_support.hpp','goal1_typed_momentum.hpp','goal1_native_ad.hpp','goal1_native_source_ad.hpp','goal1_native_reduced_source.hpp','goal1_endpoint_point.hpp','goal1_reference_connection_ad.hpp','goal1_holonomic_chart.hpp',Path(__file__).name]
for name in extra:p=root/'tst/cubed_sphere'/name;sources[str(p)]=sha(p)
for file in [W/'typed-native-ad-v1/ad_synthetic_cases.hpp',W/'holonomic-chart-v1/chart_native_cases.hpp',W/'full-boundary-v4/include/native_v2_loader.hpp']:
 sources[str(file)]=sha(file)
target=out/'tested-source';target.mkdir(exist_ok=True)
for file,h in sources.items():
 p=Path(file);dest=target/p.name
 if dest.exists():assert sha(dest)==h
 else:shutil.copy2(p,dest)
(out/'tested-source-index.json').write_text(json.dumps(sources,indent=2)+'\n')
assert sha(E/'acceptance-protocol-v0.json')=='5f5167fb994af8473a09dec1b5351754c1eb1ff0e443c353e35c5288d5eda6c9'
report={'scope':__doc__,'all_completed_phase_gates_recomputed_pass':True,'activated_numerical_sources_unchanged':len(activation),'production_binary_and_reference_helpers_unchanged':True,'previous_candidate_source_counts_unchanged':previous,'older_tested_holonomic_native_source_helpers_unchanged':True,'reused_checkpoint_reference_payloads_rehashed_equal':payloads,'tested_binaries_rehashed_equal':binaries,'reused_libraries_rehashed_equal':libraries,'tested_source_count':len(sources),'all_started_handles_terminal':True,'new_native_evolution_trajectories':0,'acceptance_protocol_v0_unchanged_unfrozen':True,'acceptance_matrix_started':False,'remote_work_started':False,'goal2_activated':False,'endpoint_dynamics_protocol_predeclared_not_executed':True,'full_endpoint_characteristics_and_numerical_closure_admitted':False,'source_sha256':sha(__file__)};(out/'preservation-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if not isinstance(v,dict)},indent=2))
