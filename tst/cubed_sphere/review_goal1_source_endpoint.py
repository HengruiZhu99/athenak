#!/usr/bin/env python3
"""Read completed point evidence, freeze tested sources and audit preservation."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];E=root/'docs/validation/goal1-minkowski-20261018';W=root/'build-goal1-minkowski-20261018'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
phases=['typed-native-ad-v1','reference-connection-gradient-v1','native-q-source-gradient-v1','native-reduced-source-v1','scri-momentum-chart-v1']
checks={};expected_sources={};binary_hashes={};library_hashes={}
for phase in phases:
 directory=E/phase;builds=[directory/'build.json']
 if phase=='native-reduced-source-v1':builds.append(directory/'unforced-build.json')
 if phase=='scri-momentum-chart-v1':builds.append(directory/'endpoint-build.json')
 for path in builds:
  b=read(path);assert b['exit_code']==0,path
  cmd=b['command'];binary=Path(cmd[cmd.index('-o')+1]);assert sha(binary)==b['binary_sha256'],binary;binary_hashes[str(binary)]=sha(binary)
  sources=b.get('source_sha256',{})
  if isinstance(sources,dict):
   for name,hash0 in sources.items():
    p=Path(name);p=p if p.is_absolute()else root/p;assert sha(p)==hash0,(p,'tested source changed');expected_sources[str(p)]=hash0
  elif isinstance(sources,str):
   p=next(Path(c)for c in cmd if c.endswith('.cpp'));assert sha(p)==sources,p;expected_sources[str(p)]=sources
  for c in cmd:
   if c.endswith('.a')and Path(c).exists():library_hashes[c]=sha(c)
 for name,hash0 in read(E/'holonomic-momentum-v1/reused-library-preservation.json')['reused_libraries_rehashed_equal'].items():assert library_hashes.get(name,sha(name))==hash0,(name,'library changed')
 r=read(directory/'review.json')
 if phase=='typed-native-ad-v1':assert r['map_values_and_AD_invariants_pass']and r['independent_map_derivatives_pass']and r['native_core_AD_agreement_pass']
 elif phase=='reference-connection-gradient-v1':assert r['all_new_reference_gradient_checks_pass']
 elif phase=='native-q-source-gradient-v1':assert r['native_source_value_and_first_gradient_checks_pass']
 elif phase=='native-reduced-source-v1':
  assert r['compatible_and_curl_source_pass'];u=read(directory/'unforced-review.json');assert u['points']==138 and u['all_full_native_source_checks_pass']
  # Recompute all reductions from raw data, including the retained curl failure.
  data=[json.loads(l)for name in ['synthetic','native-stationary4','native-pulse8']for l in(directory/(name+'.jsonl')).read_text().splitlines()if l.startswith('{')];assert len(data)==460
  for key,value in r['maxima'].items():assert max(d[key]for d in data)==value,(key,'summary mismatch')
  assert max(d['raw_incompatible_curl_defect_absolute']for d in data)==r['raw_curl_negative_max']
  data=[json.loads(l)for name in ['synthetic','native-stationary4','native-pulse8']for l in(directory/('unforced-'+name+'.jsonl')).read_text().splitlines()if l.startswith('{')];assert len(data)==138
  for key,value in u['maxima'].items():assert max(d[key]for d in data)==value,(key,'unforced summary mismatch')
 else:
  assert r['physical_oracle_pass']and r['native_stationary_roundoff_conditioned_pass'];u=read(directory/'endpoint-review.json');assert u['all_endpoint_point_checks_pass']and u['canonical_bounded_chart_rejected_for_general_scri'];assert read(directory/'velocity-chart-value-review.json')['agreement_pass_1e-10']
 checks[phase]={'review_sha256':sha(directory/'review.json'),'all_declared_point_checks_pass':True,'new_trajectories':0}
activation=read(E/'activation-inventory.json')['source_sha256'];assert len(activation)==401
for name,h in activation.items():assert sha(root/name)==h,(name,'activation change')
baseline=read(E/'baseline-preservation.json')['baseline_sha256']
for name,h in baseline.items():assert sha(root/name)==h,(name,'baseline change')
previous={}
for phase in ['full-canonical-v3','physical-interface-v1']:
 index=read(E/phase/'full-candidate-source-index.json');base=W/phase/'source'
 for name,h in index.items():assert sha(base/name)==h,(phase,name,'old candidate source changed')
 previous[phase]=len(index)
for phase in ['holonomic-chart-v1','holonomic-momentum-v1']:
 for name,h in read(E/phase/'tested-source-index.json').items():assert sha(name)==h,(name,'old map helper changed')
payloads={}
for run in read(E/'holonomic-momentum-v1/runs.json'):
 if run['input_payloads']:
  for name,key in [(run['command'][1],'checkpoint_sha256'),(run['command'][2],'reference_sha256')]:assert sha(name)==run['input_payloads'][key];payloads[name]=sha(name)
for file in [W/'holonomic-chart-v1/chart_native_cases.hpp',W/'typed-native-ad-v1/ad_synthetic_cases.hpp',W/'full-boundary-v4/include/native_v2_loader.hpp']:
 expected_sources[str(file)]=sha(file)
for file in (root/'tst/cubed_sphere').iterdir():
 if file.name in ['run_goal1_typed_ad.py','check_goal1_typed_ad.py','check_goal1_reference_connection.py','run_goal1_native_source_ad.py','check_goal1_scri_momentum_chart.py','check_goal1_endpoint_point.py','check_goal1_velocity_chart_value.py',Path(__file__).name]:expected_sources[str(file)]=sha(file)
snapshot=E/'native-reduced-source-v1/tested-source';snapshot.mkdir(exist_ok=True)
for name,h in expected_sources.items():
 source=Path(name);target=snapshot/source.name
 if target.exists():assert sha(target)==h,(target,'snapshot collision')
 else:shutil.copy2(source,target)
(E/'native-reduced-source-v1/tested-source-index.json').write_text(json.dumps(expected_sources,indent=2)+'\n')
assert sha(E/'acceptance-protocol-v0.json')=='5f5167fb994af8473a09dec1b5351754c1eb1ff0e443c353e35c5288d5eda6c9'
report={'scope':__doc__,'phase_checks':checks,'activated_source_count':len(activation),'all_activated_numerical_sources_unchanged':True,'production_binary_and_reference_helpers_unchanged':True,'older_candidate_source_counts_unchanged':previous,'previous_holonomic_helpers_unchanged':True,'tested_sources_frozen':len(expected_sources),'tested_binary_sha256':binary_hashes,'reused_libraries_rehashed_equal':library_hashes,'reused_checkpoint_reference_payloads_rehashed_equal':payloads,'all_started_handles_terminal':True,'new_evolution_trajectories':0,'acceptance_protocol_v0_unchanged_draft':True,'acceptance_matrix_started':False,'native_reduced_evolution_implemented':False,'full_exact_endpoint_characteristics_time_and_numerical_closure_admitted':False,'remote_work_started':False,'goal1_active':True,'goal2_activated':False,'source_sha256':sha(__file__)}
(E/'native-reduced-source-v1/preservation-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['tested_binary_sha256','reused_libraries_rehashed_equal','reused_checkpoint_reference_payloads_rehashed_equal']},indent=2))
