#!/usr/bin/env python3
"""Deterministic saved-data readback; does not call the numerical kernel."""
from pathlib import Path
import collections,hashlib,json,math
P=Path(__file__).resolve().parents[2]/'boundary/total-j-finite-rb-control-20261009'
O=Path(__file__).resolve().parent
names={'core':'attempt-1791555596293519000','gauge':'attempt-1791556101260565000','shell':'attempt-1791556133586266000'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
norm=lambda x:math.sqrt(math.fsum(v*v for v in x))
summary={'scope':'stationary-reference selected-field continuum numerical readback; no radial/global/discrete closure or stability theorem',
    'physical_order':['H','Mx','My','Mz','Zx','Zy','Zz','Theta_physical'],
    'parameters':{'S':1,'a':.5,'geometry':[.05,.95],'kappa_input':10,'kappa2':0,'xi':2,
        'gauge':'physical-P plus frozen spatial-norm control','rb':.98,'J':[0,1,2]},
    'point_norm':'Euclidean point comparison/sample RMS; no Penrose-volume/physical energy inference',
    'stages':{},'source_sha256':sha(__file__)}
for stage,name in names.items():
    p=P/'constraint-rate-attempts'/name;receipt=json.loads((p/'receipt.json').read_text())
    assert receipt['passed_requested_continuum_rate_gate'] and receipt['source_unchanged']
    assert receipt['requested_stage']==stage and receipt['rb']==.98 and receipt['requested_J']==[0,1,2]
    assert receipt['source_before']==receipt['source_after']
    assert all(sha(path)==digest for path,digest in receipt['source_before'].items())
    calls=json.loads((p/'calls.json').read_text());assert len(calls)==receipt['calls']
    assert all(c['returncode']==0 and c['stderr_bytes']==0 for c in calls)
    rows=[json.loads(line) for line in (p/'cases.jsonl').open()]
    assert len(rows)==receipt['case_count'] and all(r['passed'] for r in rows)
    item={'passed':True,'cases':len(rows),'api_calls':len(calls),'seconds':receipt['seconds'],
          'receipt_path':str(p/'receipt.json'),'receipt_sha256':sha(p/'receipt.json'),
          'cases_sha256':sha(p/'cases.jsonl'),'cases_bytes':(p/'cases.jsonl').stat().st_size,
          'calls_sha256':sha(p/'calls.json'),'calls_bytes':(p/'calls.json').stat().st_size,
          'all14_recorded_source_inputs_reverified':len(receipt['source_before'])==14,
          'api_query_rows_by_mode':{mode:sum(c['rows'] for c in calls if c['command'][-1]==mode)
              for mode in sorted({c['command'][-1] for c in calls})},
          'component_sample_norms':receipt['groups'][stage]}
    if stage=='core':
        item['max_errors']={key:{measure:max(r['comparisons'][key][measure] for r in rows)
            for measure in ('absolute_l2','scaled_l2','absolute_peak')} for key in rows[0]['comparisons']}
        item['launch_context_addendum_sha256']=sha(p/'launch-context-addendum.json')
    else:
        item['actual_order_status_counts']=dict(collections.Counter(r['actual_sequence']['order_status'] for r in rows))
        item['subsidiary_order_status_counts']=dict(collections.Counter(r['subsidiary_sequence']['order_status'] for r in rows))
        item['max_final_error']={key:max(r['final_error'][key] for r in rows)
            for key in ('absolute_l2','scaled_l2','absolute_peak')}
        item['max_extrapolated_error']={key:max(r['extrapolated_error'][key] for r in rows)
            for key in ('absolute_l2','scaled_l2','absolute_peak')}
        item['max_last_actual_increment_scaled_l2']=max(r['actual_sequence']['last_increment_scaled_l2'] for r in rows)
        item['max_last_subsidiary_increment_scaled_l2']=max(r['subsidiary_sequence']['last_increment_scaled_l2'] for r in rows)
        item['launch_HEAD']=receipt['launch_HEAD']
        if stage=='gauge':
            item['initial_constraints_max_abs']=max(abs(v) for r in rows for v in r['initial_constraints'])
            item['max_final_zero_rate_l2']=max(norm(r['actual_rate_sequence'][-1]) for r in rows)
            item['max_extrapolated_zero_rate_l2']=max(norm(r['actual_sequence']['richardson_last']) for r in rows)
    summary['stages'][stage]=item
proof=json.loads((O/'proof.stdout.json').read_text());assert proof['passed_exact_flat_core_constraint_rate_identity']
debug=json.loads((O/'debug-api-binding-v2/receipt.json').read_text());assert debug['passed_three_mode_release_asan_binding']
summary.update({'passed_all_declared_rb098_rate_gates':True,'total_cases':sum(x['cases'] for x in summary['stages'].values()),
    'total_scientific_api_calls':sum(x['api_calls'] for x in summary['stages'].values()),
    'total_scientific_seconds':sum(x['seconds'] for x in summary['stages'].values()),
    'exact_flat_symbolic_identity':{'dimensions':[8,20],'exact_nonzero_entries':0,
        'proof_stdout_sha256':sha(O/'proof.stdout.json'),'proof_source_sha256':sha(O/'prove_flat_core.py')},
    'release_asan_api_binding':{'three_modes_times_three_points':True,'bitwise_outputs_equal':all(r['absolute_peak_difference']==0 for r in debug['results']),
        'receipt_sha256':sha(O/'debug-api-binding-v2/receipt.json')},
    'mechanical_failed_debug_spawn':{'receipt_sha256':sha(O/'debug-api-binding/failed-attempt-receipt.json'),
        'reason':'retained byte-copy lacked executable permissions; unchanged-byte working Debug succeeded in fresh v2'},
    'rb0995_not_tested':True,'no_tolerance_relaxation':True,'no_eigenvalues_or_propagation':True})
(O/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='stages'},indent=2))
