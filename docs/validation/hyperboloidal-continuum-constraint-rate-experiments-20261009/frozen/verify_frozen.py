#!/usr/bin/env python3
"""Saved-data/hash readback only; no kernel, SymPy, eigensolver or evolution."""
from pathlib import Path
import argparse,hashlib,json,math,collections
parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path)
parser.add_argument('--metadata-only',action='store_true',help='Explicit compact archive: allow missing files tagged large_payload')
args=parser.parse_args();p=args.bundle
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
index=json.loads((p/'index.json').read_text());checked=0;omitted=[]
for f in index['files']:
    q=p/f['path']
    if not q.exists() and args.metadata_only and f['role']=='large_payload':omitted.append(f['path']);continue
    assert q.stat().st_size==f['bytes'] and sha(q)==f['sha256'],f['path'];checked+=1
summary=json.loads((p/'summary.json').read_text());norm=lambda a:math.sqrt(math.fsum(x*x for x in a))
diff=lambda a,b:[x-y for x,y in zip(a,b)]
checks={};total=0
for stage,expected_count in [('core',4431),('gauge',714),('shell',2751)]:
    receipt=json.loads((p/'results'/stage/'receipt.json').read_text())
    assert receipt['passed_requested_continuum_rate_gate'] and receipt['source_before']==receipt['source_after']
    assert receipt['case_count']==expected_count and receipt['requested_stage']==stage
    assert len(receipt['source_before'])==14
    payload=p/'payloads'/(stage+'-cases.jsonl')
    if not payload.exists():
        assert args.metadata_only;checks[stage]={'cases_from_receipt':expected_count,'large_case_readback_omitted':True};total+=expected_count;continue
    count=0;max_final=0.;max_extrap=0.;max_zero=0.;max_initial=0.;statuses=collections.Counter()
    for line in payload.open():
        row=json.loads(line);count+=1;assert row['passed']
        if stage=='core':
            for actual,expected in [('actual_source','oracle_source'),('actual_constraints','oracle_constraints'),
                                    ('actual_rate','oracle_rate'),('subsidiary_rate','oracle_rate')]:
                a=row[actual];b=row[expected];error=norm(diff(a,b))/max(1.,norm(a),norm(b));assert error<=5e-11
                max_final=max(max_final,error)
            continue
        assert len(row['h'])==5 and all(row['h'][i]/2==row['h'][i+1] for i in range(4))
        a=row['actual_rate_sequence'];b=row['subsidiary_rate_sequence'];assert len(a)==len(b)==5
        for key,seq in [('actual_sequence',a),('subsidiary_sequence',b)]:
            inc=[norm(diff(y,x)) for x,y in zip(seq,seq[1:])]
            for x,y in zip(inc,row[key]['increments_absolute_l2']):assert abs(x-y)<=1e-15*max(1.,abs(x),abs(y))
            statuses[key+':'+row[key]['order_status']]+=1
            assert row[key]['order_status']!='unresolved'
        ae=[(16*y-x)/15 for x,y in zip(a[-2],a[-1])];be=[(16*y-x)/15 for x,y in zip(b[-2],b[-1])]
        final=norm(diff(a[-1],b[-1]))/max(1.,norm(a[-1]),norm(b[-1]))
        extrap=norm(diff(ae,be))/max(1.,norm(ae),norm(be));max_final=max(max_final,final);max_extrap=max(max_extrap,extrap)
        assert final<=2e-7 and extrap<=2e-7
        if stage=='gauge':
            initial=norm(row['initial_constraints']);max_initial=max(max_initial,initial);max_zero=max(max_zero,norm(a[-1]))
            assert initial<=5e-11 and norm(a[-1])<=2e-7 and norm(ae)<=2e-7
        else:
            for seq in (a,b):assert norm(diff(seq[-1],seq[-2]))/max(1.,norm(seq[-1]),norm(seq[-2]))<=2e-7
    assert count==expected_count;total+=count
    checks[stage]={'cases':count,'max_final_scaled_l2_recomputed':max_final,
        'max_extrapolated_scaled_l2_recomputed':max_extrap,'max_final_zero_rate_l2':max_zero,
        'max_initial_constraint_l2':max_initial,'order_status_counts':dict(statuses)}
    if stage!='core':
        assert abs(max_final-summary['stages'][stage]['max_final_error']['scaled_l2'])<=1e-15
        assert abs(max_extrap-summary['stages'][stage]['max_extrapolated_error']['scaled_l2'])<=1e-15
assert total==7896
debug=json.loads((p/'debug-api-binding-v2/receipt.json').read_text())
assert debug['passed_three_mode_release_asan_binding'] and all(r['absolute_peak_difference']==0 for r in debug['results'])
print(json.dumps({'passed_saved_data_readback':True,'verified_index_files':checked,'explicitly_omitted_large_payloads':omitted,
    'total_cases':total,'stage_checks':checks,'no_scientific_batch_or_spectral_recomputation':True},indent=2,allow_nan=False))
