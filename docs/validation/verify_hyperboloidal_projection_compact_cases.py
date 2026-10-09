#!/usr/bin/env python3
"""Catalog-aware compact saved-case verifier; original frozen verifier remains unchanged."""
from pathlib import Path
import argparse,hashlib,json,math
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def scaled(a,b):return math.hypot(*(u-v for u,v in zip(a,b)))/max(1.,math.hypot(*a),math.hypot(*b))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bundle',type=Path)
    parser.add_argument('--metadata-only',action='store_true');parser.add_argument('--catalog',type=Path,required=True);a=parser.parse_args();p=a.bundle.resolve()
    assert sha(p/'verify_frozen.py')=='1c2c14962f8cbd0027cb9b1f3130146b005911a739332c7f38ed6926c6ced79d'
    catalog=read(a.catalog);omitted=catalog['omitted_large_payloads']
    allowed={k[len('frozen/'):]:v for k,v in omitted.items() if k.startswith('frozen/')}
    assert len(allowed)==len(omitted)
    index=read(p/'index.json');missing=[]
    for entry in index['files']:
        path=p/entry['path']
        if not path.exists():
            if a.metadata_only and entry['path'] in allowed:
                item=allowed[entry['path']]
                assert item['sha256']==entry['sha256'] and item['bytes']==entry['bytes']
                assert entry['role']=='large_payload' or Path(entry['path']).suffix in {'.npz','.npy'}
                missing.append(entry['path']);continue
            raise RuntimeError('missing frozen file '+entry['path'])
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],entry['path']
    first=p/'attempt-1791558113840144000';half=p/'refined-h0-half/attempt-1791558482849376000'
    exact=p/'exact-projected/attempt-1791559237496324000'
    failures=[]
    for previous,count in [(first,16),(half,4)]:
        receipt=read(previous/'receipt.json');r=rows(previous/'cases.jsonl')
        assert receipt['sources_unchanged'] and receipt['source_before']==receipt['source_after']
        assert receipt['passed_projection_defect_readback_gate'] is False and len(r)==count
        assert all(x['passed'] for x in r[:-1]) and r[-1]['passed'] is False
        assert r[-1]['checks']['continuum'] is False
        assert all(len(x['sequences']['continuum'])==5 for x in r)
        failures.append({'path':str(previous),'cases':count,'failed_checks':[k for k,v in r[-1]['checks'].items() if not v]})
    receipt=read(exact/'receipt.json');r=rows(exact/'cases.jsonl');assert len(r)==294
    assert receipt['passed_analytic_projected_constraint_point_gate'] is True
    assert receipt['full_fourteen_witness_projection_defect_gate_passed'] is False
    assert receipt['sources_unchanged'] and receipt['source_before']==receipt['source_after']
    gauge=0;linearity=0.
    for case in r:
        assert all(case['checks'].values()) and case['passed_analytic_point_controls']
        assert case['continuum_actual_source_numerically_evaluated'] is False
        q=case['physical8'];assert all(len(v)==8 and all(math.isfinite(x) for x in v) for v in q.values())
        linearity=max(linearity,scaled(q['total'],[u+v for u,v in zip(q['bulk'],q['sat'])]))
        if case['witness']['gauge']:
            gauge+=1;assert q['initial']==[0.]*8
            assert case['gauge_continuum_zero_is_derived_Einstein_sector_tangency'] is True
        else:assert case['nongauge_continuum_comparator_unresolved'] is True
    assert gauge==84 and linearity<=5e-11
    summary=read(exact/'summary.json')
    for witness in summary['by_witness']:
        subset=[case for case in r if case['witness']['name']==witness['name']]
        for label in ('initial','bulk','sat','total'):
            assert max(math.hypot(*case['physical8'][label]) for case in subset)==witness[label]['max_sample_l2']
    root=read(p/'exact-projected/root-saved-data-review.json')
    assert root['passed'] and root['rows']==294 and root['gauge_initial_exact_zero_points']==84
    assert root['full_nongauge_continuum_comparator_unresolved'] and root['both_FD_attempts_remain_failed']
    print(json.dumps({'passed_frozen_hash_and_saved_case_readback':True,'files_indexed':len(index['files']),
      'missing_tagged_large_payloads':missing,'analytic_rows':len(r),'gauge_initial_exact_zero':gauge,
      'saved_case_total_linearity_scaled':linearity,'preserved_failed_FD_attempts':failures,
      'full_nongauge_continuum_comparator_unresolved':True,'kernel_queries':0,'spectrum_or_propagation':False,'original_compact_verifier_failed_and_unchanged':True,
      'scope':'Hash and saved-case arithmetic only; the separately retained root review reconstructs modal maps.'},indent=2))
if __name__=='__main__':main()
