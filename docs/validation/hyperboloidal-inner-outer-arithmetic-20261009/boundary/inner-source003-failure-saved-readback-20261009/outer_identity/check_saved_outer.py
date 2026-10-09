#!/usr/bin/env python3
"""Saved-row bit identity and W1 classification; no numerical target replay."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    if sys.flags.optimize or not sys.dont_write_bytecode:
        raise RuntimeError('Unoptimized bytecode-off execution required')
    root=Path(__file__).resolve().parent
    recipe=json.loads((root/'recipe.json').read_text())
    pins=json.loads((root/'input-pins.json').read_text())
    def guard():
        for p in pins:
            if Path(p['path']).stat().st_size!=p['bytes'] or digest(p['path'])!=p['sha256']:
                raise RuntimeError('Pin drift: '+p['path'])
    guard()
    attempt=root/'attempt001'
    attempt.mkdir(exist_ok=False)
    started=time.monotonic()
    receipt={'passed':False,'completed':False,'inputs_unchanged':False,'scope':recipe['scope']}
    try:
        associations=json.loads(Path(recipe['associations']).read_text())
        rows={}
        for key,path in recipe['saved_sources'].items():
            with open(path) as f:
                rows[key]={line:json.loads(text) for line,text in enumerate(f,1)}
        counts=Counter();by_family_check=Counter();by_field=Counter();by_radius=Counter()
        unique_queries=set();native_equal=baseline_equal=0
        for failure in associations['source003']:
            assert len(failure['matches'])==1
            m=failure['matches'][0]
            assert m['mode']=='sources' and m['kind']=='source'
            assert m['W']==1
            line=m['line']; component=m['component']; key='parts' if failure['check']=='source-parts' else 'rhs'
            current=rows['source003'][line];old=rows['source002'][line]
            assert current['W']==old['W']==1
            assert current['input']==old['input'] and current['reference']==old['reference']
            assert current['baseline_valid'] and current['valid'] and current['assembled']
            assert current['outer_value_bitwise'] is True
            assert float(current[key][component][0]).hex()==failure['got_hex']
            assert float(current[key][component][0]).hex()==float(old[key][component][0]).hex()
            native_equal+=1
            # Original W1 contract applies to all eight returned split parts.
            assert all(float(a[0]).hex()==float(b[0]).hex()
                       for a,b in zip(current['parts'],current['baseline_parts']))
            baseline_equal+=1
            counts[str(m['W'])]+=1
            by_family_check[(m['family'],failure['check'])]+=1
            by_field[m['field']]+=1
            by_radius[str(m['fixed_radius'])]+=1
            unique_queries.add(line)
        summary={'passed_saved_identity_readback':True,
                 'source003_original_failed':True,'supplement_eligible':False,
                 'all_failures_W_exactly_one':True,'failures_checked':native_equal,
                 'failure_native_bits_equal_source002':native_equal,
                 'failure_query_all_split_parts_equal_frozen_RWM':baseline_equal,
                 'distinct_failed_query_rows':len(unique_queries),
                 'W_counts':dict(counts),'by_radius':dict(by_radius),'by_field':dict(by_field),
                 'by_family_check':[{'family':f,'check':c,'failures':n}
                                    for (f,c),n in sorted(by_family_check.items())],
                 'interpretation':'All measured remaining failures bypass the new inner arithmetic and occur in the unchanged frozen RWM branch. Inner-only modification cannot repair these rows while retaining exact outer bitwise equality.',
                 'scope':'Saved values/metadata/bit identity only; no oracle target recomputation, arithmetic replay or new scientific query.'}
        write(attempt/'summary.json',summary)
        guard()
        receipt.update(passed=True,completed=True,inputs_unchanged=True,returncode=0,
                       elapsed_seconds=time.monotonic()-started,summary_sha256=digest(attempt/'summary.json'))
        write(attempt/'receipt.json',receipt)
        print(json.dumps({'passed':True,'failed_queries':len(unique_queries),'failures':native_equal,
                          'all_W1':True,'summary_sha256':receipt['summary_sha256']}))
    except Exception as e:
        receipt.update(returncode=1,error=repr(e),elapsed_seconds=time.monotonic()-started)
        try:guard();receipt['inputs_unchanged']=True
        except Exception as drift:receipt['pin_error']=repr(drift)
        write(attempt/'receipt.json',receipt)
        raise


if __name__=='__main__':
    main()
