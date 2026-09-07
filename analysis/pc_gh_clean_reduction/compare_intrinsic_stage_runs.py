#!/usr/bin/env python3
"""Compare complete stage payloads across ranks/backends and reject mixed groups."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_stage import read_dump,assemble_ranks

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--target',type=Path,required=True)
p.add_argument('--reference-ranks',type=int,default=1)
p.add_argument('--target-ranks',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();records=[];manifest={};negative=[]

def load(root,case,ranks):
    groups={}
    for path in sorted((root/(case+'-dump')).glob('intrinsic-stage-*.dat')):
        value=read_dump(path);h=value['header'];key=(h['stage'],h['operation'])
        groups.setdefault(key,[]).append(value)
        manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(groups)==9
    if ranks>1:
        parts=next(iter(groups.values()))
        bad=[('missing_rank',parts[:-1]),('duplicate_rank',parts[:-1]+parts[:1])]
        mixed=copy.deepcopy(parts);mixed[0]['header']['stage']+=1;bad.append(('mixed_stage',mixed))
        mixed=copy.deepcopy(parts);mixed[0]['header']['blocks'][0]['gid']=mixed[1]['header']['blocks'][0]['gid'];bad.append(('duplicate_block',mixed))
        for name,values in bad:
            try:assemble_ranks(values,ranks)
            except AssertionError:negative.append(dict(case=case,control=name,status='PASS'))
            else:raise AssertionError(name+' was accepted')
    return {k:assemble_ranks(v,ranks) for k,v in groups.items()}

for dim in [2,3]:
    for order in [2,4,6]:
        case=f'fd{order}-{dim}d';ref=load(a.reference,case,a.reference_ranks);target=load(a.target,case,a.target_ranks)
        for key,expected in ref.items():
            actual=target[key]
            assert expected['header']['blocks']==actual['header']['blocks']
            errors={name:float(np.max(abs(actual[name]-v)/(1+abs(v)))) for name,v in expected.items() if name!='header'}
            assert max(errors.values())<=2e-12,(case,key,errors)
            records.append(dict(case=case,stage=key[0],operation=key[1],errors=errors,status='PASS'))
result=dict(status='PASS',tolerance=2e-12,reference=str(a.reference),target=str(a.target),records=records,negative_controls=negative,raw_sha256=manifest)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(records),'payload comparisons;',len(negative),'group rejection controls; max',max(max(r['errors'].values()) for r in records))
