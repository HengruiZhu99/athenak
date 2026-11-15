#!/usr/bin/env python3
"""Compare scri magnitudes over exactly matched unions of recorded stage windows.

The shared window can exceed the nominal output cadence because different
timesteps cross output times on different cycles. Min/max unions are exact;
interpolation of extrema and reconstruction of combined pointwise residuals
from separate reduced columns are never used.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scri_compare_endpoints import extrema_window
from scri_run_summary import history
from scri_norm_convergence import decay


def compare(roots, halfdt=None):
    roots=list(map(Path,roots));all_roots=roots+([Path(halfdt)] if halfdt else [])
    receipts=[json.loads((root/'final-checkpoint.json').read_text()) for root in all_roots]
    ns=[];identities=[];times=[];hashes=[]
    for root,receipt in zip(all_roots,receipts):
        match=re.search(r';n=(\d+),(\d+),(\d+);',receipt['identity'])
        assert match and len(set(match.groups()))==1
        ns.append(int(match.group(1)))
        identities.append(re.sub(r';n=\d+,\d+,\d+|;p.requested_dt=[^;]*','',receipt['identity']))
        group_times=[];source={}
        for group in range(4):
            path=next(root.glob(f'*.scri-investigation-{group}.hst'))
            array,_=history(path);group_times.append(3*array[:,0])
            source[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(np.array_equal(t,group_times[0]) for t in group_times)
        assert abs(group_times[0][-1]-receipt['time_M0'])<1e-9
        times.append(group_times[0]);hashes.append(source)
    assert ns==[12,16,20]+([12] if halfdt else [])
    assert len(set(identities))==1, 'different numerical physics or boundary treatment'
    assert max(t[-1] for t in times)-min(t[-1] for t in times)<1e-9
    dt=np.array([r['dt_native'] for r in receipts]);assert np.isfinite(dt).all() and np.all(dt>0)
    np.testing.assert_allclose(dt[:3]*[12,16,20],dt[0]*12,rtol=1e-14,atol=0)
    if halfdt:np.testing.assert_allclose(2*dt[3],dt[0],rtol=1e-14,atol=0)
    common=[t for t in times[0][:-1] if all(np.any(abs(other[:-1]-t)<1e-9) for other in times[1:])]
    assert common, 'no shared recorded beginning for final window'
    begin=float(max(common));end=float(times[0][-1]);magnitudes=[];sources=[]
    for root in all_roots:
        signed,absolute,source=extrema_window(root,begin,end)
        magnitudes.append({**{k:max(abs(v['min']),abs(v['max'])) for k,v in signed.items()},
                           **{k:v['max'] for k,v in absolute.items()}})
        sources.append(source)
    assert all(v.keys()==magnitudes[0].keys() for v in magnitudes)
    metrics={k:decay([m[k] for m in magnitudes[:3]],magnitudes[3][k] if halfdt else None)
             for k in magnitudes[0]}
    for metric in metrics.values():
        metric['meaning']='Decay of scri diagnostic magnitude over a matched time window; not solution-field convergence or proof of regularity.'
    return dict(window_M0=[begin,end],resolutions=[12,16,20],metrics=metrics,
                halfdt_source=str(halfdt) if halfdt else None,
                sources=[dict(directory=str(root),final_checkpoint=r,history_sha256=h,windows=w)
                         for root,r,h,w in zip(all_roots,receipts,hashes,sources)],
                limitation='Subwindows are joined only by exact min/max unions with complete coverage. Solved residuals near roundoff do not establish accuracy.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs=3,type=Path)
    parser.add_argument('--halfdt',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compare(args.runs,args.halfdt)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(window_M0=result['window_M0'],metrics=len(result['metrics'])),indent=2))
