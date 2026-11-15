#!/usr/bin/env python3
"""Compare common-point samples at matched times on N12/N16/N20.

Reported p is an effective three-grid order, conditional on a dominant aligned
error pattern. Width sensitivity is an estimate, not a rigorous error bound.
Field sampling RMS is unweighted over the fixed sample locations/components;
native exterior/shell constraint histories use their own volume quadratures.
"""
import argparse,json,re
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

GROUPS={'chi':[0], 'metric':list(range(1,7)), 'A':list(range(7,13)), 'P':[13],
        'Theta':[14], 'Lambda':list(range(15,18)), 'alpha':[18],
        'B':list(range(19,22)), 'beta':list(range(22,25)), 'D':list(range(25,28))}

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))

def ratio(p):
    a,b=np.log(16/12),np.log(20/16)
    return a/b if abs(p)<1e-8 else np.expm1(p*a)/(-np.expm1(-p*b))

def statistics(values, alternatives, timestep=None):
    low,high=values[0]-values[1],values[1]-values[2]
    d0,d1=rms(low),rms(high)
    sensitivity=[rms(a-b) for a,b in zip(values,alternatives)]
    threshold=1e-13*max(1.,max(rms(v) for v in values))
    order=alignment=None
    if min(d0,d1)>threshold:
        alignment=float(np.mean(low*high)/(d0*d1))
        observed=d0/d1
        if ratio(-30)<observed<ratio(30):
            order=float(brentq(lambda p:ratio(p)-observed,-30,30))
    interpolation=[(sensitivity[0]+sensitivity[1])/d0 if d0 else None,
                   (sensitivity[1]+sensitivity[2])/d1 if d1 else None]
    consistent=bool(order is not None and order>0 and alignment>.9 and
                    all(v is not None and v<.1 for v in interpolation))
    dt_difference=rms(values[0]-timestep) if timestep is not None else None
    return dict(d12_16_rms=d0,d16_20_rms=d1,
        d12_16_max=float(np.max(abs(low))),d16_20_max=float(np.max(abs(high))),
        effective_order=order,difference_alignment_cosine=alignment,
        sampling_width_sensitivity_rms=sensitivity,sampling_to_difference_ratios=interpolation,
        positive_aligned_order_above_sampling_sensitivity=consistent,
        N12_half_dt_difference_rms=dt_difference,
        N12_timestep_to_spatial_difference=dt_difference/d0 if dt_difference is not None and d0 else None)

def read(path):
    metadata=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:
        arrays={k:z[k] for k in z.files}
    return metadata,arrays

def validate_receipt(metadata, path):
    receipt=json.loads(Path(path).read_text())
    assert receipt['sha256']==metadata['checkpoint_sha256'], 'sample/checkpoint hash mismatch'
    assert receipt['cycle']==metadata['cycle'], 'sample/checkpoint cycle mismatch'
    assert abs(receipt['time_M0']-metadata['time_M0'])<1e-9, 'sample/checkpoint time mismatch'
    match=re.search(r';n=(\d+),(\d+),(\d+);',receipt['identity'])
    assert match and set(map(int,match.groups()))=={metadata['n']}, 'sample/checkpoint resolution mismatch'
    common=re.sub(r';n=\d+,\d+,\d+|;p.requested_dt=[^;]*','',receipt['identity'])
    assert common==metadata['physics_identity_without_resolution_and_dt'], 'sample/checkpoint physics mismatch'
    dt=float(receipt['dt_native'])
    assert np.isfinite(dt) and dt>0, 'invalid native timestep'
    return dict(path=str(Path(path)),checkpoint_sha256=receipt['sha256'],dt_native=dt)

def compare(paths, halfdt=None, receipts=None, halfdt_receipt=None):
    records=[read(p) for p in paths]
    assert [r[0]['n'] for r in records]==[12,16,20]
    identities=[r[0]['physics_identity_without_resolution_and_dt'] for r in records]
    assert len(set(identities))==1, 'different PDE, boundary, or numerical settings'
    times=[r[0]['time_M0'] for r in records]
    assert max(times)-min(times)<1e-9,('unmatched physical times',times)
    for _,data in records[1:]:
        for key in ('coordinates','radii','angles'):
            assert np.array_equal(data[key],records[0][1][key]),key
    verified=None
    if receipts is not None:
        assert len(receipts)==3, 'provide one native receipt for each resolution'
        verified=[validate_receipt(record[0],path) for record,path in zip(records,receipts)]
        np.testing.assert_allclose([v['dt_native']*n for v,n in zip(verified,(12,16,20))],
                                   verified[0]['dt_native']*12,rtol=1e-14,atol=0)
    half=None
    half_verified=None
    if halfdt:
        assert verified is not None and halfdt_receipt is not None, 'half-dt analysis requires all four native checkpoint receipts'
        meta,half=read(halfdt)
        assert meta['n']==12 and abs(meta['time_M0']-times[0])<1e-9
        assert meta['physics_identity_without_resolution_and_dt']==identities[0]
        for key in ('coordinates','radii','angles'):
            assert np.array_equal(half[key],records[0][1][key]),key
        half_verified=validate_receipt(meta,halfdt_receipt)
        np.testing.assert_allclose(2*half_verified['dt_native'],verified[0]['dt_native'],rtol=1e-14,atol=0)
    else:
        assert halfdt_receipt is None, 'half-dt receipt requires half-dt samples'
    result=dict(time_M0=times,scope='fixed common samples on all six faces; r=.9431..1',
                norm='unweighted sample/component RMS; not a physical-volume norm',
                criteria='effective p>0, error-vector cosine>.9, both sampling sensitivities <.1 of grid difference; descriptive, not proof of convergence',
                sources=[str(p) for p in paths],native_receipts=verified,
                halfdt_source=str(halfdt) if halfdt else None,
                halfdt_native_receipt=half_verified,regions={})
    for region,selection in [('near_scri',slice(None,-1)),('scri',slice(-1,None))]:
        result['regions'][region]={}
        for group,fields in GROUPS.items():
            values=[r[1]['fields_w6'][:,selection,...,fields] for r in records]
            alternatives=[r[1]['fields_w8'][:,selection,...,fields] for r in records]
            dt=half['fields_w6'][:,selection,...,fields] if half is not None else None
            result['regions'][region][group]=statistics(values,alternatives,dt)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples',nargs=3,type=Path,help='N12, N16, N20 samples, in that order')
    p.add_argument('--halfdt',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipts',nargs=3,type=Path,help='Native final checkpoint receipts in N12, N16, N20 order; required with --halfdt')
    p.add_argument('--halfdt-receipt',type=Path,help='Native final checkpoint receipt for --halfdt')
    args=p.parse_args();result=compare(args.samples,args.halfdt,args.receipts,args.halfdt_receipt)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
