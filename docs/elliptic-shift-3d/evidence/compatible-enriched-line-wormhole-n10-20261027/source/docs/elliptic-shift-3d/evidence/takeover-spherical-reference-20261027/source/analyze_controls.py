#!/usr/bin/env python3
"""Assess complete histories and independent controls, never just final errors."""
import argparse,json
from pathlib import Path
import numpy as np
from numerics import weights
from kernel import PARITY

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def read(name):
    folder=a.input/name;m=json.loads((folder/'manifest.json').read_text())
    rows=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as z:r=z['r'].copy();q=z['q'].copy()
    return dict(manifest=m,rows=rows,r=r,q=q)
def norm(q,r,mask):return float(np.sqrt(np.sum(r[mask,None]**2*q[mask,:7]**2)/np.sum(r[mask]**2)))
def interpolate(z,r):
    fine=z['r'];q=z['q'];width=9;ext=np.r_[-fine[:width][::-1],fine]
    values=np.r_[q[:width][::-1]*PARITY,q];out=[]
    for x in r:
        start=min(max(np.searchsorted(ext,x)-width//2,0),len(ext)-width)
        out.append(weights(ext[start:start+width],x)@values[start:start+width])
    return np.array(out)
records={}
for folder in a.input.iterdir():
    if not folder.is_dir() or not (folder/'checkpoint.npz').exists():continue
    z=read(folder.name);rows=z['rows'];m=z['manifest']
    maxima={key:max(abs(x[key]) for x in rows) for key in
        ['scri_clock','scri_null','scri_pole','scri_lapse_pole','H_rms','M_rms','H_endpoint_max','M_endpoint_max']}
    records[folder.name]=dict(status=m['status'],time=m['time'],maxima=maxima,
        preliminary_boundary_gates=bool(maxima['scri_clock']<=1e-13 and maxima['scri_null']<=1e-12 and
            maxima['scri_pole']<=1e-7 and maxima['scri_lapse_pole']<=1e-7),
        runtime=m['runtime_seconds'],binding=m['binding'])
spatial={};zs=[read(f'spatial-n{n}') for n in [256,512,1024]]
for region in ['bulk','endpoint']:
    errors=[]
    for lo,hi in zip(zs[:-1],zs[1:]):
        r=lo['r'];mask=r<.8 if region=='bulk' else r>.9
        errors.append(norm(lo['q']-interpolate(hi,r),r,mask))
    spatial[region]=dict(differences=errors,order=float(np.log2(errors[0]/errors[1])))
baseline=zs[0];r=baseline['r'];allpoints=np.ones(len(r),dtype=bool)
times=[read('time-cfl0.1'),baseline,read('time-cfl0.025'),read('time-cfl0.0125')]
errors=[norm(x['q']-y['q'],r,allpoints) for x,y in zip(times[:-1],times[1:])]
temporal=dict(differences=errors,orders=np.log2(np.array(errors[:-1])/np.array(errors[1:])).tolist(),
    standard_CFL_error_upper_bound=errors[1],fraction_of_middle_spatial_difference=errors[1]/spatial['bulk']['differences'][0])
quadrature={}
for order in [5,9,11]:
    z=read('quadrature-q'+str(order));delta=z['q']-baseline['q']
    quadrature[order]=dict(max_difference=float(np.max(abs(delta))),rms_difference=norm(delta,r,allpoints))
record=dict(scope='Short controls; every failed gate retained. No claim of complete literature reproduction.',
    runs=records,spatial=spatial,temporal=temporal,quadrature=quadrature)
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({key:record[key] for key in ['spatial','temporal','quadrature']}))
