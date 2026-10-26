#!/usr/bin/env python3
"""Compare boundary/time controls at common times without claiming interpolation is exact."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

KEYS=['H_l2','M_conformal_l2','Z_conformal_l2','H_linf','min_alpha','min_chi']


def interpolate(data,time):
    ts=np.array([float(r['time']) for r in data]);assert np.all(np.diff(ts)>0)
    exact=np.flatnonzero(ts==time)
    if len(exact):
        return {'method':'exact recorded sample','values':{k:float(data[exact[0]][k]) for k in KEYS},
                'linear_cubic_difference_indicator':{k:0. for k in KEYS}}
    if not ts[0]<time<ts[-1]:return None
    index=int(np.searchsorted(ts,time));lo,hi=index-1,index
    weight=(time-ts[lo])/(ts[hi]-ts[lo])
    start=max(0,min(index-2,len(ts)-4));indices=list(range(start,min(start+4,len(ts))))
    weights=[]
    for i in indices:
        w=1.
        for j in indices:
            if i!=j:w*=(time-ts[j])/(ts[i]-ts[j])
        weights.append(w)
    assert abs(sum(weights)-1)<1e-12
    linear={k:(1-weight)*float(data[lo][k])+weight*float(data[hi][k]) for k in KEYS}
    cubic={k:sum(w*float(data[i][k]) for w,i in zip(weights,indices)) for k in KEYS}
    return {'method':'local cubic interpolation of recorded scalar diagnostics',
            'bracketing_times':[float(ts[lo]),float(ts[hi])],'values':cubic,
            'linear_cubic_difference_indicator':{k:abs(linear[k]-cubic[k]) for k in KEYS}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('histories',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();series={};inputs={}
    for name in ['degree2','degree4','half-degree2','half-degree4']:
        path=a.histories/(name+'-history.csv')
        if not path.exists():continue
        data=list(csv.DictReader(path.open()));series[name]={region:[r for r in data if r['region']==region] for region in ['bulk','shell']}
        inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    samples={};comparisons={}
    for time in [.05,.1,.2,.4,.6,.7927734374996852,1.,1.5837303635638276,2.]:
        name=str(time);samples[name]={};comparisons[name]={}
        for case,regions in series.items():
            values={region:interpolate(data,time) for region,data in regions.items()}
            if all(v is not None for v in values.values()):samples[name][case]=values
        for pair in [('degree2','degree4'),('half-degree2','half-degree4'),('degree2','half-degree2'),('degree4','half-degree4')]:
            if not all(case in samples[name] for case in pair):continue
            result={}
            for region in ['bulk','shell']:
                first,second=[samples[name][case][region] for case in pair];result[region]={}
                for key in KEYS:
                    denominator=max(abs(first['values'][key]),abs(second['values'][key]),1e-30)
                    result[region][key]={'relative_difference':abs(first['values'][key]-second['values'][key])/denominator,
                        'relative_interpolation_indicator':sum(x['linear_cubic_difference_indicator'][key] for x in [first,second])/denominator,
                        'both_exact_samples':all(x['method']=='exact recorded sample' for x in [first,second])}
            comparisons[name][' versus '.join(pair)]=result
    result={'input_sha256':inputs,'scope':'Scalar diagnostic comparisons only, not evolved-field or solution convergence. Linear/cubic differences are heuristic indicators, not rigorous error bounds. Decisive final t=2 comparison must use exact recorded samples.',
            'samples':samples,'comparisons':comparisons}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({time:list(pairs) for time,pairs in comparisons.items()}))


if __name__=='__main__':main()
