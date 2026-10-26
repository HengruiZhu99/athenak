#!/usr/bin/env python3
"""CPU/GPU admission of surface, horizon-candidate and gauge diagnostics."""
import argparse,csv,json,math
from pathlib import Path

def compare(cpu,gpu):
    names=json.loads((cpu/'summary.json').read_text())['cases'].keys();checks=[]
    for name in names:
        for filename,keys in [('cubed_sphere_surfaces.csv',['time','cycle','r']),
                              ('cubed_sphere_horizon_candidates.csv',['time','cycle','found']),
                              ('cubed_sphere_gauge_monitor.csv',['time','cycle','region','subset'])]:
            a=list(csv.DictReader((cpu/name/filename).open()));b=list(csv.DictReader((gpu/name/filename).open()))
            assert len(a)==len(b),(name,filename,'row count')
            for x,y in zip(a,b):
                # Coordinate arithmetic can differ by one ULP across devices.
                assert x.keys()==y.keys() and all(
                    math.isclose(float(x[k]),float(y[k]),rel_tol=1e-14,abs_tol=1e-15)
                    if k=='r' else x[k]==y[k] for k in keys),(name,filename,'row identity')
                for field in x.keys()-set(keys):
                    u,v=float(x[field]),float(y[field])
                    if math.isnan(u) and math.isnan(v):
                        assert filename=='cubed_sphere_horizon_candidates.csv' and x['found']=='0';continue
                    assert math.isfinite(u) and math.isfinite(v)
                    error=abs(u-v);tolerance=1e-10+1e-6*max(abs(u),abs(v))
                    checks.append(dict(case=name,file=filename,field=field,error=error,tolerance=tolerance,passed=error<=tolerance))
    return dict(passed=all(x['passed'] for x in checks),checks=checks,scope='CPU/GPU diagnostics admission only; horizon candidate needs angular residual/refinement for physical acceptance.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('gpu',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=compare(a.cpu,a.gpu);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('passed',r['passed']);raise SystemExit(0 if r['passed'] else 1)
