#!/usr/bin/env python3
"""Measure the physical effect of grid-scaled scalar reduction damping."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scalar_evidence import history, snapshot
from assess_minkowski_matrix import norms

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed',nargs=3,type=Path,required=True)
    p.add_argument('--scaled',nargs=2,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise RuntimeError('Existing review is immutable')
    h=[history(d) for d in a.fixed];checks=[]
    for t in [2,6,12]:
        differences=[]
        for fixed,scaled in zip(a.fixed[1:],a.scaled):
            f,s=snapshot(fixed,t),snapshot(scaled,t)
            if not np.array_equal(f[:,:4],s[:,:4]):raise RuntimeError('Changed grid or weights')
            differences.append({name:norms(f[:,cols]-s[:,cols],f[:,3]) for name,cols in [('psi',slice(4,5)),('v',slice(5,6)),('Dpsi',slice(9,12))]})
        for field in ['psi','v','Dpsi']:
            for norm in ['L2','Linf']:
                delta=[d[field][norm] for d in differences]
                uncertainty=[abs(float(h[k][t,'all'][field+'_'+norm])-float(h[k+1][t,'all'][field+'_'+norm])) for k in [0,1]]
                checks.append(dict(time=t,field=field,norm=norm,differences=delta,adjacent_fixed_error_gap=uncertainty,passed=delta[1]<delta[0] and all(d<=u for d,u in zip(delta,uncertainty))))
    result=dict(passed=all(c['passed'] for c in checks),fixed=[str(d) for d in a.fixed],scaled=[str(d) for d in a.scaled],checks=checks,definition='Physical field differences must decrease from N32 to N40 and be no larger than the adjacent fixed-kappa analytic-error gap, in both norms at each declared time.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],failed=[c for c in checks if not c['passed']]),indent=2))
if __name__=='__main__':main()
