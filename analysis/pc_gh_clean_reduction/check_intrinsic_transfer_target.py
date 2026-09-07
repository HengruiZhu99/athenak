#!/usr/bin/env python3
"""All-cell polynomial exactness and primary-only reads for intrinsic halo targets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);points=[];expected=[];cases=[]
for order in [2,4,6]:
    for dim in [2,3]:
        start=len(points)
        for k in range(16 if dim==3 else 1):
            for j in range(16):
                for i in range(16):
                    points.append([order,dim,k,j,i]);x=(i-3.5)*.125;y=(j-3.5)*.1625;z=(k-3.5)*.2125 if dim==3 else 0
                    w=1+.01*(x+2*y+3*z);rho=1.2+.02*(2*x-y+z)
                    dw=np.array([.01,.02,.03]);drho=np.array([.04,-.02,.02])
                    gradients=np.empty((10,3));gradients[0]=dw;gradients[1]=rho*dw+w*drho
                    for n in range(2,10):
                        gradients[n]=.001*np.array([(n+1)*order*x**(order-1)+.1*(n+1)*y,
                            .5*(n+2)*order*y**(order-1)+.1*(n+1)*x,.25*(n+3)*order*z**(order-1)])
                    if dim==2:gradients[:,2]=0
                    expected.append(np.concatenate([gradients[0],gradients[1],gradients[2:7].T.reshape(-1),gradients[7:10].T.reshape(-1)]))
        cases.append(dict(order=order,dimensions=dim,start=start,end=len(points)))
np.savetxt(a.output/'input.txt',points,fmt='%d',header=str(len(points)),comments='')
command=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())]
with (a.output/'run.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert r.returncode==0
observed=np.loadtxt(a.output/'output.txt');expected=np.array(expected)
assert observed.shape==expected.shape and np.isfinite(observed).all()
error=abs(observed-expected)/(1+abs(expected));records=[]
for case in cases:
    e=error[case['start']:case['end']]
    record=dict(order=case['order'],dimensions=case['dimensions'],points=len(e),max_error=float(e.max()),per_auxiliary_max=e.max(axis=0).tolist(),status='PASS' if e.max()<=2e-12 else 'FAIL')
    records.append(record);print(json.dumps(record),flush=True)
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
(a.output/'manifest.json').write_text(json.dumps(dict(command=command,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),point_count=len(points),comparison_count=expected.size),indent=2)+'\n')
assert all(r['status']=='PASS' for r in records)
