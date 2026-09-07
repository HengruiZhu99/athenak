#!/usr/bin/env python3
"""Analytic rotated shear controls for the primary-only compiled H/M stencil."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--fixture',choices=['shear','curved'],default='shear')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[];expected=[];metadata=[]
for dim in [2,3]:
    normal=np.array([.6,.8,0] if dim==2 else [2/7,3/7,6/7])
    for order in [2,4,6]:
        for n in [16,32,64]:
            for phase in np.arange(17)*2*np.pi/17:
                rows.append([order,dim,n,phase,order//2,int(a.fixture=='curved')]);metadata.append((dim,order,n))
                w=1+.1*np.cos(phase);rho=.8+.03*np.sin(phase)
                dw=-.1*np.sin(phase);ddw=-.1*np.cos(phase);k=.03*np.sin(phase);dk=.03*np.cos(phase)
                h=2/3*k*k-2*.02**2+4*w*ddw-6*dw*dw
                m=normal*(3*.02*dw/w-2/3*dk)
                if a.fixture=='curved':
                    # g=e^(2f)vv^T+e^(-2f)tt^T+nn^T, f=.12 cos(theta):
                    # R[g]=-2(f')^2, det(g)=1, Laplacian_g(w)=w'', A=0.
                    h=2/3*k*k-2*w*w*(.12*np.sin(phase))**2+4*w*ddw-6*dw*dw
                    m=normal*(-2/3*dk)
                expected.append([h,*m,*(rho*w*m)])
rows.append([6,3,32,.27,2,int(a.fixture=='curved')])  # insufficient FD6 reach
np.savetxt(a.output/'input.txt',rows,fmt='%.17g',header=str(len(rows)),comments='')
command=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())]
with (a.output/'run.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert r.returncode==0
raw=np.loadtxt(a.output/'output.txt');assert raw.shape==(len(rows),7)
assert not np.isfinite(raw[-1]).all()
values=raw[:-1];assert np.isfinite(values).all();expected=np.array(expected);records=[]
for dim in [2,3]:
    for order in [2,4,6]:
        errors=[]
        for n in [16,32,64]:
            mask=np.array([x==(dim,order,n) for x in metadata]);delta=abs(values[mask]-expected[mask])
            errors.append([float(delta[:,0].max()),float(delta[:,1:4].max()),float(delta[:,4:].max())])
        errors=np.array(errors);rates=np.log2(errors[:-1]/errors[1:])
        record=dict(fixture=a.fixture,dimensions=dim,order=order,N=[16,32,64],components=['H','M','alpha_M'],errors=errors.tolist(),observed_orders=rates.tolist(),minimum_order=order-.6,status='PASS' if rates.min()>=order-.6 else 'FAIL')
        records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
        assert record['status']=='PASS'
for row,value in zip(rows[:-1],values):
    phase=row[3];alpha=(1+.1*np.cos(phase))*(.8+.03*np.sin(phase))
    assert np.max(abs(value[4:]-alpha*value[1:4]))<2e-12
(a.output/'manifest.json').write_text(json.dumps(dict(status='PASS',command=command,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),positive_points=len(values),insufficient_halo_rejected=True),indent=2)+'\n')
