#!/usr/bin/env python3
"""Uniform spatial self-convergence with a separately suppressed temporal error."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart,global_state

def at_coarse_centers(field,target):
    out=field.copy()
    for axis in [-1,-2]:
        n=out.shape[axis];frequency=np.fft.fftfreq(n)*n
        location=(np.arange(target)+.5)/target-.5/n
        phase=2*np.pi*location[:,None]*frequency[None,:]
        evaluation=np.exp(1j*phase)
        if n%2==0:evaluation[:,n//2]=np.cos(phase[:,n//2])
        coefficients=np.moveaxis(np.fft.fft(out,axis=axis)/n,axis,0)
        result=np.tensordot(evaluation,coefficients,axes=(1,0))
        assert np.max(abs(result.imag))<5e-13
        out=np.moveaxis(result.real,0,axis)
    return out

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--template',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--orders',type=int,nargs='+',default=[2,4,6],choices=[2,4,6])
p.add_argument('--ko',type=float,default=.3)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);runs=[];records=[]
x=(np.arange(32)+.5)/32;y=x[:,None]
fixture=1+np.sin(2*np.pi*(2*x+3*y))+.2*np.cos(2*np.pi*x)
x=(np.arange(16)+.5)/16;y=x[:,None]
expected=1+np.sin(2*np.pi*(2*x+3*y))+.2*np.cos(2*np.pi*x)
interpolation_error=float(np.max(abs(at_coarse_centers(fixture,16)-expected)))
assert interpolation_error<5e-13

def run(order,n,dt):
    folder=a.output/f'fd{order}-n{n}-dt{dt}';folder.mkdir();steps=round(.02/dt)
    text=a.template.read_text();start=text.index('<mesh>');end=text.index('<meshblock>')
    mesh=text[start:end]
    for dim in [1,2]:mesh=re.sub(rf'(?m)^nx{dim} = .*$',f'nx{dim} = {n}',mesh)
    text=text[:start]+mesh+text[end:]
    for key,value in [('spatial_order',order),('research_dt_ceiling',dt),('nlim',steps),('intrinsic_diagnostic_dcycle',steps//4),('dissipation',a.ko)]:
        text=re.sub(rf'(?m)^{key} = .*$',f'{key} = {value}',text)
    inp=folder/'used.athinput';inp.write_text(text);command=[str(a.binary.resolve()),'-i',str(inp.resolve())];start=time.time()
    with (folder/'run.log').open('w') as log:r=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    record=dict(order=order,n=n,dt=dt,command=command,returncode=r.returncode,wall_seconds=time.time()-start,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())
    runs.append(record);(a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert r.returncode==0
    path=sorted((folder/'rst').glob('*.rst'))[-1];data=read_restart(path)
    assert data['cycle']==steps and abs(data['time']-.02)<1e-14 and list(data['mesh'][1:4])==[n,n,1]
    record['restart_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    print(order,n,dt,'completed',flush=True)
    return global_state(data)

for order in a.orders:
    states=[run(order,n,.000125) for n in [16,32,64]]
    fine_time=run(order,64,.0000625)
    common=[at_coarse_centers(u,16) for u in states]
    d1=common[0]-common[1];d2=common[1]-common[2]
    temporal=at_coarse_centers(fine_time-states[-1],16);groups={}
    for name,count in [('primary_curvature_GH',20),('all_fields',50)]:
        a1=d1[:count].reshape(-1);a2=d2[:count].reshape(-1)
        n1=np.sqrt(np.sum(a1*a1));n2=np.sqrt(np.sum(a2*a2))
        observed=float(np.log2(n1/n2));alignment=float(np.sum(a1*a2)/(n1*n2))
        contamination=float(np.sqrt(np.sum(temporal[:count]**2))/n2)
        groups[name]=dict(difference_L2=[float(n1),float(n2)],observed_order=observed,alignment=alignment,temporal_control_ratio=contamination,status='PASS' if observed>=order-.6 and alignment>=.99 and contamination<=.05 else 'FAIL')
    np.savez(a.output/f'fd{order}-signed-differences.npz',coarse_minus_medium=d1,medium_minus_fine=d2,temporal_control=temporal)
    record=dict(order=order,KO=a.ko,status='PASS' if all(g['status']=='PASS' for g in groups.values()) else 'FAIL',groups=groups,interpolation_error=interpolation_error)
    records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True);assert record['status']=='PASS'
(a.output/'summary.json').write_text(json.dumps(dict(status='PASS',binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),scope='uniform 2D nonlinear off-constraint spatial self-convergence with temporal control; no physical Einstein-data or interface qualification'),indent=2)+'\n')
