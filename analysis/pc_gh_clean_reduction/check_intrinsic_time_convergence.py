#!/usr/bin/env python3
"""Fixed-grid nonlinear RK3 self-convergence; not an Einstein-data qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart,global_state
from run_legacy_equivalence import input_text
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);runs=[];states=[]
for dt in [.001,.0005,.00025]:
    steps=round(.02/dt);folder=a.output/f'dt{dt}';folder.mkdir()
    text=input_text(6,2,'collision_factorized',1)
    start=text.index('<mesh>');end=text.index('<meshblock>')
    text=text[:start]+text[start:end].replace('nx1 = 8','nx1 = 16').replace('nx2 = 8','nx2 = 16')+text[end:]
    text=text.replace('nlim = 1',f'nlim = {steps}').replace('tlim = 0.0001','tlim = 0.02')
    start=text.index('<pc_gh>');end=text.index('<problem>')
    text=text[:start]+f'''<pc_gh>
formulation = intrinsic_clean
spatial_order = 6
shift_eta = 2
kappa = 1
reduction_rate = 1
reduction_profile = lapse_scaled
dissipation = 0.3
research_dt_ceiling = {dt}
intrinsic_diagnostics = true
intrinsic_diagnostic_dcycle = {steps//4}
'''+text[end:].replace('legacy_equivalence','intrinsic_smooth')
    text+='\n<output1>\nfile_type = rst\ndt = 0.02\n'
    inp=folder/'used.athinput';inp.write_text(text)
    command=[str(a.binary.resolve()),'-i',str(inp.resolve())];start=time.time()
    with (folder/'run.log').open('w') as log:r=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    record=dict(dt=dt,steps=steps,command=command,wall_seconds=time.time()-start,returncode=r.returncode,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())
    runs.append(record);(a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert r.returncode==0
    path=sorted((folder/'rst').glob('*.rst'))[-1];data=read_restart(path)
    assert list(data['mesh'][1:4])==[16,16,1]
    assert data['cycle']==steps and abs(data['time']-.02)<1e-14
    assert len(list(folder.glob('intrinsic-diagnostics-*.csv')))==5
    states.append(global_state(data));record['restart_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');print(dt,'completed',flush=True)
first=states[0]-states[1];second=states[1]-states[2];groups={}
for name,count in [('primary_curvature_GH',20),('all_fields',50)]:
    d1=first[:count].reshape(-1);d2=second[:count].reshape(-1)
    n1=np.sqrt(np.sum(d1*d1));n2=np.sqrt(np.sum(d2*d2))
    order=float(np.log2(n1/n2));alignment=float(np.sum(d1*d2)/(n1*n2))
    groups[name]=dict(difference_L2=[float(n1),float(n2)],observed_order=order,alignment=alignment,status='PASS' if order>=2.7 and alignment>=.99 else 'FAIL')
np.savez(a.output/'signed-temporal-differences.npz',coarse_minus_medium=first,medium_minus_fine=second)
result=dict(status='PASS' if all(g['status']=='PASS' for g in groups.values()) else 'FAIL',minimum_order=2.7,minimum_alignment=.99,groups=groups,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),scope='fixed-grid temporal convergence of the nonlinear off-constraint PDE fixture; not physical Einstein evolution qualification')
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert result['status']=='PASS'
