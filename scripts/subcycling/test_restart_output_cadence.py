"""Verify explicit physical-time output overrides inherited restart dcycle."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();exe=a.exe.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
subprocess.run([sys.executable,str(Path(__file__).with_name('test_checkpoint_probe.py')),str(exe),str(root/'fixture'),'--seed-only','--production-orders'],check=True)
seed=root/'fixture/seed';checkpoint=sorted((seed/'rst').glob('*.rst'))[-1]
start=np.loadtxt(next(seed.glob('*.hst')),ndmin=2)[-1,0];end=start+.008
results=[]
for mode in ['auto','time']:
    case=root/mode;case.mkdir()
    text=f'''<time>
integrator = rk4_classical
subcycle_max_ratio = 2
subcycle_interval_cap = .002
nlim = -1
tlim = {end:.17g}
'''
    for n in [1,2]:
        text+=f'''<output{n}>
cadence = {mode}
dcycle = 0
dt = .002
last_time = {start:.17g}
file_number = 0
'''
    (case/'override.athinput').write_text(text)
    with (case/'run.log').open('w') as log:
        subprocess.run([str(exe),'-r',str(checkpoint),'-i','override.athinput'],cwd=case,stdout=log,stderr=subprocess.STDOUT,check=True)
    histories=list(case.glob('*.hst'));checkpoints=list((case/'rst').glob('*.rst'))
    if mode=='auto':assert not histories and not checkpoints
    else:
        assert len(histories)==1 and len(checkpoints)==4
        t=np.loadtxt(histories[0],ndmin=2)[:,0]
        assert len(t)==4 and np.max(np.abs(t-(start+.002*np.arange(1,5))))<1e-14,t
    results.append(dict(cadence=mode,histories=len(histories),checkpoints=len(checkpoints)))
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print((root/'results.json').read_text())
