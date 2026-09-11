"""Isolated real-restart entry test; does not qualify the production Brill run."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('output',type=Path)
p.add_argument('--dt',type=float,default=2e-3)
a=p.parse_args();exe=a.exe.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[2]
s=(repo/'tst/inputs/z4c_vc_minkowski_full_constraint_bjorhus.athinput').read_text()
s=s.replace('nlim = 3','nlim = 1').replace('tlim = 0.01','tlim = 1.0')
s=s.replace('<time>','<time>\nsubcycle_probe_dt = 1e-4\nsubcycle_probe_ratio = 2')
s=s.replace('refinement = none','refinement = static\nnum_levels = 1')
s=s.replace('<z4c>','''<z4c>
telegraph_lapse = true
telegraph_damping_prescription = max_domain_abs_K
telegraph_tau = 1
telegraph_kappa = 1
shift_mode = prescribed_zero
target_kappa1 = 0''')
s=s.replace('<problem>','<problem>\nlapse_gaussian_amplitude = 0.1')
s+='''
<refined_region1>
level = 1
x1min = 0
x1max = 1
x2min = -1
x2max = 0
<output2>
file_type = rst
dcycle = 1
'''
seed=root/'seed';seed.mkdir();(seed/'input.athinput').write_text(s)
def run(args,cwd,env=None):
    with (cwd/'run.log').open('w') as log:
        subprocess.run([str(exe),*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
run(['-i','input.athinput'],seed)
checkpoint=sorted((seed/'rst').glob('*.rst'))[-1];before=sha(checkpoint)
results=[]
for dt in [a.dt,a.dt/2,a.dt/4]:
    arrays={}
    for ratio in [1,2]:
        case=root/f'dt{dt}_ratio{ratio}';case.mkdir();output=case/'probe'
        env=os.environ.copy();env['ATHENA_TEST_SUBCYCLE_INTERVAL_DIR']=str(output)
        run(['-r',str(checkpoint),f'time/subcycle_probe_dt={dt}',f'time/subcycle_probe_ratio={ratio}'],case,env)
        meta=dict(line.split('=',1) for line in (output/'probe.txt').read_text().splitlines())
        dtype={8:np.float64,4:np.float32}[int(meta['real_bytes'])]
        values=np.fromfile(output/'fields.bin',dtype=dtype)
        assert values.size==int(meta['leaves'])*int(meta['variables'])*int(meta['ni'])*int(meta['nj'])
        assert np.all(np.isfinite(values))
        arrays[ratio]=values
        results.append(dict(dt=dt,ratio=ratio,metadata=meta,fields_sha256=sha(output/'fields.bin')))
    delta=arrays[2]-arrays[1]
    results.append(dict(dt=dt,comparison='ratio2 minus ratio1',rms=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.max(np.abs(delta)))))
assert sha(checkpoint)==before
(root/'results.json').write_text(json.dumps(dict(checkpoint_sha256=before,executable_sha256=sha(exe),scope='local refined gauge-pulse checkpoint; not Brill reproduction or speedup',results=results),indent=2)+'\n')
print((root/'results.json').read_text())
