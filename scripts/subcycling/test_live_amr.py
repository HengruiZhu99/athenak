"""Exercise real refine/derefine and restart with the opt-in subcycling driver."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('output',type=Path)
p.add_argument('--mixed',action='store_true')
p.add_argument('--production-orders',action='store_true',help='Use Brill spatial_order=4 and extrap_order=2')
a=p.parse_args();exe=a.exe.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[2]
s=(repo/'tst/inputs/z4c_vc_minkowski_full_constraint_bjorhus.athinput').read_text()
if a.production_orders:
    s=s.replace('spatial_order = 6','spatial_order = 4')
    s=s.replace('<z4c>','<z4c>\nextrap_order = 2')
s=s.replace('refinement = none','refinement = adaptive\nnum_levels = 2\nmax_nmb_per_rank = 64\nrefinement_interval = 1')
s=s.replace('integrator = rk4','integrator = rk4_classical\nsubcycle_max_ratio = 2\nsubcycle_interval_cap = 0.002\nsubcycle_cycle_unit = synchronization')
s=s.replace('nlim = 3','nlim = 4').replace('tlim = 0.01','tlim = 0.008')
s=s.replace('<output1>','<output1>\ndata_format = %24.16e')
s=s.replace('<z4c>','''<z4c>
telegraph_lapse = true
telegraph_damping_prescription = max_domain_abs_K
telegraph_tau = 1
telegraph_kappa = 1
shift_mode = prescribed_zero
target_kappa1 = 0''')
s=s.replace('<problem>','''<problem>
lapse_gaussian_amplitude = 0.1
exercise_deterministic_amr = true
amr_target_count = 2
amr_target_lx1 = 0
amr_target_lx2 = 0
amr_target1_lx1 = 0
amr_target1_lx2 = 1''')
s+='''
<amr_criterion1>
method = user
<output2>
file_type = rst
dcycle = 1
'''
if a.mixed:
    s=s.replace('amr_target_count = 2','amr_target_count = 1')
    s=s.replace('<problem>','<problem>\nexercise_mixed_amr = true\namr_mixed_refine_lx1 = 1\namr_mixed_refine_lx2 = 1')
expected_blocks=[4,7,7,7] if a.mixed else [4,10,4,4]
def run(name,ratio,rst=None,nlim=4):
    out=root/name;out.mkdir();(out/'input.athinput').write_text(s)
    args=['-r',str(rst)] if rst else ['-i','input.athinput']
    args += [f'time/subcycle_max_ratio={ratio}',f'time/nlim={nlim}']
    with (out/'run.log').open('w') as log:
        subprocess.run([str(exe),*args],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    return out
# Validate payload size before decoding; mixed case retains a refined family.
def fields(rst):
    raw=rst.read_bytes();nb,nv,nn=expected_blocks[-1],25,25;block_bytes=nv*nn*nn*8
    offset=len(raw)-nb*block_bytes
    assert int.from_bytes(raw[offset-8:offset],byteorder='little')==block_bytes
    return np.frombuffer(raw,dtype=np.float64,offset=offset).reshape(nb,nv,nn,nn)[:,:,4:21,4:21].copy()
results=[]
for ratio in [1,2]:
    whole=run(f'whole{ratio}',ratio)
    log=(whole/'run.log').read_text()
    assert ('6 MeshBlocks created, 3 deleted by AMR' if a.mixed else '6 MeshBlocks created, 6 deleted by AMR') in log
    steps=np.loadtxt(whole/'subcycling_intervals.csv',delimiter=',',skiprows=1,ndmin=2)
    assert list(steps[:,6])==expected_blocks,steps
    expected=fields(sorted((whole/'rst').glob('*.rst'))[-1])
    assert np.all(np.isfinite(expected))
    partial=run(f'partial{ratio}',ratio,nlim=2 if a.mixed else 1)
    checkpoint=sorted((partial/'rst').glob('*.rst'))[-1]
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    resumed=run(f'resumed{ratio}',ratio,checkpoint)
    actual=fields(sorted((resumed/'rst').glob('*.rst'))[-1])
    error=float(np.max(np.abs(actual-expected)))
    assert error<1e-12,(ratio,error)
    hst=np.loadtxt(sorted(resumed.glob('*.hst'))[-1],ndmin=2)
    assert abs(hst[-1,0]-.008)<1e-14
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    results.append(dict(ratio=ratio,blocks=steps[:,6].astype(int).tolist(),restart_max=error,end=float(hst[-1,0])))
(root/'results.json').write_text(json.dumps(dict(executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),results=results),indent=2)+'\n')
print((root/'results.json').read_text())
