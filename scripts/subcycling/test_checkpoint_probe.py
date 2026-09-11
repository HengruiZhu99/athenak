"""Isolated real-restart entry test; does not qualify the production Brill run."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('output',type=Path)
p.add_argument('--uniform',action='store_true')
p.add_argument('--seed-only',action='store_true')
p.add_argument('--intervals',type=int,default=1)
p.add_argument('--dt',type=float,default=2e-3)
p.add_argument('--fixed-duration',type=float)
p.add_argument('--boundary-rhs',choices=['sommerfeld','full_constraint_bjorhus'],default='full_constraint_bjorhus')
p.add_argument('--production-orders',action='store_true',help='Use Brill spatial_order=4 and extrap_order=2')
p.add_argument('--production-gauge',action='store_true',help='Use Brill telegraph_tau=kappa=0.01')
a=p.parse_args();assert a.intervals>0;exe=a.exe.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[2]
s=(repo/'tst/inputs/z4c_vc_minkowski_full_constraint_bjorhus.athinput').read_text()
if a.production_orders:
    s=s.replace('spatial_order = 6','spatial_order = 4')
    s=s.replace('<z4c>','<z4c>\nextrap_order = 2')
s=s.replace('boundary_rhs = full_constraint_bjorhus','boundary_rhs = '+a.boundary_rhs)
s=s.replace('nlim = 3','nlim = 1').replace('tlim = 0.01','tlim = 1.0')
s=s.replace('<output1>','<output1>\ndata_format = %24.16e')
s=s.replace('<time>','<time>\nsubcycle_probe_dt = 1e-4\nsubcycle_probe_ratio = 2\nsubcycle_probe_duration = 0\nsubcycle_max_ratio = 0\nsubcycle_interval_cap = 0.002\nsubcycle_cycle_unit = synchronization')
s=s.replace('refinement = none','refinement = static\nnum_levels = 1')
s=s.replace('<z4c>','''<z4c>
telegraph_lapse = true
telegraph_damping_prescription = max_domain_abs_K
telegraph_tau = 1
telegraph_kappa = 1
shift_mode = prescribed_zero
target_kappa1 = 0''')
if a.production_gauge:
    s=s.replace('telegraph_tau = 1','telegraph_tau = 0.01').replace('telegraph_kappa = 1','telegraph_kappa = 0.01')
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
if a.uniform:
    start=s.index('<refined_region1>');end=s.index('<output2>',start)
    s=s[:start]+s[end:]
    s=s.replace('refinement = static','refinement = none')
seed=root/'seed';seed.mkdir();(seed/'input.athinput').write_text(s)
def run(args,cwd,env=None):
    with (cwd/'run.log').open('w') as log:
        subprocess.run([str(exe),*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
run(['-i','input.athinput'],seed)
checkpoint=sorted((seed/'rst').glob('*.rst'))[-1];before=sha(checkpoint)
if a.seed_only: raise SystemExit(0)
results=[]
fixed_fields={1:[],2:[]}
if a.fixed_duration is not None: assert a.fixed_duration>0
for dt in [a.dt,a.dt/2,a.dt/4]:
    arrays={}
    duration=a.fixed_duration if a.fixed_duration is not None else dt*a.intervals
    explicit_duration=duration if a.fixed_duration is not None or a.intervals>1 else 0
    for ratio in [1,2]:
        case=root/f'dt{dt}_ratio{ratio}';case.mkdir();output=case/'probe'
        env=os.environ.copy();env['ATHENA_TEST_SUBCYCLE_INTERVAL_DIR']=str(output)
        run(['-r',str(checkpoint),f'time/subcycle_probe_dt={dt}',f'time/subcycle_probe_ratio={ratio}',f'time/subcycle_probe_duration={explicit_duration}'],case,env)
        meta=dict(line.split('=',1) for line in (output/'probe.txt').read_text().splitlines())
        # This comparison requires identical endpoints. Retry has a separate
        # regression; never compare a shortened probe against the requested dt.
        assert abs(float(meta['dt'])-duration)<1e-14 and float(meta['requested_dt'])==dt
        assert int(meta['interval_attempts'])==int(meta['accepted_intervals'])
        assert int(meta['accepted_intervals'])>=int(round(duration/dt))
        dtype={8:np.float64,4:np.float32}[int(meta['real_bytes'])]
        values=np.fromfile(output/'fields.bin',dtype=dtype)
        assert values.size==int(meta['leaves'])*int(meta['variables'])*int(meta['ni'])*int(meta['nj'])
        assert np.all(np.isfinite(values))
        arrays[ratio]=values
        fixed_fields[ratio].append(values)
        results.append(dict(dt=dt,ratio=ratio,metadata=meta,fields_sha256=sha(output/'fields.bin')))
    # Independent existing driver: classical RK4 to the identical endpoint.
    baseline=root/f'dt{dt}_driver';baseline.mkdir()
    end=float(meta['end_time'])
    # Match each accepted synchronization endpoint with an independent native
    # restart step. This also exercises consistency across restart initialization.
    intervals=np.loadtxt(root/f'dt{dt}_ratio1'/'probe/intervals.csv',delimiter=',',skiprows=1,ndmin=2)
    restart=checkpoint
    for index,row in enumerate(intervals):
        stepdir=baseline/f'step{index}';stepdir.mkdir()
        run(['-r',str(restart),'time/integrator=rk4_classical',
             'time/nlim=-1',f'time/tlim={row[2]:.17g}'],stepdir)
        restart=sorted((stepdir/'rst').glob('*.rst'))[-1]
    baseline=stepdir
    nv=int(meta['variables']);ni=int(meta['ni']);nj=int(meta['nj']);nb=int(meta['leaves'])
    # This fixture is vacuum Z4c only: restart.cpp writes native u0 as the
    # sole final payload, in source-leaf order with all four ghosts per side.
    # Validate the preceding per-block byte count before interpreting that tail.
    raw=restart.read_bytes();block_bytes=nv*(ni+8)*(nj+8)*8
    offset=len(raw)-nb*block_bytes
    assert int.from_bytes(raw[offset-8:offset],byteorder='little')==block_bytes
    native=np.frombuffer(raw,dtype=np.float64,offset=offset).reshape(nb,nv,nj+8,ni+8)
    leafmap=np.loadtxt(root/f'dt{dt}_ratio2'/'probe/leaves.txt',dtype=int,ndmin=2)
    reference=native[leafmap[:,0],:,4:4+nj,4:4+ni].ravel()
    assert np.all(np.isfinite(reference))
    assert 'Terminating on time limit' in (baseline/'run.log').read_text()
    hst=np.loadtxt(sorted(baseline.glob('*.hst'))[-1],ndmin=2)
    assert abs(hst[-1,0]-end)<1e-14
    for ratio in [1,2]:
        difference=arrays[ratio]-reference
        if ratio==1:
            assert np.max(np.abs(difference))<1e-12, "synchronous hierarchy differs from native driver"
        location=np.unravel_index(np.argmax(np.abs(difference)),(nb,nv,nj,ni))
        results.append(dict(dt=dt,comparison=f'ratio{ratio} minus existing classical driver',
            maximum_location=dict(source_leaf=int(leafmap[location[0],0]),key=leafmap[location[0],1:].tolist(),component=int(location[1]),j=int(location[2]),i=int(location[3])),
            active_values=reference.size,rms=float(np.sqrt(np.mean(difference**2))),
            max_abs=float(np.max(np.abs(difference)))))
    delta=arrays[2]-arrays[1]
    results.append(dict(dt=dt,comparison='ratio2 minus ratio1',rms=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.max(np.abs(delta)))))
if a.fixed_duration is not None:
    for ratio,fields in fixed_fields.items():
        differences=[float(np.sqrt(np.mean((fields[i]-fields[i+1])**2))) for i in [0,1]]
        order_ratio=differences[0]/differences[1]
        results.append(dict(comparison='fixed-time self convergence',ratio=ratio,
            duration=a.fixed_duration,successive_rms=differences,order_ratio=order_ratio))
        assert 10<order_ratio<22, f'fixed-time temporal order failed: ratio={ratio}, reduction={order_ratio}'
assert sha(checkpoint)==before
(root/'results.json').write_text(json.dumps(dict(checkpoint_sha256=before,executable_sha256=sha(exe),scope='local refined gauge-pulse checkpoint; not Brill reproduction or speedup',results=results),indent=2)+'\n')
print((root/'results.json').read_text())
