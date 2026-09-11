"""Native-driver integration on a fixed VC hierarchy; not a live-AMR qualification."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();exe=a.exe.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
fixture=root/'fixture'
subprocess.run([sys.executable,str(Path(__file__).with_name('test_checkpoint_probe.py')),
                str(exe),str(fixture),'--seed-only'],check=True)
checkpoint=sorted((fixture/'seed/rst').glob('*.rst'))[-1]
initial_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
start=np.loadtxt(sorted((fixture/'seed').glob('*.hst'))[-1],ndmin=2)[-1,0]
duration=.008;dt=.002;end=start+duration
results=[]
def run(name,rst,ratio,finish,frozen=False):
    directory=root/name;directory.mkdir();env=os.environ.copy()
    args=['-r',str(rst),'time/integrator=rk4_classical','time/nlim=-1',f'time/tlim={finish:.17g}',
          f'time/subcycle_max_ratio={ratio}',f'time/subcycle_interval_cap={dt}']
    if frozen:
        env['ATHENA_TEST_SUBCYCLE_INTERVAL_DIR']=str(directory/'probe')
        args += [f'time/subcycle_probe_dt={dt}',f'time/subcycle_probe_ratio={ratio}',
                 f'time/subcycle_probe_duration={duration}']
    with (directory/'run.log').open('w') as log:
        subprocess.run([str(exe),*args],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return directory
for ratio in [1,2]:
    frozen=run(f'frozen{ratio}',checkpoint,ratio,end,True)
    meta=dict(line.split('=',1) for line in (frozen/'probe/probe.txt').read_text().splitlines())
    nb,nv,ni,nj=[int(meta[x]) for x in ['leaves','variables','ni','nj']]
    ids=np.loadtxt(frozen/'probe/leaves.txt',dtype=int,ndmin=2)[:,0]
    def fields(rst):
        raw=rst.read_bytes();block_bytes=nv*(ni+8)*(nj+8)*8;offset=len(raw)-nb*block_bytes
        assert int.from_bytes(raw[offset-8:offset],byteorder='little')==block_bytes
        u=np.frombuffer(raw,dtype=np.float64,offset=offset).reshape(nb,nv,nj+8,ni+8)
        return u[ids,:,4:4+nj,4:4+ni].ravel()
    live=run(f'live{ratio}',checkpoint,ratio,end)
    final=sorted((live/'rst').glob('*.rst'))[-1]
    expected=np.fromfile(frozen/'probe/fields.bin',dtype=np.float64)
    actual=fields(final);error=float(np.max(np.abs(actual-expected)))
    assert np.all(np.isfinite(actual)) and error<1e-12,(ratio,error)
    hst=np.loadtxt(sorted(live.glob('*.hst'))[-1],ndmin=2)
    assert abs(hst[-1,0]-end)<1e-14
    intervals=np.loadtxt(live/'subcycling_intervals.csv',delimiter=',',skiprows=1,ndmin=2)
    assert abs(intervals[0,1]-start)<1e-14 and abs(intervals[-1,2]-end)<1e-14
    assert np.all(intervals[1:,1]==intervals[:-1,2])
    levels=np.loadtxt(frozen/'probe/leaves.txt',dtype=int,ndmin=2)[:,1]
    first=max(min(levels),max(levels)-int(np.log2(ratio)))
    expected_steps=sum(2**max(0,int(level)-first) for level in levels)
    assert np.all(intervals[:,7]==expected_steps)
    half=run(f'half{ratio}',checkpoint,ratio,start+duration/2)
    restart=sorted((half/'rst').glob('*.rst'))[-1]
    resumed=run(f'resumed{ratio}',restart,ratio,end)
    continued=fields(sorted((resumed/'rst').glob('*.rst'))[-1])
    restart_error=float(np.max(np.abs(actual-continued)))
    assert restart_error<1e-12,(ratio,restart_error)
    results.append(dict(ratio=ratio,start=float(start),end=float(end),
        live_minus_frozen_max=error,restart_max=restart_error,intervals=len(intervals)))
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==initial_hash
(root/'results.json').write_text(json.dumps(dict(checkpoint_sha256=initial_hash,
    executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),results=results),indent=2)+'\n')
print((root/'results.json').read_text())
