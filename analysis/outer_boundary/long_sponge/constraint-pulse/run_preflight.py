"""Authorized bounded local MPI preflight only; contains no queue submission."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time

root=Path(__file__).resolve().parent
exe=Path('/Users/hz0693/research/TDE/outer-boundary-fix-20260920/bin/athena-active-stencil-mpi')
expected='67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8'
checker=root.parent/'gpu/check_minkowski_checkpoint.py'
regression='/Users/hz0693/research/TDE/athenak-outer-boundary-fix/tst/regression'
p=argparse.ArgumentParser();p.add_argument('phase',choices=['gates','early','ablation','coremask','coremask_zero','ablation_coremask_zero']);a=p.parse_args()
assert hashlib.sha256(exe.read_bytes()).hexdigest()==expected
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',KOKKOS_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ATHENA_REGRESSION_PATH=regression)
manifest=json.loads((root/'input-manifest.json').read_text())
cases={'gates':['zero_standard','pulse_gate_standard','zero_loweta','pulse_gate_loweta'],
       'early':['theta_loweta_early_dt3p2','theta_loweta_early_dt1p6'],
       'ablation':['zero_loweta_lapse01','pulse_gate_loweta_lapse01'],
       'coremask':['pulse_gate_loweta_coremask'],
       'coremask_zero':['zero_loweta_coremask'],
       'ablation_coremask_zero':['zero_loweta_lapse01_coremask']}[a.phase]
if a.phase=='early':
    for name in ['zero_standard','pulse_gate_standard','zero_loweta','pulse_gate_loweta']:
        assert json.loads((root/'runs'/name/'checkpoint-validation.json').read_text())['passed']
for name in cases:
    work=root/'runs'/name
    if work.exists():raise RuntimeError(f'Preserving existing run {work}')
    work.mkdir(parents=True)
    content=(root/(name+'.athinput')).read_bytes()
    assert hashlib.sha256(content).hexdigest()==manifest['cases'][name]['input_sha256']
    (work/'input.athinput').write_bytes(content)
    cap='00:05:00' if a.phase=='early' else '00:02:00'
    cmd=['/opt/homebrew/bin/mpiexec','-n','8',str(exe),'-i',str(work/'input.athinput'),'-d',str(work),'-t',cap]
    meta=dict(executable=str(exe),sha256=expected,command=cmd,start=datetime.datetime.now(datetime.timezone.utc).isoformat(),ranks=8,threads_per_rank=1)
    (work/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    tick=time.monotonic()
    with (work/'run.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    (work/'exit_code.txt').write_text(str(r.returncode)+'\n')
    meta.update(exit=r.returncode,wall_seconds=time.monotonic()-tick,end=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (work/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    args=[sys.executable,str(checker),str(work),'--ranks','8']
    if name.startswith('zero'):args+=['--exact-zero']
    with (work/'checker.log').open('w') as log:c=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,application_exit=r.returncode,checker_exit=c.returncode,seconds=meta['wall_seconds'])),flush=True)
    assert r.returncode==0 and c.returncode==0,work
