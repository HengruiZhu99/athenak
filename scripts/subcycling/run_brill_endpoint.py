"""Run prepared cases serially inside one single-GPU Slurm allocation."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
m=json.loads(a.manifest.read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert os.environ.get('SLURM_JOB_ID'), 'requires an allocated GPU node'
assert sha(m['exe'])==m['exe_sha256'] and sha(m['checkpoint'])==m['checkpoint_sha256']
for row in m['cases']:
    case=Path(row['directory'])
    assert sha(m['exe'])==m['exe_sha256'], 'executable changed between cases'
    assert sha(case/'input.athinput')==row['input_sha256']
    assert sha(case/'amr_history.jsonl')==row['amr_prefix_sha256']
    command=['srun','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1',
        '--gpu-bind=single:1','--cpu-bind=cores','--exact','--kill-on-bad-exit=1',
        m['exe'],'-r',m['checkpoint'],'-i',str(case/'input.athinput'),'-t','00:25:00']
    # Exclusive files prevent accidentally rerunning into existing outputs.
    with (case/'stdout.log').open('x') as out,(case/'stderr.log').open('x') as err:
        before=time.monotonic();result=subprocess.run(command,cwd=case,stdout=out,stderr=err)
    report=dict(job_id=os.environ['SLURM_JOB_ID'],command=command,
        returncode=result.returncode,wall_seconds=time.monotonic()-before)
    (case/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if result.returncode:raise SystemExit(f'{row["name"]} failed: {result.returncode}')
    assert 'Terminating on time limit' in (case/'stdout.log').read_text(), 'incomplete endpoint'
    if row['ratio']:assert (case/'subcycling_intervals.csv').is_file(), 'live subcycling was not executed'
assert sha(m['checkpoint'])==m['checkpoint_sha256']
print('Execution complete; scientific comparison still required.')
