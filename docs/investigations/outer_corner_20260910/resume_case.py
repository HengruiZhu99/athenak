#!/usr/bin/env python3
"""Resume an authenticated clean wall-clock stop; never classify an incomplete case."""
from pathlib import Path
import argparse,fcntl,json,math,os,re,shlex,shutil,struct,subprocess,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lapse-bisection-t200'))
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-t200-20260909')
sys.path.insert(0,str(Path(__file__).resolve().parent/'bisection'))
from criterion import read_history

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def checkpoint_info(path):
    # Archived executable uses little-endian doubles, 9 Real RegionSize,
    # 19 int RegionIndcs; see outputs/restart.cpp and mesh/mesh.hpp.
    data=path.open('rb')
    with data:prefix=data.read(1024*1024)
    marker=b'<par_end>\n';offset=prefix.find(marker)
    if offset<0:raise ValueError('Restart header missing')
    header=prefix[:offset].decode();payload=prefix[offset+len(marker):]
    nmb,root=struct.unpack_from('<ii',payload,0)
    dims=struct.unpack_from('<19i',payload,80)
    t,dt,cycle=struct.unpack_from('<ddi',payload,232)
    if nmb<=0 or root!=3 or dims[1:4]!=(128,256,1):raise ValueError('Unexpected N256 restart layout')
    if not math.isfinite(t) or not math.isfinite(dt) or dt<=0 or cycle<0:raise ValueError('Invalid checkpoint state')
    nh=re.search(r'^history_bytes\s*=\s*(\d+)',header,re.M)
    if not nh:raise ValueError('AMR history byte count missing')
    return dict(time=t,dt=dt,cycle=cycle,nmb=nmb,history_bytes=int(nh[1]))

def validate_checkpoint(case):
    if (case/'run-status').read_text().strip()!='0':raise RuntimeError('Previous evolution failed')
    log=(case/'stdout.log').read_text()
    terminations=re.findall(r'Terminating on [^\n]+',log)
    if not terminations or terminations[-1]!='Terminating on wall clock limit':raise RuntimeError('Not a clean wall-clock stop')
    if list(case.glob('*.termination.json')):raise RuntimeError('Physical/resource stopping marker exists')
    hist=read_history(next(case.glob('*.hst')))
    if not 0<hist[-1]['time']<200:raise RuntimeError('Case does not need continuation to200')
    rst=sorted((case/'rst').glob('*.rst'))[-1];info=checkpoint_info(rst)
    if abs(info['time']-hist[-1]['time'])>1e-9 or info['cycle']!=int(hist[-1]['cycle']):raise RuntimeError('Final history/checkpoint mismatch')
    if (case/'amr_history.jsonl').stat().st_size!=info['history_bytes']:raise RuntimeError('AMR history length mismatch; do not truncate silently')
    checks={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split()[0] for line in (case/'inputs.sha256').read_text().splitlines()}
    for name in ['input.athinput','initial.coefficients']:
        if checks.get(name)!=sha(case/name):raise RuntimeError('Input changed: '+name)
    return rst,info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);ap.add_argument('--previous-job',required=True);a=ap.parse_args()
    case=a.case.resolve(strict=True)
    with (case/'continuation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (case/'job-id.txt').read_text().strip()!=a.previous_job:raise RuntimeError('Previous job ID mismatch')
        live=subprocess.check_output(['squeue','-h','-j',a.previous_job,'-o','%i'],text=True).strip()
        if live:raise RuntimeError('Previous allocation is still active; do not duplicate')
        accounting=subprocess.check_output(['sacct','-j',a.previous_job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
        rows=[x.split('|') for x in accounting.splitlines() if x.split('|')[0]==a.previous_job]
        if len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:raise RuntimeError('Previous allocation is not verified COMPLETED0:0')
        rst,info=validate_checkpoint(case)
        provenance=json.loads((case/'provenance.json').read_text());base=Path(provenance['baseline']);exe=base/'athena.history_extrema'
        if sha(exe)!=provenance['exe_sha256']:raise RuntimeError('Executable hash changed')
        segment=case/'segments'/('after-'+a.previous_job);segment.mkdir(parents=True,exist_ok=False)
        for name in ['stdout.log','stderr.log','run-status','job-id.txt','slurm-job.txt','started.txt','finished.txt']:
            if (case/name).exists():shutil.copyfile(case/name,segment/name)
        manifest=dict(previous_job=a.previous_job,checkpoint=str(rst),checkpoint_sha256=sha(rst),checkpoint_info=info,executable_sha256=sha(exe),prepared=time.time())
        (segment/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        script=segment/'run.sh'
        script.write_text('''#!/usr/bin/env bash
set -euo pipefail
cd CASE
trap 'rc=$?; printf "%s\\n" "$rc" > run-status; date -Is > finished.txt' EXIT
printf '%s\\n' "$SLURM_JOB_ID" > job-id.txt
date -Is > started.txt
scontrol show job "$SLURM_JOB_ID" > slurm-job.txt
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 EXE -r RESTART -i INPUT -t 03:30:00 >> stdout.log 2>> stderr.log
'''.replace('CASE',shlex.quote(str(case))).replace('EXE',shlex.quote(str(exe))).replace('RESTART',shlex.quote(str(rst))).replace('INPUT',shlex.quote(str(case/'input.athinput'))))
        cmd=['salloc','--account=m3328_g','--qos=shared_interactive','--constraint=gpu&hbm80g','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--time=04:00:00','--job-name=corner-continue','bash',str(script)]
        (segment/'allocation-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        (case/'run-status').unlink()
        with (segment/'allocation.log').open('w') as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
        (segment/'allocation-status.json').write_text(json.dumps({'rc':rc,'finished':time.time()})+'\n')
        if rc or not (case/'run-status').exists() or (case/'run-status').read_text().strip()!='0':raise RuntimeError('Continuation failed; no further submission')
        print('Continuation ended; inspect final time before classification or another segment.',flush=True)
if __name__=='__main__':main()
