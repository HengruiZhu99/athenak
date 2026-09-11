#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
out="$root/corrector-probe-fb3d341c"
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1 ATHENA_SUBCYCLE_TRACE=1
trap 'echo $? > "$out/job-exit-status"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
python3 - "$out" <<'PY'
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[1]);m=json.loads((out/'manifest.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(m['executable'])==m['executable_sha256']
assert sha(out/'input.athinput')==m['input_sha256']
assert sha(out/'amr_history.jsonl')==m['prefix_sha256']
assert sha(m['checkpoint'])==m['checkpoint_sha256']
cmd=['srun','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--gpu-bind=single:1','--cpu-bind=cores','--exact','--kill-on-bad-exit=1',m['executable'],'-r',m['checkpoint'],'-i',str(out/'input.athinput'),'-t','00:08:00']
with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
 start=time.monotonic();p=subprocess.run(cmd,cwd=out,stdout=stdout,stderr=stderr)
(out/'result.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,wall_seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
raise SystemExit(p.returncode)
PY
