from pathlib import Path
import json,subprocess
r=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910/production')
for n in [256,512]:
 d=r/f'N{n}'
 assert not (d/'job-id.txt').exists()
 assert 'QOSMaxSubmitJobPerUserLimit' in (d/'allocation.log').read_text()
 (d/'allocation.log').rename(d/'shared-submission-rejected.log')
script='''#!/bin/bash
set -uo pipefail
ROOT=/pscratch/sd/h/hzhu/chi-truncation-amr-20260910/production
printf '%s\\n' "$SLURM_JOB_ID" > "$ROOT/packed-job-id.txt"
bash "$ROOT/N256/run.sh" "$ROOT/N256" &
p256=$!
bash "$ROOT/N512/run.sh" "$ROOT/N512" &
p512=$!
wait "$p256"; r256=$?
wait "$p512"; r512=$?
printf '{"N256":%s,"N512":%s}\\n' "$r256" "$r512" > "$ROOT/packed-status.json"
if ((r256 != 0 || r512 != 0)); then exit 1; fi
'''
(r/'packed.sh').write_text(script)
cmd=['salloc','--account=m3328_g','--qos=interactive','--constraint=gpu&hbm80g','--nodes=1','--ntasks=2','--cpus-per-task=32','--gpus=4','--time=04:00:00','--job-name=chiTE-N256-N512','bash',str(r/'packed.sh')]
(r/'packed-command.json').write_text(json.dumps(cmd,indent=2))
with (r/'packed-allocation.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
(r/'packed-submission.json').write_text(json.dumps(dict(pid=p.pid,command=cmd),indent=2));print(p.pid)
