from pathlib import Path
import sys,json,hashlib,subprocess,shutil
root=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910');sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
from controller import setparam
v=root/'validation_symmetry';v.mkdir(exist_ok=False);exe=v/'athena';shutil.copy2(root/'build/src/athena',exe)
for n in [128,512]:
 ref=Path(f'/pscratch/sd/h/hzhu/n{n}-failed-amplitude-20260910');d=v/f'N{n}';d.mkdir();s=(ref/'input.athinput').read_text()
 for block,key,value in [('time','tlim',8 if n==128 else 6),('time','nlim',-1),('mesh_refinement','amr_history_file',d/'amr_history.jsonl'),('z4c_amr','method','chi_truncation'),('z4c_amr','chi_error_max',1e-4),('z4c_amr','chi_error_reference_nx1',64),('z4c_amr','chi_error_length',1),('z4c_amr','chi_error_start_time',5),('z4c_amr','chi_error_derefine_factor',.25),('z4c_amr','chi_error_parent_derefine_factor',.25)]:s=setparam(s,block,key,value)
 (d/'input.athinput').write_text(s)
 for f in ['initial.coefficients','amplitude.txt','initial-data.sha256']:(d/f).write_bytes((ref/f).read_bytes())
 script=(ref/'run.sh').read_text().replace('-t 03:30:00','-t 00:15:00').replace('"$campaign/athena.history_extrema"',str(exe));assert str(exe) in script
 (d/'run.sh').write_text(script)
(v/'job.sh').write_text('#!/bin/bash\nset -euo pipefail\n'+ '\n'.join(f'bash {v}/N{n}/run.sh {v}/N{n}' for n in [128,512])+'\n')
cmd=['salloc','--account=m3328_g','--qos=shared_interactive','--constraint=gpu&hbm80g','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--time=00:35:00','--job-name=chiTE-symmetry','bash',str(v/'job.sh')]
(v/'command.json').write_text(json.dumps(cmd));(v/'executable.sha256').write_text(hashlib.sha256(exe.read_bytes()).hexdigest())
with (v/'allocation.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
print(p.pid)
