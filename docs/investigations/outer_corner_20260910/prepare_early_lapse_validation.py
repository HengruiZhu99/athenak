from pathlib import Path
import sys,shutil,json,shlex
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'bisection'))
from controller import setparam,sha
from resume_case import checkpoint_info
case=root/'fresh_super_cpbc_linear'
base=Path('/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901/bisection_N256_20260907')
new=root/'early-lapse/build/src/athena'
assert new.is_file()
validation=root/'early-lapse/validation';validation.mkdir()
manifest=[]
for label,number,steps,exe,threshold in [('above_old',50,8,base/'athena.history_extrema',0),('above_new',50,8,new,1e-5),('below_new',98,8,new,1e-5)]:
 d=validation/label;d.mkdir()
 rst=case/'rst'/f'boundary200.{number:05d}.rst';info=checkpoint_info(rst)
 shutil.copyfile(case/'initial.coefficients',d/'initial.coefficients')
 with (case/'amr_history.jsonl').open('rb') as f:(d/'amr_history.jsonl').write_bytes(f.read(info['history_bytes']))
 inp=(case/'input.athinput').read_text()
 for sec,key,value in [('job','basename','verify'),('time','nlim',info['cycle']+steps),('mesh_refinement','amr_history_file',d/'amr_history.jsonl'),('problem','collapse_lapse_threshold',threshold)]:inp=setparam(inp,sec,key,value)
 (d/'input.athinput').write_text(inp)
 entry=dict(label=label,checkpoint=str(rst),checkpoint_sha256=sha(rst),checkpoint_info=info,executable=str(exe),executable_sha256=sha(exe),input_sha256=sha(d/'input.athinput'))
 manifest.append(entry)
 script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(d))+'''\ntrap 'rc=$?; echo "$rc" > run-status' EXIT
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 '''+shlex.quote(str(exe))+' -r '+shlex.quote(str(rst))+' -i input.athinput > stdout.log 2> stderr.log\n'
 (d/'run.sh').write_text(script)
(validation/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(validation/'run.sh').write_text('#!/bin/bash\nset -euo pipefail\nmodule load PrgEnv-gnu cudatoolkit cmake cray-hdf5\nprintf "%s\\n" "$SLURM_JOB_ID" > '+str(validation/'job-id.txt')+'\n'+''.join('bash '+shlex.quote(str(validation/x['label']/'run.sh'))+'\n' for x in manifest))
print(validation)
