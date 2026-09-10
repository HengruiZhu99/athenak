from pathlib import Path
import sys,json,shutil,re,shlex
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'bisection'))
from controller import setparam,sha
from resume_case import checkpoint_info
v=root/'recovery-lapse/validation';source=v/'recovered';d=v/'persisted';d.mkdir()
rst=sorted((source/'rst').glob('*.rst'))[-1];info=checkpoint_info(rst)
shutil.copyfile(source/'initial.coefficients',d/'initial.coefficients')
with (source/'amr_history.jsonl').open('rb') as f:(d/'amr_history.jsonl').write_bytes(f.read(info['history_bytes']))
inp=re.sub(r'^\s*termination_min_lapse\s*=.*\n','',(source/'input.athinput').read_text(),flags=re.M)
inp=setparam(inp,'time','nlim',info['cycle']+8);inp=setparam(inp,'mesh_refinement','amr_history_file',d/'amr_history.jsonl')
(d/'input.athinput').write_text(inp)
(d/'provenance.json').write_text(json.dumps(dict(checkpoint=str(rst),checkpoint_sha256=sha(rst),checkpoint_info=info,input_sha256=sha(d/'input.athinput')),indent=2))
exe=root/'recovery-lapse/build/src/athena'
(d/'run.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(d))+'''\ntrap 'rc=$?; echo "$rc" > run-status' EXIT
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 '''+str(exe)+' -r '+str(rst)+' -i input.athinput > stdout.log 2> stderr.log\n')
