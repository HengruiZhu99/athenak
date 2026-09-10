from pathlib import Path
import sys,shutil,json,shlex
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'bisection'))
from controller import setparam,sha
from criterion import read_history
from resume_case import checkpoint_info
oldcampaign=Path('/pscratch/sd/h/hzhu/lapse-bisection-t200-corner-fixed-20260910')
mid=oldcampaign/'cycle_01';sup=root/'fresh_super_cpbc_linear'
new=root/'recovery-lapse/build/src/athena';assert new.is_file()
v=root/'recovery-lapse/validation';v.mkdir()
hist=read_history(next(mid.glob('*.hst')));prior=min(r['minLapse'] for r in hist if r['time']<=74)
manifest=[]
for label,source,num,dipstate in [('no_dip',mid,74,None),('recovered',mid,74,prior),('above',sup,50,None),('collapse',sup,98,None)]:
 d=v/label;d.mkdir();basename='lapse200' if source==mid else 'boundary200'
 rst=source/'rst'/f'{basename}.{num:05d}.rst';info=checkpoint_info(rst)
 shutil.copyfile(source/'initial.coefficients',d/'initial.coefficients')
 with (source/'amr_history.jsonl').open('rb') as f:(d/'amr_history.jsonl').write_bytes(f.read(info['history_bytes']))
 inp=(source/'input.athinput').read_text()
 for sec,key,value in [('job','basename','verify'),('time','nlim',info['cycle']+8),('mesh_refinement','amr_history_file',d/'amr_history.jsonl'),('problem','collapse_lapse_threshold',1e-5),('problem','dispersion_lapse_dip',.1),('problem','dispersion_lapse_recovery',.8)]:inp=setparam(inp,sec,key,value)
 if dipstate is not None:inp=setparam(inp,'problem','termination_min_lapse',dipstate)
 (d/'input.athinput').write_text(inp)
 manifest.append(dict(label=label,checkpoint=str(rst),checkpoint_sha256=sha(rst),checkpoint_info=info,executable_sha256=sha(new),input_sha256=sha(d/'input.athinput'),seeded_minimum=dipstate,history_evidence_sha256=sha(next(source.glob('*.hst'))),original_directory=str(source)))
 script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(d))+'''\ntrap 'rc=$?; echo "$rc" > run-status' EXIT
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 '''+shlex.quote(str(new))+' -r '+shlex.quote(str(rst))+' -i input.athinput > stdout.log 2> stderr.log\n'
 (d/'run.sh').write_text(script)
(v/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(v/'run.sh').write_text('#!/bin/bash\nset -euo pipefail\nmodule load PrgEnv-gnu cudatoolkit cmake cray-hdf5\nprintf "%s\\n" "$SLURM_JOB_ID" > '+str(v/'job-id.txt')+'\n'+''.join('bash '+shlex.quote(str(v/x['label']/'run.sh'))+'\n' for x in manifest))
with (v/'run.sh').open('a') as f:
 f.write('/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3 '+str(root/'prepare_recovery_persistence.py')+'\n')
 f.write('bash '+str(v/'persisted/run.sh')+'\n')
print(v)
