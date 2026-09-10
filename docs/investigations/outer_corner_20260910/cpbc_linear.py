from pathlib import Path
import json, subprocess, sys, time, re
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-t200-20260909')
from controller import setparam,sha
R=Path(__file__).resolve().parent
base=Path('/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901/bisection_N256_20260907')
prod=Path('/pscratch/sd/h/hzhu/lapse-bisection-t200-20260909/endpoint_sub')
rst=prod/'rst/lapse200.00090.rst'
(R/'provenance.json').write_text(json.dumps(dict(checkpoint=str(rst),checkpoint_sha256=sha(rst),executable_sha256=sha(base/'athena.history_extrema')),indent=2))
with rst.open('rb') as f: header=f.read(200000).split(b'<par_end>')[0].decode()
nh=int(re.search(r'history_bytes\s*=\s*(\d+)',header)[1])
with (prod/'amr_history.jsonl').open('rb') as f: history=f.read(nh)
variants=[('bjorhus_extrap2',[('z4c','boundary_rhs','full_constraint_bjorhus'),('z4c','extrap_order',2),('z4c','vc_single_rank_device_sync','true'),('output5','dt',1)])]
for name,changes in variants:
 d=R/name;d.mkdir(exist_ok=True)
 (d/'amr_history.jsonl').write_bytes(history)
 inp=(prod/'input.athinput').read_text()
 for section,key,value in [('job','basename','bench'),('time','tlim',110),('mesh_refinement','amr_history_file',d/'amr_history.jsonl'),('problem','brill_global_coefficients_file',prod/'initial.coefficients'),('problem','constraint_summary_file',d/'initial-constraints.dat')]+changes:
  inp=setparam(inp,section,key,value)
 (d/'input.athinput').write_text(inp)
 cmd=['srun','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--gpu-bind=single:1','--cpu-bind=cores','--exact',str(base/'athena.history_extrema'),'-r',str(rst),'-i',str(d/'input.athinput'),'-t','00:15:00']
 start=time.time()
 with (d/'stdout.log').open('w') as out,(d/'stderr.log').open('w') as err: result=subprocess.run(cmd,cwd=d,stdout=out,stderr=err)
 (d/'status.json').write_text(json.dumps(dict(rc=result.returncode,wall_seconds=time.time()-start,command=cmd),indent=2))
 if result.returncode: raise RuntimeError(name+' failed; inspect stderr')
 print(name,'finished',time.time()-start,flush=True)
