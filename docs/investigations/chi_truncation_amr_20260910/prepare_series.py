from pathlib import Path
import sys,json,hashlib,subprocess,shutil
r=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910');c=Path('/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910');sys.path.insert(0,str(c));from controller import setparam,input_parameters
src=c/'cycle_04';prod=r/'production';prod.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exe=prod/'athena';shutil.copy2(r/'build/src/athena',exe);exe.chmod(0o750)
source_sha='4041c73a73eb8a8ea1f2e787623b167b7c2a6ff7' # expanded by caller after validation
manifest=dict(source_sha=source_sha,executable_sha256=sha(exe),reference_amplitude='-0.04896875',reference_nx1=64,reference_tolerance=1e-4,normalization_length=1,bootstrap_until=5,calibration_job='58148291',runs=[])
for n in [128,256,512]:
 d=prod/f'N{n}';d.mkdir();s=(src/'input.athinput').read_text()
 changes=[('mesh','nx1',n//2),('mesh','nx2',n),('meshblock','nx1',n//8),('meshblock','nx2',n//8),('job','basename',f'chiTE_N{n}'),('mesh_refinement','amr_history_file',d/'amr_history.jsonl'),('z4c_amr','method','chi_truncation'),('z4c_amr','chi_error_max',1e-4),('z4c_amr','chi_error_reference_nx1',64),('z4c_amr','chi_error_length',1),('z4c_amr','chi_error_start_time',5),('z4c_amr','chi_error_derefine_factor',.25)]
 for section,key,value in changes:s=setparam(s,section,key,value)
 old=input_parameters((src/'input.athinput').read_text());new=input_parameters(s);allowed={(section,key) for section,key,_ in changes}
 assert all(old.get(k)==v for k,v in new.items() if k not in allowed)
 (d/'input.athinput').write_text(s)
 for f in ['initial.coefficients','amplitude.txt']:(d/f).write_bytes((src/f).read_bytes())
 (d/'initial-data.sha256').write_text(sha(d/'initial.coefficients')+'  initial.coefficients\n')
 script=Path('/pscratch/sd/h/hzhu/n128-failed-amplitude-20260910/run.sh').read_text().replace('"$campaign/athena.history_extrema"',str(exe))
 script=script.replace('sha256sum input.athinput initial.coefficients > inputs.sha256',f'echo "{sha(exe)}  {exe}" | sha256sum -c >> preflight.log\nsha256sum input.athinput initial.coefficients > inputs.sha256')
 (d/'run.sh').write_text(script)
 proof=dict(N=n,input_sha256=sha(d/'input.athinput'),coefficients_sha256=sha(d/'initial.coefficients'),executable_sha256=sha(exe),source_sha=source_sha,effective_threshold=1e-4*(128/n)**4,reference=str(src),changes=[dict(section=x,key=y,value=str(z)) for x,y,z in changes])
 (d/'provenance.json').write_text(json.dumps(proof,indent=2))
 cmd=['salloc','--account=m3328_g','--qos=shared_interactive','--constraint=gpu&hbm80g','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--time=04:00:00',f'--job-name=chiTE-N{n}','bash',str(d/'run.sh'),str(d)]
 (d/'allocation-command.json').write_text(json.dumps(cmd,indent=2))
 with (d/'allocation.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 manifest['runs'].append(dict(N=n,directory=str(d),salloc_pid=p.pid))
 (prod/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
