from pathlib import Path
import json,subprocess,time,hashlib,os
root=Path('/pscratch/sd/h/hzhu/vc-subcycling-20260911')
for row in json.loads((root/'profiles.json').read_text()):
 case=Path(row['case'])
 row['exe_sha256']=hashlib.sha256(Path(row['exe']).read_bytes()).hexdigest()
 row['job_id']=os.environ['SLURM_JOB_ID']
 (case/'manifest.json').write_text(json.dumps(row,indent=2))
 command=['srun','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--gpu-bind=single:1','--cpu-bind=cores','--exact','--kill-on-bad-exit=1',row['exe'],'-r',row['checkpoint'],'-i',str(case/'input.athinput'),'-t','00:25:00']
 start=time.monotonic()
 with (case/'stdout.log').open('w') as out,(case/'stderr.log').open('w') as err:
  result=subprocess.run(command,cwd=case,stdout=out,stderr=err)
 row.update(returncode=result.returncode,wall_seconds=time.monotonic()-start)
 (case/'result.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
 if 'Terminating on cycle limit' not in (case/'stdout.log').read_text():raise SystemExit('Unexpected termination')
(root/'PROFILES_DONE').write_text('complete\n')
