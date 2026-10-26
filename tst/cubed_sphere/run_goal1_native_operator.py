#!/usr/bin/env python3
"""One bounded local actual-donor functional pilot; no evolution or scheduler."""
from pathlib import Path
import argparse,json,hashlib,subprocess,shutil,time,csv,resource
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--bulk',type=int,default=8);p.add_argument('--angular',type=int,default=8);p.add_argument('--radial',type=int,default=24);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
resource.setrlimit(resource.RLIMIT_CORE,(0,0));work=root/'build-goal1-minkowski-20261018/native-donor-operator-v1';out=root/'docs/validation/goal1-minkowski-20261018/native-donor-operator-v1'
name=f'B{a.bulk}-A{a.angular}-R{a.radial}-ungraded'+(f'-attempt{a.attempt}' if a.attempt>1 else '');case=work/name
if (case/'receipt.json').exists():print('reuse',case);raise SystemExit(0)
case.mkdir(exist_ok=False);shutil.copyfile(work/'source/inputs/z4c/cubed_sphere_conformal_probe.athinput',case/'input.athinput')
binary=work/'athena_operator';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert h(binary)==json.loads((out/'build.json').read_text())['binary_sha256']
cmd=[str(binary),'-i',str(case/'input.athinput'),'-d',str(case)]
cmd += [f'{group}/nx{i}={a.bulk}' for group in ['mesh','meshblock'] for i in [1,2,3]]
cmd += [f'cubed_sphere/nr={a.radial}',f'cubed_sphere/nangular={a.angular}','cubed_sphere/outer_radius=0.98','cubed_sphere/radial_stretch=0',
 'cubed_sphere/conformal_amplitude=0','cubed_sphere/conformal_operator_only=true','cubed_sphere/conformal_radial_sbp=true',
 'cubed_sphere/conformal_canonical_ko=true','cubed_sphere/conformal_canonical_full=true','cubed_sphere/conformal_boundary_sat=false',
 'cubed_sphere/conformal_binary_reference=true']
record={'command':cmd,'binary_sha256':h(binary),'input_sha256':h(case/'input.athinput'),'kind':'isolated actual native donor/operator only; zero evolution trajectories','grid':{'bulk':a.bulk,'angular':a.angular,'radial':a.radial},'setup_attempt':a.attempt}
(case/'launch.json').write_text(json.dumps(record,indent=2)+'\n');t=time.time()
with (case/'run.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
record.update(exit_code=r.returncode,elapsed_seconds=time.time()-t,run_log_sha256=h(case/'run.log'))
for name in ['cubed_sphere_operator.csv','cubed_sphere_operator_dependency.json','cubed_sphere_reference_before.bin','cubed_sphere_reference_after.bin']:
 f=case/name
 if f.exists():record.setdefault('payloads',{})[name]={'sha256':h(f),'bytes':f.stat().st_size}
if r.returncode==0:
 assert 'Native donor operator diagnostic complete; no evolution' in (case/'run.log').read_text()
 assert record['payloads']['cubed_sphere_reference_before.bin']['sha256']==record['payloads']['cubed_sphere_reference_after.bin']['sha256']
 record['reference_unchanged']=True;record['operator_rows']=sum(1 for _ in csv.DictReader((case/'cubed_sphere_operator.csv').open()))
 record['dependency']=json.loads((case/'cubed_sphere_operator_dependency.json').read_text())
(case/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print({k:v for k,v in record.items() if k not in ['command','payloads']})
if r.returncode:raise RuntimeError('Native operator pilot failed; retained '+str(case))
