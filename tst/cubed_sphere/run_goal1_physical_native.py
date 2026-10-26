#!/usr/bin/env python3
"""Native true-face field loader entry, zero trajectories and private build."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];work=root/'build-goal1-minkowski-20261018/physical-interface-v1';out=root/'docs/validation/goal1-minkowski-20261018/physical-interface-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();binary=work/'athena_operator';assert sha(binary)==json.loads((out/'native-build.json').read_text())['binary_sha256']
for physical,n in [(False,8),(True,16)]:
 name=f'B{n}-A{n}-R24-'+('physical' if physical else 'control');case=work/name
 if (case/'receipt.json').exists():print('reuse',name);continue
 case.mkdir(exist_ok=False);shutil.copy2(work/'source/inputs/z4c/cubed_sphere_conformal_probe.athinput',case/'input.athinput')
 cmd=[str(binary),'-i',str(case/'input.athinput'),'-d',str(case)]+[f'{group}/nx{i}={n}' for group in ['mesh','meshblock'] for i in [1,2,3]]
 cmd+=['cubed_sphere/nr=24',f'cubed_sphere/nangular={n}','cubed_sphere/outer_radius=.98','cubed_sphere/radial_stretch=0','cubed_sphere/conformal_amplitude=0','cubed_sphere/conformal_operator_only=true',f'cubed_sphere/conformal_physical_interface={str(physical).lower()}','cubed_sphere/conformal_radial_sbp=true','cubed_sphere/conformal_canonical_ko=true','cubed_sphere/conformal_canonical_full=true','cubed_sphere/conformal_boundary_sat=false','cubed_sphere/conformal_binary_reference=true']
 record={'command':cmd,'binary_sha256':sha(binary),'input_sha256':sha(case/'input.athinput'),'evolution_trajectories':0};(case/'launch.json').write_text(json.dumps(record,indent=2)+'\n');t=time.time()
 with (case/'run.log').open('w')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 record.update(exit_code=r.returncode,elapsed_seconds=time.time()-t,payloads={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in case.iterdir() if p.name.startswith('cubed_sphere_') and p.is_file()})
 if r.returncode==0:
  assert 'Native donor operator diagnostic complete; no evolution' in (case/'run.log').read_text();record['reference_unchanged']=sha(case/'cubed_sphere_reference_before.bin')==sha(case/'cubed_sphere_reference_after.bin');assert record['reference_unchanged']
 (case/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(name,r.returncode,'elapsed',record['elapsed_seconds'],flush=True)
 if r.returncode:raise RuntimeError('Retained failure '+str(case))
