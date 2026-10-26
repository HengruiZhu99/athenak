#!/usr/bin/env python3
"""Build a private native velocity map and read existing checkpoint fixtures."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];E=root/'docs/validation/goal1-minkowski-20261018';out=E/'velocity-momentum-v1';work=root/'build-goal1-minkowski-20261018/velocity-momentum-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=json.loads((E/'native-reduced-source-v1/build.json').read_text())['command'][:];cmd.insert(1,'-I'+str(work));cmd[cmd.index(str(root/'tst/cubed_sphere/goal1_native_reduced_probe.cpp'))]=str(root/'tst/cubed_sphere/goal1_velocity_probe.cpp');cmd[cmd.index('-o')+1]=str(work/'velocity_probe')
t=time.time()
with(out/'build.log').open('w')as f:done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
files=[root/'tst/cubed_sphere'/name for name in ['goal1_velocity_momentum.hpp','goal1_velocity_probe.cpp','goal1_velocity_endpoint.hpp','run_goal1_velocity_momentum.py']]+[work/'velocity_fixture_tools.hpp']
report={'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'source_sha256':{str(p):sha(p)for p in files},'binary_sha256':sha(work/'velocity_probe')if done.returncode==0 else None};(out/'build.json').write_text(json.dumps(report,indent=2)+'\n');assert done.returncode==0
runs=[];rows=[]
for label,name in [('synthetic',None),('native-stationary4','full-stationary4'),('native-pulse8','full-resume8')]:
 path=out/(label+'.jsonl');assert not path.exists();args=[str(work/'velocity_probe')];inputs={}
 if name:
  case=root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission'/(name+'-segment0');ck=case/'probe_checkpoint.bin';ref=case/'cubed_sphere_reference_before.bin';args+=[str(ck),str(ref),label];inputs={'checkpoint_sha256':sha(ck),'reference_sha256':sha(ref)}
 t=time.time()
 with path.open('w')as f,(out/(label+'.stderr')).open('w')as e:done=subprocess.run(args,stdout=f,stderr=e,timeout=90)
 runs.append({'label':label,'command':args,'exit_code':done.returncode,'seconds':time.time()-t,'input_payloads':inputs,'stdout_sha256':sha(path)});(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert done.returncode==0
 part=[json.loads(l)for l in path.read_text().splitlines()if l.startswith('{')];assert len(part)==(120 if name is None else 170);assert len({p['ordinal']for p in part})==len(part);rows+=part
keys=['source_parts_value_scaled','canonical_Pi_plus_F_scaled','actual_native_velocity_scaled','native_value_inverse_scaled','AD_inverse_roundtrip_scaled','AD_tracefree_A_scaled','configuration_change_scaled'];maxima={k:max(r[k]for r in rows)for k in keys};report={'scope':__doc__,'points':len(rows),'maxima':maxima,'all_finite_map_checks_pass':max(maxima.values())<=1e-10 and maxima['configuration_change_scaled']==0,'independent_oracle_points_emitted':sum('input_u'in r for r in rows),'native_trajectories':0,'endpoint_checks_pending':True,'source_sha256':sha(__file__)};(out/'finite-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
