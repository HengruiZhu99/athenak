#!/usr/bin/env python3
"""Native nonlinear momentum and q source checks; reuse all saved trajectories."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/holonomic-momentum-v1';work=root/'build-goal1-minkowski-20261018/holonomic-momentum-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();binary=work/'momentum_probe';assert sha(binary)==json.loads((out/'build.json').read_text())['binary_sha256'];runs=[];rows=[]
for label,name in[('synthetic',None),('native-stationary4','full-stationary4'),('native-pulse8','full-resume8')]:
 path=out/(label+'.jsonl');assert not path.exists(),'Reuse completed output; changed tests need another version.';cmd=[str(binary)];inputs={}
 if name:
  case=root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission'/(name+'-segment0');ck=case/'probe_checkpoint.bin';ref=case/'cubed_sphere_reference_before.bin';cmd+=[str(ck),str(ref),label];inputs={'checkpoint_sha256':sha(ck),'reference_sha256':sha(ref)}
 t=time.time()
 with path.open('w')as f,(out/(label+'.stderr')).open('w')as e:done=subprocess.run(cmd,stdout=f,stderr=e,timeout=90)
 runs.append({'label':label,'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'input_payloads':inputs,'stdout_sha256':sha(path)});(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert done.returncode==0,(out/(label+'.stderr')).read_text();part=[json.loads(line)for line in path.read_text().splitlines()if line.startswith('{')];assert len(part)==(144 if name is None else 170);rows+=part
checks=[r for r in rows if r['kind']!='source-invariance'];groups=[]
for kind in['synthetic','native-stationary4','native-pulse8']:
 rs=[r for r in checks if r['kind']==kind];keys=['native_q_decomposition_scaled','canonical_inverse_scaled','native_momentum_value_roundtrip_scaled','tracefree_A_absolute','configuration_jet_change_absolute','naive_reference_A_trace_defect'];maxima={k:max(abs(r[k])for r in rs)for k in keys};groups.append({'kind':kind,'points':len(rs),'maxima':maxima,'passed':bool(max(maxima[k]for k in keys[:-1])<=1e-10 and maxima['configuration_jet_change_absolute']==0)})
invariance=max(r['gradient_momentum_independence_absolute']for r in rows if r['kind']=='source-invariance');negative=max(r['naive_reference_A_trace_defect']for r in checks)>1e-6
report={'scope':__doc__,'native_nonlinear_q_momentum_map_pass':all(r['passed']for r in groups)and invariance<=1e-10 and negative,'groups':groups,'configuration_source_gradient_momentum_independence_absolute':invariance,'naive_reference_A_tf_negative_control_detected':negative,'rows':len(rows),'native_p_source_transformation_implemented':False,'native_nonlinear_evolution_started':False,'exact_scri_implemented':False,'source_sha256':sha(Path(__file__))};(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
