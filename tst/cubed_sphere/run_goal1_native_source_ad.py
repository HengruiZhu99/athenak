#!/usr/bin/env python3
"""Actual nonlinear q source first gradients; read-only reused point data."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/native-q-source-gradient-v1';work=root/'build-goal1-minkowski-20261018/native-q-source-gradient-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();binary=work/'source_probe';assert sha(binary)==json.loads((out/'build.json').read_text())['binary_sha256'];runs=[];rows=[]
for label,name in[('synthetic',None),('native-stationary4','full-stationary4'),('native-pulse8','full-resume8')]:
 path=out/(label+'.jsonl');assert not path.exists();cmd=[str(binary)];inputs={}
 if name:
  case=root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission'/(name+'-segment0');ck=case/'probe_checkpoint.bin';ref=case/'cubed_sphere_reference_before.bin';cmd+=[str(ck),str(ref),label];inputs={'checkpoint_sha256':sha(ck),'reference_sha256':sha(ref)}
 t=time.time()
 with path.open('w')as f,(out/(label+'.stderr')).open('w')as e:done=subprocess.run(cmd,stdout=f,stderr=e,timeout=90)
 runs.append({'label':label,'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'input_payloads':inputs,'stdout_sha256':sha(path)});(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert done.returncode==0,(out/(label+'.stderr')).read_text();part=[json.loads(l)for l in path.read_text().splitlines()if l.startswith('{')];assert len(part)==(120 if name is None else 170);assert len({p['ordinal']for p in part})==len(part);rows+=part
fd=[r for r in rows if'source_gradient_FD_scaled'in r];value=max(r['native_source_value_scaled']for r in rows);defect=max(r['source_gradient_FD_scaled']for r in fd);uncertainty=max(r['FD_refinement_uncertainty_scaled']for r in fd);unused=max(r['unused_higherjet_invariance_absolute']for r in fd)
report={'scope':__doc__,'points':len(rows),'source_gradient_points':len(fd),'native_source_value_scaled_max':value,'source_gradient_FD_scaled_max':defect,'FD_refinement_uncertainty_scaled_max':uncertainty,'unused_higherjet_invariance_absolute_max':unused,'native_source_value_and_first_gradient_checks_pass':bool(value<=1e-10 and max(defect,uncertainty)<=1e-7 and unused==0),'reference_gradient_highprecision_review_sha256':sha(root/'docs/validation/goal1-minkowski-20261018/reference-connection-gradient-v1/review.json'),'AD_map_highprecision_review_sha256':sha(root/'docs/validation/goal1-minkowski-20261018/typed-native-ad-v1/review.json'),'native_reduced_P_source_implemented':False,'native_trajectories_run':0,'exact_scri_implemented':False,'source_sha256':sha(Path(__file__))};(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
