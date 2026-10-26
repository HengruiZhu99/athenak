#!/usr/bin/env python3
"""Read-only chart fixtures with pinned native checkpoints; no trajectories."""
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/holonomic-chart-v1';work=root/'build-goal1-minkowski-20261018/holonomic-chart-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();build=json.loads((out/'build.json').read_text());binary=work/'chart_probe';assert sha(binary)==build['binary_sha256'];assert not(out/'fixtures.jsonl').exists()
rng=np.random.default_rng(20261018);lines=[];cases=[]
for i in range(25):
 q=np.zeros((10,13));q[:,0]=rng.normal(0,.12,10);q[0,0]=1+rng.uniform(-.2,.2);q[6,0]=1+rng.uniform(-.2,.2);q[:,1:4]=rng.normal(0,.15,(10,3));h=rng.normal(0,.12,(10,3,3));h=.5*(h+h.swapaxes(1,2));q[:,4:]=h.reshape(10,9)
 if i==24:q[1:6,0]=[1.5,-1.,.3,-.2,.25]
 lines.append('config '+str(i)+' '+' '.join(format(v,'.17g')for v in q.ravel()));cases.append({'kind':'config','case':i,'admit':True})
index=0
for radius in[0,.05-1e-10,.05,.05+1e-10,.5,.94,.95-1e-10,.95,.95+1e-10,.98,1-1e-8,1-1e-12,1.]:
 for n in[[1,0,0],[.8,-.48,.36],[1/np.sqrt(3)]*3]:
  if radius==1 and n!=[1,0,0]:continue
  x=np.array(n)*radius;lines.append('reference '+str(index)+' '+' '.join(format(v,'.17g')for v in x));cases.append({'kind':'reference','case':index,'admit':True,'x':x.tolist()});index+=1
for kind in range(5):lines.append(f'invalid {kind} {kind}');cases.append({'kind':'invalid-value','case':kind,'admit':False})
for index in[100,101]:
 m=np.zeros((6,13));m[[0,3,5],0]=1
 if index==100:m[5,1]=.2
 else:m[5,8]=.4
 lines.append('metric '+str(index)+' '+' '.join(format(v,'.17g')for v in m.ravel()));cases.append({'kind':'metric-input','case':index,'admit':True,'off_differential_constraint':True})
(out/'fixtures-input.txt').write_text('\n'.join(lines)+'\n');(out/'fixture-protocol-v1.json').write_text(json.dumps({'fixed_seed':20261018,'cases':cases,'expected_native_points_per_checkpoint':170,'oracle_precision_digits':80,'native_oracle_sample':'All synthetic/reference rows plus every23rd saved native row; all native rows get roundtrip and differential/time defect review.','source_sha256':sha(Path(__file__))},indent=2)+'\n')
runs=[]
def run(label,cmd,input=None,payloads=None):
 stdout=out/(label+'.jsonl');stderr=out/(label+'.stderr');assert not stdout.exists();t=time.time()
 with stdout.open('w')as f,stderr.open('w')as e:done=subprocess.run(cmd,input=input,text=True,stdout=f,stderr=e,timeout=90)
 record={'label':label,'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'input_payloads':payloads or {},'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)};runs.append(record);(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert done.returncode==0,stderr.read_text();return [json.loads(line)for line in stdout.read_text().splitlines()if line.startswith('{')]
rows=run('fixtures',[str(binary)],(out/'fixtures-input.txt').read_text(),{'stdin_sha256':sha(out/'fixtures-input.txt')});assert len(rows)==len(cases)
for label,name in[('native-stationary4','full-stationary4'),('native-pulse8','full-resume8')]:
 case=root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission'/(name+'-segment0');ck=case/'probe_checkpoint.bin';ref=case/'cubed_sphere_reference_before.bin';rows=run(label,[str(binary),str(ck),str(ref),label],payloads={'checkpoint_sha256':sha(ck),'reference_sha256':sha(ref)});assert len(rows)==170
print('chart fixtures complete; zero evolution')
