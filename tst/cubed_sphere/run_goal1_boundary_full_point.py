#!/usr/bin/env python3
"""Saved full native jets and independent40-matrix check, no evolution."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from compare_closure_field_norms import read
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();out=args.output
protocol=json.loads((out/'point-protocol-v1.json').read_text());build=json.loads((out/'point-build.json').read_text());assert build['exit_code']==0
work=Path('build-goal1-minkowski-20261018/full-boundary-v4');binary=(work/'boundary_point').resolve();h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert h(binary)==build['binary_sha256']
runs=[];allrows=[]
for label,name in [('native-stationary4','full-stationary4'),('native-pulse8','full-resume8'),('synthetic',None)]:
 stdout=out/(label+'.jsonl');stderr=out/(label+'.stderr');assert not stdout.exists(),'Reuse completed point output; changed source needs a new version.'
 command=[str(binary)];payload={}
 if name:
  case=Path('build-goal1-minkowski-20261018/full-canonical-v3/native-admission')/(name+'-segment0');data=read(case);ref=case/'cubed_sphere_reference_before.bin';pin=data['receipt']['reference_payloads'][ref.name];assert h(ref)==pin['sha256']
  command += [str((case/'probe_checkpoint.bin').resolve()),str(ref.resolve()),label];payload={'checkpoint':data['payload'],'reference':pin,'time':data['time'],'cycle':data['cycle']}
 start=time.time()
 with stdout.open('w') as f,stderr.open('w') as e:r=subprocess.run(command,stdout=f,stderr=e)
 runs.append({'label':label,'command':command,'exit_code':r.returncode,'seconds':time.time()-start,'payloads':payload,'stdout_sha256':h(stdout),'stderr_sha256':h(stderr)})
 (out/'point-runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert r.returncode==0
 rows=[json.loads(s) for s in stdout.read_text().splitlines() if s.startswith('{')];assert len(rows)==(384 if name else 18);allrows.extend(rows)

def mm(a,b):return np.einsum('ij,jk->ik',a,b)
groups=[];matrices=[]
for label in ['native-stationary4','native-pulse8','synthetic-reference','synthetic-off-constraint']:
 rows=[r for r in allrows if r['kind']==label];fail=[];port=[];checks=[]
 for r in rows:
  for key,limit in [('normal_row_error',1e-10),('full40_row_error',1e-10),('residual_target_error',1e-10),('corrected_constraint_error',1e-10),('free_condition_error',1e-10),('outgoing_change',1e-10),('metric_derivative_trace_change',1e-10),('A_trace_change',1e-10),('Gamma_time_variation_error',1e-7)]:
   if abs(r[key])>limit:fail.append({'case':r['case'],'quantity':key,'value':r[key],'limit':limit})
  if label=='native-stationary4' and r['incoming_correction']>1e-8:fail.append({'case':r['case'],'quantity':'stationary incoming correction','value':r['incoming_correction']})
  K=np.array(r['K']);F=K[4:];U=np.array(r['full40_rows']);a=np.array(r['spatial_metric']);n=np.array(r['normal']);bn=r['beta'];c=r['c'];v=a.dot(n)/(c*c)
  incoming=np.hstack([np.eye(10)/c,*[v[i]*np.eye(10) for i in range(3)]])/2;sensors=np.vstack([U,mm(F,incoming)]);lift=2*c*np.linalg.solve(K,sensors)
  W=np.zeros((40,40));W[:10,:10]=np.eye(10);W[10:,10:]=np.kron(a,np.eye(10));flux=bn*W
  for i in range(3):
   an=float(a[i].dot(n));flux[:10,10+10*i:20+10*i]+=an*np.eye(10);flux[10+10*i:20+10*i,:10]+=an*np.eye(10)
  selector=np.hstack([np.eye(10),np.zeros((10,30))]);flux-=(c+bn)*(mm(selector.T,lift)+mm(lift.T,selector))
  L=np.linalg.cholesky(W);left=np.linalg.solve(L,flux);Z=np.linalg.solve(L,left.T).T;rate=float(np.linalg.eigvalsh((Z+Z.T)/2)[-1]/max(1,c,abs(bn)));port.append(rate)
  cert=bool(r['port_certified']);matches=bool((cert and rate<=1e-10) or (not cert and rate>=-1e-10));checks.append(matches)
  if not matches or (label.startswith('native-') and (not cert or rate>1e-10)):fail.append({'case':r['case'],'quantity':'independent port certificate','C++':cert,'Python_rate':rate})
  if r['port_flux_sample'] is not None:
   native=np.array(r['port_flux_sample']);defect=float(np.linalg.norm(native-flux)/max(1,np.linalg.norm(flux)));matrices.append({'kind':label,'case':r['case'],'relative_matrix_defect':defect});
   if defect>1e-10:fail.append({'case':r['case'],'quantity':'independent flux matrix','value':defect})
 scalar=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)];maxima={k:max(abs(r[k]) for r in rows) for k in scalar}
 groups.append({'kind':label,'points':len(rows),'passed':not fail,'failures':fail,'maxima':maxima,'maximum_signed_normalized_port':max(port),'all_Cpp_Python_certificate_agree':all(checks),'positive_port_points':sum(v>1e-10 for v in port)})
report={'scope':protocol['scope'],'passed':all(g['passed'] for g in groups),'points':len(allrows),'groups':groups,'independent_flux_matrices':matrices,
 'source_sha256':h(Path(__file__)),'binary_sha256':h(binary),'new_native_trajectories':0,'native_SAT_installed':False,'full_time_compatibility_admitted':False,'exact_scri_admitted':False}
(out/'point-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items() if k not in ['groups','independent_flux_matrices']});print([(g['kind'],g['passed'],g['failures'],g['maximum_signed_normalized_port']) for g in groups])
