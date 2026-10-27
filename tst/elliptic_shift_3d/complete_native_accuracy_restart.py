#!/usr/bin/env python3
"""Complete the dt-half restart control from its actual output-cadence checkpoint."""
import argparse,hashlib,json,os,resource,signal,subprocess,time
from pathlib import Path
import numpy as np
from analyze_native_wormhole import sidecar
p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--capsule',type=Path,required=True);a=p.parse_args();root=Path.cwd();controls=a.controls.resolve();case=controls/'dt_half';capsule=a.capsule.resolve();binary=capsule/'athena-qualified';checkpoint=case/'full/rst/native_wormhole.00001.rst';out=case/'restart-corrected';assert checkpoint.exists() and not out.exists();mid=sidecar(Path(str(checkpoint)+'.composite'));assert mid['completed_cycles']==2 and mid['time']==.0001;resource.setrlimit(resource.RLIMIT_CORE,(0,0));assert not(case/'initial.dat').exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();command=[str(binary),'-r',str(checkpoint),'-d',str(out)];m={'scope':'Corrected dt-half restart from cycle2 at actual dt-output cadence. Original missing-cycle3 failure is retained.','command':command,'binary_sha256':sha(binary),'checkpoint_sha256':sha(checkpoint),'maximum_seconds':180};start=time.monotonic()
with (case/'restart-corrected.stdout').open('w') as so,(case/'restart-corrected.stderr').open('w') as se:
 process=subprocess.Popen(command,cwd=root,stdout=so,stderr=se,start_new_session=True);m['owned_process_group']=process.pid
 try:m['exit_code']=process.wait(timeout=180)
 except subprocess.TimeoutExpired:
  assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
  m['exit_code']=124;m['owned_process_reaped']=True
m['seconds']=time.monotonic()-start;(controls/'restart-followup.json').write_text(json.dumps(m,indent=2)+'\n');assert m['exit_code']==0
m['snapshots']={}
for suffix in ['.csv','.scri.csv']:
 f=case/('full/native_wormhole.00004'+suffix);g=out/f.name;m['snapshots'][suffix]={'bitwise_identical':f.read_bytes()==g.read_bytes(),'full_sha256':sha(f),'restart_sha256':sha(g)}
f=sidecar(case/'full/rst/native_wormhole.00002.rst.composite');g=sidecar(out/'rst/native_wormhole.00002.rst.composite');m['groups']={group:{field:np.array_equal(f[group][field],g[group][field]) for field in f[group]} for group in ['values','registers','rk']};m['vectors']={field:np.array_equal(f[field],g[field]) for field in ['radii','chart_parameters','shift']};m['topology_equal']=f['topology']==g['topology'];m['passed']=all(r['bitwise_identical'] for r in m['snapshots'].values()) and all(all(r.values()) for r in m['groups'].values()) and all(m['vectors'].values()) and m['topology_equal'];assert m['passed'];m['binary_unchanged']=sha(binary)==m['binary_sha256'];m['checkpoint_unchanged']=sha(checkpoint)==m['checkpoint_sha256'];assert m['binary_unchanged'] and m['checkpoint_unchanged'];(controls/'restart-followup.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'passed':m['passed'],'seconds':m['seconds']}))
