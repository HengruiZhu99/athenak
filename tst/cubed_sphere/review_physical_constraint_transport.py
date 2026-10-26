#!/usr/bin/env python3
import argparse,csv,json,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();E=a.evidence;records=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,count in [('synthetic',48),('extended',192),('massive',60)]:
 rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((E/(name+'.csv')).open())];assert len(rows)==count;fail=[]
 for i,r in enumerate(rows):
  for k in ['physical_Theta_error','physical_Zup_error','physical_C_error']:
   if r[k]>1e-10:fail.append(dict(row=i,quantity=k,value=r[k],gate=1e-10))
 records.append(dict(name=name,probes=count,passed=not fail,failures=fail,maxima={k:max(abs(z[k]) for z in rows) for k in ['physical_Theta_error','physical_Zup_error','physical_C_error','omitted_raising_error']}))
builds=json.loads((E/'builds.json').read_text());assert all(z['returncode']==z['run_returncode']==0 for z in builds)
for z in builds:
 cpp=next(Path(x) for x in z['command'] if x.endswith('.cpp'));z['source_sha256']=sha(cpp);z['binary_sha256']=sha(Path(z['run_command'][0]));shutil.copy2(cpp,E/cpp.name)
(E/'builds.json').write_text(json.dumps(builds,indent=2)+'\n');r=dict(passed=all(z['passed'] for z in records),negative_control_detected=max(z['maxima']['omitted_raising_error'] for z in records)>1e-8,records=records,source_sha256=sha(Path(__file__)),scope='Live physical-index first-order constraint transport/volume identities only. Original double source/time failures remain; no native norm substitution, subsidiary-wave energy/SAT validation or physical successor.');(E/'review.json').write_text(json.dumps(r,indent=2)+'\n');print(r['passed'],r['negative_control_detected'])
