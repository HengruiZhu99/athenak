#!/usr/bin/env python3
import argparse,csv,json,hashlib,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();V=a.evidence;O=V/'result';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();integrity=json.loads((O/'independent-integrity.json').read_text());assert integrity['passed'] and integrity['precision']=='64 16';assert integrity['diagnostic_manifest']==json.loads((V/'manifest.json').read_text())
for name,h in integrity['completed_files'].items():
 if (O/name).exists():assert sha(O/name)==h,name
commands=json.loads((O/'commands.json').read_text());assert all(z['returncode']==0 for z in commands);records=[]
for case,count in [('synthetic',192),('massive',60)]:
 rows=[{k:float(v) for k,v in z.items()} for z in csv.DictReader((O/(case+'.csv')).open())];assert len(rows)==count;jets=[json.loads(z) for z in (O/(case+'-rhs-jets.jsonl')).read_text().splitlines()];assert len(jets)==count;fail=[]
 gates={'primitive_error':1e-12,'reference_Omega_error':1e-12,'closed_source_error':1e-9,'closed_H_error':1e-9,'closed_M_error':1e-9,'time_error_eps1':5e-7,'time_error_eps2':5e-7,'time_error_eps3':5e-7,'Theta_transport_error':1e-10,'Z_transport_error':1e-10,'H_functional_error':1e-10,'M_functional_error':1e-10}
 if case=='massive':gates.update(native_jet_error=1e-11,integrability_error=1e-10)
 for i,(r,j) in enumerate(zip(rows,jets)):
  assert all(math.isfinite(v) for v in r.values());assert len(j['state'])==len(j['rhs_jet'])==283
  for k,tol in gates.items():
   if r[k]>tol:fail.append(dict(row=i,quantity=k,value=r[k],gate=tol))
  if case=='massive' and r['variant']<2:
   constraint=max(abs(v) for v in j['H_M']);constraint=max(constraint,abs(j['state'][192]),max(abs(.5*j['state'][244+13*c]) for c in range(3)))
   if constraint>1e-10:fail.append(dict(row=i,quantity='initial_H_M_Theta_Z',value=constraint,gate=1e-10))
 records.append(dict(case=case,probes=count,passed=not fail,failures=fail,maxima={k:max(abs(z[k]) for z in rows) for k in rows[0]},time_eps=[2.5e-6,1.25e-6,6.25e-7] if case=='synthetic' else [1e-5,5e-6,2.5e-6]))
r=dict(passed=all(z['passed'] for z in records),records=records,precision_mantissa_bits=64,precision_bytes=16,integrity_passed=True,original_double_failures_preserved=True,native_boundary_admitted=False,physical_successor_admitted=False,source_sha256=sha(Path(__file__)),scope='Wider-precision analytic equation/oracle validation, not production accuracy, GPU admission or native boundary stability. Original double source/time failures are not rewritten.');(V/'scientific-review.json').write_text(json.dumps(r,indent=2)+'\n');print(r['passed'],[(z['case'],len(z['failures'])) for z in records])
