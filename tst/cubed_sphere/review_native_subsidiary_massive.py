#!/usr/bin/env python3
import argparse,csv,json,hashlib,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();E=a.evidence;rows=[{k:float(v) for k,v in z.items()} for z in csv.DictReader((E/'massive.csv').open())];jets=[json.loads(z) for z in (E/'rhs-jets.jsonl').read_text().splitlines()];assert len(rows)==len(jets)==60;fail=[];free=[]
for i,(r,j) in enumerate(zip(rows,jets)):
 assert all(math.isfinite(z) for z in r.values());assert r['mass']==j['mass'] and r['variant']==j['variant'] and r['radius']==j['radius'];assert len(j['state'])==len(j['rhs_jet'])==283
 for k,tol in [('primitive_error',1e-12),('reference_Omega_error',1e-12),('native_jet_error',1e-11),('integrability_error',1e-10),('closed_source_error',1e-9),('closed_H_error',1e-9),('closed_M_error',1e-9),('time_error_eps1',5e-7),('time_error_eps2',5e-7),('time_error_eps3',5e-7),('Theta_transport_error',1e-10),('Z_transport_error',1e-10),('H_functional_error',1e-10),('M_functional_error',1e-10)]:
  if r[k]>tol:fail.append(dict(row=i,quantity=k,value=r[k],gate=tol,mass=r['mass'],radius=r['radius'],direction=r['direction'],variant=r['variant']))
 if r['variant']<2:
  theta=abs(j['state'][192]);Z=max(abs(.5*j['state'][244+13*c]) for c in range(3));v=max(abs(r['H']),r['Mmax'],theta,Z);free.append(dict(row=i,max_constraint=v))
  if v>1e-10:fail.append(dict(row=i,quantity='initial_H_M_Theta_Z',value=v,gate=1e-10))
r=dict(passed=not fail,probes=len(rows),failures=fail,maxima={k:max(abs(z[k]) for z in rows) for k in rows[0]},constraint_satisfying=free,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Outer massive analytic subsidiary closure controls, not evolution, stationary-reference substitution or native boundary admission. Original source-subtraction gate failures retained.');(E/'review.json').write_text(json.dumps(r,indent=2)+'\n');print('pass',r['passed'],'failures',len(fail),[(z['quantity'],z['mass'],z['radius'],z['direction'],z['variant']) for z in fail]);print('initial',max(z['max_constraint'] for z in free))
