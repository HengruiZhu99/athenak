#!/usr/bin/env python3
"""Verify independent differentiation controls; no closed-form subsidiary claim."""
import csv,json,hashlib,math,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();E=a.evidence;r=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((E/'oracle.csv').open())];jets=[json.loads(s) for s in (E/'rhs-jets.jsonl').read_text().splitlines()];assert len(r)==len(jets)==48
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((E/'build.json').read_text());assert b['returncode']==b['run_returncode']==0;assert b['source_sha256']==sha(Path('tst/cubed_sphere/check_native_hm_time_oracle.cpp'));assert b['binary_sha256']==sha(Path(b['run_command'][0]));fail=[]
for row,j in zip(r,jets):
 assert row['radius']==j['radius'] and row['direction']==j['direction'] and row['variant']==j['variant'] and row['kappa1']==j['kappa1'];assert len(j['state'])==len(j['rhs_jet']);assert all(math.isfinite(v) for v in list(row.values())+j['state']+j['rhs_jet']+j['H_M']+j['H_M_dot']);assert abs(row['Hdot']-j['H_M_dot'][0])<1e-12*max(1,abs(row['Hdot']))
 for k in ['primitive_error','reference_Omega_error']:
  if row[k]>1e-12:fail.append((row['radius'],row['direction'],row['variant'],row['kappa1'],k,row[k]))
 for k in ['time_error_eps1','time_error_eps2','time_error_eps3']:
  if row[k]>5e-7:fail.append((row['radius'],row['direction'],row['variant'],row['kappa1'],k,row[k]))
 for k in ['Theta_transport_error','Z_transport_error','H_functional_error','M_functional_error']:
  if row[k]>1e-10:fail.append((row['radius'],row['direction'],row['variant'],row['kappa1'],k,row[k]))
maxima={k:max(abs(z[k]) for z in r) for k in r[0]};result=dict(passed=not fail and maxima['H']>1e-6 and maxima['Mmax']>1e-6 and maxima['Hdot']>1e-6 and maxima['Mdotmax']>1e-6,failures=fail,probes=len(r),jet_entries=len(jets[0]['state']),maxima=maxima,negative_control_detected=maxima['omitted_spatial_jet_error']>1e-3,source_sha256=sha(Path(__file__)),inputs={str(E/name):sha(E/name) for name in ['criteria.json','functional-criteria.json','oracle.csv','rhs-jets.jsonl','build.json']},scope='Native RHS spatial AD and H/M time-variation oracle only. Same native constraint functional is differentiated two ways; this is not independent proof of a closed subsidiary formula, native boundary correctness or physical stability. Synthetic probes are off constraint and not production evolution.')
(E/'review.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['passed'],'probes',len(r),'jetentries',result['jet_entries']);print({k:maxima[k] for k in ['time_error_eps1','time_error_eps2','time_error_eps3','Theta_transport_error','Z_transport_error','omitted_spatial_jet_error']})
