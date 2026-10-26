#!/usr/bin/env python3
import argparse,csv,json,hashlib,math,ctypes
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();E=a.evidence;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cases=[]
for name,buildname,sourcefile in [('candidate.csv','build.json','check_native_subsidiary_closed_candidate.cpp'),('extended.csv','extended-build.json','check_native_subsidiary_extended.cpp'),('precision.csv','precision-build.json','check_native_subsidiary_precision.cpp')]:
 b=json.loads((E/buildname).read_text());assert b['returncode']==b['run_returncode']==0;rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((E/name).open())];fail=[]
 for i,r in enumerate(rows):
  assert all(math.isfinite(z) for z in r.values())
  for k,tol in [('closed_source_error',1e-9),('closed_H_error',1e-9),('closed_M_error',1e-9),('time_error_eps1',5e-7),('time_error_eps2',5e-7),('time_error_eps3',5e-7),('Theta_transport_error',1e-10),('Z_transport_error',1e-10)]:
   if r[k]>tol:fail.append(dict(row=i,quantity=k,value=r[k],gate=tol,family=r.get('family',0),radius=r['radius'],direction=r['direction'],variant=r['variant'],kappa1=r['kappa1'],kappa2=r.get('kappa2',0)))
 cases.append(dict(name=name,probes=len(rows),passed=not fail,failures=fail,maxima={k:max(abs(z[k]) for z in rows) for k in rows[0]},source_sha256=sha(Path('tst/cubed_sphere')/sourcefile),binary_sha256=sha(Path(b['run_command'][0])),csv_sha256=sha(E/name)))
r=dict(cases=cases,all_original_validation_pass=False,all_closed_H_M_values_pass=all(z['maxima']['closed_H_error']<=1e-9 and z['maxima']['closed_M_error']<=1e-9 for z in cases),long_double_bytes=ctypes.sizeof(ctypes.c_longdouble),double_bytes=ctypes.sizeof(ctypes.c_double),higher_precision_native_control_available=ctypes.sizeof(ctypes.c_longdouble)>ctypes.sizeof(ctypes.c_double),decimal_control=json.loads((E/'minkowski-decimal-review.json').read_text()),scope='Candidate closed equations agree with all H/M oracle values, but extended original validation fails time-difference/source gates. Original failures remain authoritative. No native boundary or physical successor admitted.',source_sha256=sha(Path(__file__)) );(E/'review.json').write_text(json.dumps(r,indent=2)+'\n');print([(z['name'],z['passed'],len(z['failures'])) for z in cases])
