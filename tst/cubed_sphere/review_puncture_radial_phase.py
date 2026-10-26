#!/usr/bin/env python3
"""Common-time radial-sensitivity diagnosis; never a balanced-convergence claim."""
import argparse,csv,json,math,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directories',nargs=4,type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def read(d):
 c=json.loads((d/'receipt.json').read_text());assert c['bulk']==32 and c['nangular']==24 and c['reference_unchanged'] and c['initial_profile_unchanged'] and c['all_rows_finite'];values={};sampling={}
 for file,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
  rows=list(csv.DictReader((d/file).open()));assert max(float(x['time']) for x in rows)==.1
  for x in rows:
   if float(x['time'])!=.1:continue
   k=x['region']+'/'+x['subset'];sampling.setdefault(k,{}).update({s:v for s,v in x.items() if s in ['cells','volume'] or s.startswith('max_')})
   for f in fields:
    v=float(x[f]);assert math.isfinite(v) and v>=0;values[k+'/'+f]=v
 omega={x['region']:float(x['min_Omega']) for x in csv.DictReader((d/'cubed_sphere_conformal.csv').open()) if float(x['time'])==.1}
 return dict(directory=str(d.resolve()),nr=c['nr'],values=values,sampling=sampling,minimum_Omega=omega,receipt_sha256=hashlib.sha256((d/'receipt.json').read_bytes()).hexdigest(),extra_parameters=c['extra_parameters'])
runs=[read(d) for d in a.directories];assert [r['nr'] for r in runs]==[160,164,168,205];assert all(r['values'].keys()==runs[0]['values'].keys() for r in runs)
# Different references are expected on different grids. Physical/gauge parameters remain matched.
ignore=['cubed_sphere/conformal_checkpoint','cubed_sphere/conformal_restart','cubed_sphere/conformal_wall_seconds','cubed_sphere/conformal_checkpoint_tag']
for r in runs:
 for k,v in runs[1]['extra_parameters'].items():
  if k not in ignore:assert r['extra_parameters'].get(k)==v,(k,r['nr'])
checks=[]
for k in runs[0]['values']:
 v=[r['values'][k] for r in runs];checks.append(dict(region_field=k,values=v,relative_changes_from164=[(x-v[1])/v[1] if v[1] else None for x in v]))
collar=[x for x in checks if x['region_field'].startswith('shell/fixed_outer_collar/') and any(x['region_field'].endswith('/'+f) for f in ['H_l2','H_linf','M_conformal_l2','M_conformal_linf'])];bulk=[x for x in checks if x['region_field'].startswith('bulk/fixed_interior/')]
result=dict(time=.1,runs=runs,checks=checks,collar_H_M=collar,nearby_radial_scan_max_relative_collar_change=max(abs(x['relative_changes_from164'][i]) for x in collar for i in [0,2]),nearby_radial_scan_max_relative_bulk_change=max(abs(x['relative_changes_from164'][i]) for x in bulk for i in [0,2]),scope='Fixed bulk/angular resolution, changed radial grid and near-scri sampling. Diagnostic sensitivity only; no balanced convergence, stability, unique boundary/layer/PDE causation or horizon exclusion.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
result['strong_radial_sensitivity_guide']=result['nearby_radial_scan_max_relative_collar_change']>=.25 and result['nearby_radial_scan_max_relative_bulk_change']<.1
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if k not in ['runs','checks','collar_H_M']});print(collar)
