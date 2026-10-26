#!/usr/bin/env python3
"""Predeclared independent radial/angular resolution contrasts, not unique causation."""
import argparse,csv,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('radial',type=Path);p.add_argument('angular',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def read(root):
 values={};sampling={};peaks={}
 for file,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
  rows=list(csv.DictReader((root/file).open()));time=max(float(x['time']) for x in rows);assert time==.1
  for x in rows:
   if float(x['time'])!=time:continue
   reg=x['region']+'/'+x['subset'];sampling[reg]={k:x[k] for k in ['cells','volume'] if k in x};peaks.setdefault(reg,{}).update({k:v for k,v in x.items() if k.startswith('max_')})
   for f in fields:
    value=float(x[f]);assert math.isfinite(value) and value>=0;values[reg+'/'+f]=value
 return values,sampling,peaks
runs=[read(p) for p in [a.baseline,a.radial,a.angular]];assert all(x[0].keys()==runs[0][0].keys() for x in runs);checks=[]
for k,b in runs[0][0].items():
 bulk,shell=runs[1][0][k],runs[2][0][k];db=(bulk/b-1) if b>1e-10 else None;ds=(shell/b-1) if b>1e-10 else None;guide='coupled_or_inconclusive'
 if db is not None:
  if db<=-.25 and abs(ds)<.1:guide='radial_resolution_contribution_supported'
  elif ds<=-.25 and abs(db)<.1:guide='angular_resolution_contribution_supported'
  elif db<=-.25 and ds<=-.25:guide='both_resolutions_contribute'
 checks.append(dict(region_field=k,baseline=b,radial_only=bulk,angular_only=shell,radial_fractional_change=db,angular_fractional_change=ds,predeclared_localization_guide=guide))
r=dict(time=.1,checks=checks,sampling={n:x[1] for n,x in zip(['baseline','radial-only','angular-only'],runs)},peak_coordinates={n:x[2] for n,x in zip(['baseline','radial-only','angular-only'],runs)},scope='Same native equations/closure/reference and target time, fixed bulk32. Independent radial/angular changes diagnose sensitivity, not unique boundary/PDE fault. Maxima and minimum Omega change with radial sampling; no regions/horizon interior omitted. Worsening with radial refinement is not proof of boundary causation or convergence.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print({v:sum(x['predeclared_localization_guide']==v for x in checks if '/radial_' not in x['region_field']) for v in ['radial_resolution_contribution_supported','angular_resolution_contribution_supported','both_resolutions_contribute','coupled_or_inconclusive']})
