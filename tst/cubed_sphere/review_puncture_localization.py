#!/usr/bin/env python3
"""Predeclared independent bulk/shell resolution contrasts, not unique causation."""
import argparse,csv,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('bulk',type=Path);p.add_argument('shell',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
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
runs=[read(p) for p in [a.baseline,a.bulk,a.shell]];assert all(x[0].keys()==runs[0][0].keys() for x in runs);checks=[]
for k,b in runs[0][0].items():
 bulk,shell=runs[1][0][k],runs[2][0][k];db=(bulk/b-1) if b>1e-10 else None;ds=(shell/b-1) if b>1e-10 else None;guide='coupled_or_inconclusive'
 if db is not None:
  if db<=-.25 and abs(ds)<.1:guide='bulk_resolution_contribution_supported'
  elif ds<=-.25 and abs(db)<.1:guide='shell_resolution_contribution_supported'
  elif db<=-.25 and ds<=-.25:guide='both_resolutions_contribute'
 checks.append(dict(region_field=k,baseline=b,bulk_only=bulk,shell_only=shell,bulk_fractional_change=db,shell_fractional_change=ds,predeclared_localization_guide=guide))
r=dict(time=.1,checks=checks,sampling={n:x[1] for n,x in zip(['baseline','bulk-only','shell-only'],runs)},peak_coordinates={n:x[2] for n,x in zip(['baseline','bulk-only','shell-only'],runs)},scope='Same native equations/closure/reference and target time. Independent resolution changes diagnose sensitivity, not unique boundary/PDE fault. Maxima can shift with sampling; no regions/horizon interior omitted. Full balanced40 common-time review remains separate.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print({v:sum(x['predeclared_localization_guide']==v for x in checks if '/radial_' not in x['region_field']) for v in ['bulk_resolution_contribution_supported','shell_resolution_contribution_supported','both_resolutions_contribute','coupled_or_inconclusive']})
