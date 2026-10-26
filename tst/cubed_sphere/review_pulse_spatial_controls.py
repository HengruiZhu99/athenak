#!/usr/bin/env python3
"""All-region common-time spatial/time/background review of the native angular pulse."""
import argparse,csv,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('coarse',type=Path);p.add_argument('middle',type=Path);p.add_argument('fine',type=Path);p.add_argument('half_step',type=Path);p.add_argument('stationary',type=Path);p.add_argument('output',type=Path);p.add_argument('--time',type=float,default=.1);a=p.parse_args()
def read(root):
 values={};sampling={};times=[]
 for file,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
  rows=list(csv.DictReader((root/file).open()));time=max(float(x['time']) for x in rows);times.append(time)
  for x in rows:
   if float(x['time'])!=time:continue
   key=x['region']+'/'+x['subset'];sampling[key]={k:x[k] for k in ['cells','volume'] if k in x}
   for f in fields:
    v=float(x[f]);assert math.isfinite(v) and v>=0;values[key+'/'+f]=v
 assert len(set(times))==1
 globals=list(csv.DictReader((root/'cubed_sphere_conformal.csv').open()));omega={x['region']:float(x['min_Omega']) for x in globals if float(x['time'])==time}
 return time,values,sampling,omega
runs=[read(x) for x in [a.coarse,a.middle,a.fine,a.half_step]];assert all(x[0]==a.time for x in runs),'Common requested endpoint required';assert all(x[1].keys()==runs[0][1].keys() for x in runs)
checks=[]
for key in runs[0][1]:
 c,m,f,h=[x[1][key] for x in runs];spatial=[abs(c-m),abs(m-f)];temporal=abs(m-h)
 checks.append(dict(region_field=key,values=[c,m,f],decreases=c>m>f,orders=[math.log(c/m)/math.log(1.5) if c*m>0 else None,math.log(m/f)/math.log(4/3) if m*f>0 else None],spatial_differences=spatial,middle_temporal_difference=temporal,temporal_small=temporal<=.1*min(spatial)))
fixed=[x for x in checks if '/radial_' not in x['region_field']]
background=read(a.stationary);assert background[0]==a.time and background[1].keys()==runs[1][1].keys()
bgchecks=[dict(region_field=k,stationary=v,disturbed=runs[1][1][k],negligible=v<=max(1e-10,.01*runs[1][1][k])) for k,v in background[1].items()]
r=dict(background_checks=bgchecks,background_negligible=all(x['negligible'] for x in bgchecks),time=a.time,all_fixed_region_constraints_decrease=all(x['decreases'] for x in fixed),temporal_error_small_for_all_fixed_regions=all(x['temporal_small'] for x in fixed),checks=checks,sampling=[dict(resolution=b,regions=x[2],minimum_Omega=x[3]) for b,x in zip([16,24,32],runs[:3])],scope=f'Angular-pulse diagnostic at common t={a.time}, native constraints with mapped compactified coordinate-volume weights. All active regions retained. Timestep control also bounds different restart milestone truncation. Decreasing errors do not establish long-time stability or settle the original conservative growth trigger. Rb=.98, reference scri1; minimumOmega and near-scri sampling change with resolution.')
a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print({k:v for k,v in r.items() if k not in ['checks','sampling','background_checks']})
