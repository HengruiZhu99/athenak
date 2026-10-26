#!/usr/bin/env python3
"""Compare native pulse endpoint norms at matched physical time, all regions."""
import argparse,csv,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('middle',type=Path);p.add_argument('half_step',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def read(root):
 out={};times=[]
 for f,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
  rows=list(csv.DictReader((root/f).open()));t=max(float(x['time']) for x in rows);times.append(t)
  for x in rows:
   if float(x['time'])==t:
    for k in fields:
     value=float(x[k]);assert math.isfinite(value) and value>=0;out[x['region']+'/'+x['subset']+'/'+k]=value
 assert len(set(times))==1;return t,out
u,x=read(a.middle);v,y=read(a.half_step);assert u==v==.1 and x.keys()==y.keys();checks=[dict(region_field=k,middle=x[k],half_step=y[k],difference=abs(x[k]-y[k]),relative_difference=abs(x[k]-y[k])/max(x[k],1e-10),within_one_percent=abs(x[k]-y[k])<=max(.01*x[k],1e-10)) for k in x]
r=dict(time=.1,all_fixed_regions_within_one_percent=all(z['within_one_percent'] for z in checks if '/radial_' not in z['region_field']),all_regions_within_one_percent=all(z['within_one_percent'] for z in checks),checks=checks,scope='Combined timestep and differing restart milestone truncation control; not proof of spatial accuracy or stability. Three-grid spatial/time comparison still required.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='checks'});print('max relative difference',max(z['relative_difference'] for z in checks))
