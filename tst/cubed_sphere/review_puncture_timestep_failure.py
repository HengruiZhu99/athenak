#!/usr/bin/env python3
"""Apply predeclared restart/timestep failure controls to fixed regional norms."""
import argparse,csv,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('controls',type=Path);p.add_argument('stage01',type=Path);p.add_argument('stage02',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def endpoint(root):
 out={}
 for file,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
  rows=list(csv.DictReader((root/file).open()));t=max(float(x['time']) for x in rows)
  for x in rows:
   if float(x['time'])!=t:continue
   for f in fields:
    v=float(x[f]);assert math.isfinite(v) and v>=0;out[(x['region'],x['subset'],f)]=v
 return t,out
s=json.loads((a.controls/'summary.json').read_text());old=json.loads((a.stage02/'summary.json').read_text());assert s['binary_sha256']==old['binary_sha256']
for path,h in old['source_sha256'].items():assert s['source_sha256'][path]==h
for n,c in s['cases'].items():assert c['exit_code']==0 and c['reached_time_target'] and c['reference_unchanged'] and c.get('initial_profile_unchanged',True)
t0,start=endpoint(a.stage01/'puncture');tb,base=endpoint(a.controls/'fresh-baseline');th,half=endpoint(a.controls/'fresh-half-step');tr,resumed=endpoint(a.stage02/'puncture');assert tb==th==tr==.1 and t0==.02
assert base.keys()==half.keys()==resumed.keys()==start.keys();checks=[]
for kind,other in [('timestep',half),('restart_and_milestone',resumed)]:
 for key,b in base.items():
  error=abs(other[key]-b);relative_limit=.01*max(b,1e-10);growth_limit=.1*max(abs(b-start[key]),1e-10)
  checks.append(dict(control=kind,region=key[0],subset=key[1],field=key[2],baseline=b,other=other[key],absolute_difference=error,within_one_percent=error<=max(relative_limit,1e-10),within_ten_percent_growth=error<=max(growth_limit,1e-10),small=error<=max(relative_limit,growth_limit,1e-10)))
reproduced=base[('shell','all','H_linf')]>1 and base[('shell','all','M_conformal_linf')]>1
result=dict(failure_reproduced=reproduced,all_regional_time_differences_small=all(x['small'] for x in checks if x['control']=='timestep'),all_regional_restart_differences_small=all(x['small'] for x in checks if x['control']=='restart_and_milestone'),checks=checks,scope='Fixed-grid t=.1 native failure diagnosis; neither convergence nor stability acceptance. Every region retained; radial bin sensitivity must be stated.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='checks'})
