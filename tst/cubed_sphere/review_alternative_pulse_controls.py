#!/usr/bin/env python3
"""Alternative closure own time/background controls at a common endpoint."""
import argparse,csv,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args();m=json.loads(a.manifest.read_text());target=float(m['time'])
def read(root):
 root=Path(root);c=json.loads((root/'receipt.json').read_text());assert c['all_rows_finite'] and c['reference_unchanged'] and c.get('initial_profile_unchanged',True);assert float(c['extra_parameters']['cubed_sphere/conformal_mass'])==0.;v={};sampling={};times=[]
 for file,fields in [('cubed_sphere_conformal_regions.csv',['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),('cubed_sphere_conformal_maxima.csv',['M_conformal_linf','Z_conformal_linf'])]:
  rows=list(csv.DictReader((root/file).open()));t=max(float(x['time']) for x in rows);times.append(t)
  for x in rows:
   if float(x['time'])!=t:continue
   k=x['region']+'/'+x['subset'];sampling.setdefault(k,{}).update({s:y for s,y in x.items() if s in ['cells','volume'] or s.startswith('max_')})
   for f in fields:
    z=float(x[f]);assert math.isfinite(z) and z>=0;v[k+'/'+f]=z
 assert len(set(times))==1
 state=[x for x in csv.DictReader((root/'cubed_sphere_conformal.csv').open()) if float(x['time'])==times[0]]
 return dict(time=times[0],values=v,sampling=sampling,minimum_Omega={x['region']:float(x['min_Omega']) for x in state},grid=[c['bulk'],c['nangular'],c['nr']],directory=str(root.resolve()),extra_parameters=c['extra_parameters'],cfl=c['cfl'],pole_cfl=c['pole_cfl'])
r={k:read(v) for k,v in m['runs'].items()};common=all(x['time']==target for x in r.values());result=dict(common_target_reached=common,time=target,runs=r,checks=None,all_time_small_relative_spatial=None,all_time_small_relative_closure=None,scope='Native regional constraint-norm timestep differences include restart-milestone effects. They are not norms of pointwise field differences. Unresolved tiny closure effects remain reported. No long stability or closure adoption from this control.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())
if common:
 c,b,f,h,q=[r[k] for k in ['coarse','middle','fine','half','quintic_middle']];assert b['grid']==h['grid']==q['grid'] and b['minimum_Omega']==h['minimum_Omega']==q['minimum_Omega'];assert all(x['values'].keys()==b['values'].keys() for x in [c,f,h,q]);assert h['cfl']==.5*b['cfl'] and h['pole_cfl']==.5*b['pole_cfl'];checks=[]
 for k in b['values']:
  time=abs(b['values'][k]-h['values'][k]);spatial=[abs(c['values'][k]-b['values'][k]),abs(b['values'][k]-f['values'][k])];effect=abs(b['values'][k]-q['values'][k]);checks.append(dict(region_field=k,time_difference=time,spatial_differences=spatial,closure_effect=effect,time_small_relative_spatial=time<=.1*min(spatial),time_small_relative_closure=time<=.1*effect))
 result.update(checks=checks,all_time_small_relative_spatial=all(x['time_small_relative_spatial'] for x in checks),all_time_small_relative_closure=all(x['time_small_relative_closure'] for x in checks))
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if k not in ['runs','checks']})
