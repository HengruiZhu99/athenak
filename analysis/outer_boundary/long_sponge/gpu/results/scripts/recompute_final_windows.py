#!/usr/bin/env python3
"""Recompute every final C/D metric and window from packaged lossless histories."""
from pathlib import Path
import argparse,json
import numpy as np
root=Path(__file__).resolve().parents[1]
def summarize(t,y):
 result={'initial':float(y[0]),'final':float(y[-1]),'final_over_initial':float(y[-1]/y[0]) if y[0]>0 else None,'windows':[]}
 for lo,hi in [(10000,20000),(20000,30000),(30000,40000),(40000,50000)]:
  mask=(t>=lo-1e-7)&(t<=hi+1e-7)&(y>0);x=t[mask];v=y[mask]
  if len(x)<20:continue
  z=np.log(v);slope,intercept=np.polyfit(x-x[0],z,1);pred=intercept+slope*(x-x[0]);den=np.sum((z-z.mean())**2)
  result['windows'].append({'actual_time':[float(x[0]),float(x[-1])],'first':float(v[0]),'last':float(v[-1]),'minimum':float(v.min()),'maximum':float(v.max()),'end_over_start':float(v[-1]/v[0]),'max_over_min':float(v.max()/v.min()),'empirical_log_slope_per_M':float(slope),'R_squared':float(1-np.sum((z-pred)**2)/den) if den>0 else None})
 return result

def series(job,case,kind):
 with np.load(root/'histories'/(job+'-'+case+'.npz'),allow_pickle=False) as f:
  return {str(n):f[kind][:,i] for i,n in enumerate(f[kind+'_columns'])}
results={}
for job,name in [('8842248','theta_primary'),('8842248','theta_lapse01'),('8842283','theta_amplitude')]:
 c=series(job,name,'z4c');h=series(job,name,'user');t=c['time']
 fields={k+'_exterior_RMS':np.sqrt(c[key]/c['Volume']) for k,key in [('Theta','Theta-norm'),('H','H-norm2'),('M','M-norm2'),('Z','Z-norm2')]}
 fields.update({'Theta_core_L2':np.sqrt(c['Theta-int2']),'H_core_L2':np.sqrt(c['H-int2']),'M_core_L2':np.sqrt(c['M-int2'])})
 metrics={key:summarize(t,v) for key,v in fields.items()}
 for key in ['Theta-max','alpha-res','beta-res','Gam-res']:metrics[key]=summarize(h['time'],h[key])
 results[name]=metrics
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
if args.check:
 reference=json.loads((root/'final-cd-comparison.json').read_text())
 def compare(a,b,path):
  if isinstance(a,dict):
   assert a.keys()==b.keys(),path
   for key in a:compare(a[key],b[key],path+'/'+key)
  elif isinstance(a,list):
   assert len(a)==len(b),path
   for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
  elif a is None:assert b is None,path
  else:assert np.isclose(a,b,rtol=1e-9,atol=1e-25), (path,a,b)
 for name,metrics in results.items():compare(metrics,reference['cases'][name]['metrics'],name)
 print(json.dumps({'passed':True,'cases':len(results),'metric_series':sum(len(v) for v in results.values()),'windows_per_series':4}))
else:print(json.dumps(results,indent=2))
