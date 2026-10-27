#!/usr/bin/env python3
"""Plot and summarize the independent fixed-background scalar control."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
d=json.loads((a.runs/'review.json').read_text());summary={k:v for k,v in d.items() if k not in ('spatial_runs','temporal_controls')}
summary['spatial_runs']=[{k:v for k,v in row.items() if k!='history'} for row in d['spatial_runs']]
summary['temporal_controls']=[{k:v for k,v in row.items() if k!='history'} for row in d['temporal_controls']]
fig,ax=plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
for row,color in zip(d['spatial_runs'],['#2563eb','#d97706','#15803d']):
 with np.load(a.runs/f"wave-n{row['n']}.npz") as z:
  ax[0].plot(z['time'],z['signals'][:,0],label=f"N={row['n']}",color=color)
  if row==d['spatial_runs'][-1]:ax[0].plot(z['time'],z['signals'][:,1],'k--',label='Analytic F(t)',lw=1)
h=np.array([row['h'] for row in d['spatial_runs']]);e=np.array([row['max_scri_error'] for row in d['spatial_runs']])
ax[1].loglog(h,e,'o-',color='#2563eb',label='Scri waveform max error');ax[1].loglog(h,e[-1]*(h/h[-1])**2,'k--',label='Second order')
ax[0].set(xlabel='Hyperboloidal time (Bondi time at scri)',ylabel='Rescaled scalar at scri',title='Reflected radial scalar pulse')
ax[1].set(xlabel='Radial spacing',ylabel='Maximum error',title='Three-resolution analytic comparison')
for x in ax:x.grid(alpha=.2);x.legend(fontsize=8)
fig.suptitle('Independent fixed-Minkowski control; coupled Einstein–scalar tests remain unrun',fontsize=10)
summary['plots']={}
for ext in ['png','pdf']:
 f=a.output/('fixed-scalar-oracle-v3.'+ext);fig.savefig(f,dpi=180);summary['plots'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(a.output/'fixed-scalar-oracle-v3.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'passed':d['passed_preliminary_control'],'finest_scri_error':float(e[-1]),'observed_orders':d['scri_orders']}))
