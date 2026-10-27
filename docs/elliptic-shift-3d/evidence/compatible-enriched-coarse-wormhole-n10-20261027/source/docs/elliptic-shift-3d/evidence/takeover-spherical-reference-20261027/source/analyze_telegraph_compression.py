#!/usr/bin/env python3
"""Compare measured stage-ray compression with the leading AF shift model.

The endpoint model is diagnostic, not a decomposition along the actual rays.
Saved-time trapezoidal integration cannot replace the RK-stage characteristic
integration or its independent temporal convergence test.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegraph_analysis_io import trajectory,first_derivative_stencil
p=argparse.ArgumentParser()
p.add_argument('--cases',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();records=[]
for folder in a.cases:
 manifest=json.loads((folder/'manifest.json').read_text())
 assert manifest['status']=='finished'
 data,ancestry=trajectory(folder)
 start=manifest['arguments']['ray_start']
 times=sorted(t for t in data if t>=start);assert times[0]==start
 rows=[]
 for t in times:
  d=data[t];idx,w=first_derivative_stencil(tuple(d['r']))
  b1=float(np.dot(w[0],d['beta'][idx[0]]-d['beta'][0]))
  rows.append(dict(t=t,beta_linear_coefficient=b1,min_inward_logJ=float(min(d['rays'][::2,1]))))
 b=np.array([r['beta_linear_coefficient'] for r in rows]);ts=np.array(times)
 # c=-beta+O(r^2) near the AF end gives d log(J_aff)/dt=-b1.
 integral=np.r_[0,np.cumsum(np.diff(ts)*(b[1:]+b[:-1])/2)]
 for row,I in zip(rows,integral):
  row['affine_shift_logJ']=float(-I)
  row['excess_log_compression']=float(row['min_inward_logJ']+I)
 last=[next(r for r in rows if r['t']==t) for t in [8.,10.]]
 change=lambda key:last[1][key]-last[0][key]
 records.append(dict(path=str(folder.resolve()),n=manifest['arguments']['n'],ancestry=ancestry,
  manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),rows=rows,
  late_interval=[8,10],measured_logJ_change=change('min_inward_logJ'),affine_shift_logJ_change=change('affine_shift_logJ'),
  affine_to_measured_change_ratio=change('affine_shift_logJ')/change('min_inward_logJ')))
a.output.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
colors=['#4477AA','#EE7733','#228833']
for rec,color in zip(records,colors):
 rows=rec['rows'];t=[r['t'] for r in rows]
 axes[0].plot(t,[r['min_inward_logJ'] for r in rows],color=color,label=f"Stage rays, N={rec['n']}")
 axes[0].plot(t,[r['affine_shift_logJ'] for r in rows],color=color,ls='--',alpha=.8)
 axes[1].plot(t,[r['excess_log_compression'] for r in rows],color=color,label=f"N={rec['n']}")
axes[0].plot([],[],color='0.4',ls='--',label='Leading AF shift model')
axes[0].set(title='Measured compression and coordinate contraction',xlabel='t',ylabel=r'$\log J$')
axes[1].set(title='Compression beyond the endpoint shift model',xlabel='t',ylabel=r'$\min\log J+\int b_1\,dt$')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(a.output/'compression.png',dpi=160);fig.savefig(a.output/'compression.pdf');plt.close(fig)
result=dict(records=records,scope='Stage-ray logJ is measured. The affine model uses the endpoint coefficient beta=b1*r+O(r^2), a seven-node derivative and saved-time trapezoidal integration. Their difference is not an exact split of forces along rays. Minimum seed may vary. Neither small positive J nor this model proves continuum crossing or its absence; no acceptance criterion is replaced.')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k not in ['rows','ancestry']} for r in records],indent=2))
