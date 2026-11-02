"""Explicit initial snapshot controls; no evolved-horizon or stability claims."""
from pathlib import Path
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
ns=[16,24,32];fields=['H','M','Mr'];summary={'scope':'Native t=0, common r>.063 cutoff. Exact interior halos replace interpolated values below scri only; direct scri radial donor interpolation and outer ghost lifts remain. Named regions overlap and cannot be summed. Projection changes point values and recomputes all derivatives.','resolutions':{}}
fig,axes=plt.subplots(2,2,figsize=(12,8))
colors=['#b55315','#1769aa','#228844']
for n,color in zip(ns,colors):
 a=json.loads((p/f'patch-n{n}.json').read_text());v={v['name']:v for v in a['variants']};prod=v['production'];ext=prod['regions']['exterior']
 old=json.loads((p/f'n{n}-radial.json').read_text())['norms']['exterior']
 for f in fields:assert abs(ext[f]['rms']/old[f+'_L2']-1)<1e-11,(n,f)
 assert a['max_projection_field_change']<1e-12 and a['scri_constraint_field_change']<1e-12
 rec={'projection_max_field_change':a['max_projection_field_change'],'surface_projection_max_field_change':a['max_surface_projection_field_change'],'scri_constraint_field_change':a['scri_constraint_field_change'],'maximum_locations':{f:ext[f] for f in fields},'controls':{},'cutoff_control':{}}
 for name in ['exact_interior_halos','algebraic_projection']:
  rec['controls'][name]={f:{'relative_norm_change':v[name]['regions']['exterior'][f]['rms']/ext[f]['rms']-1,'difference_rms':v[name]['difference_from_production']['exterior'][f]['rms']} for f in fields}
 rec['regional_squared_fractions']={name:{f:reg[f]['sum2']/ext[f]['sum2'] for f in fields} for name,reg in prod['regions'].items()}
 for cut in prod['cutoffs'][:3]:
  m=cut['measured'];rec['cutoff_control'][str(cut['r'])]={'relative_volume_error':m['volume']/cut['exact_volume']-1,'norms':{f:m[f]['rms'] for f in fields},'exact_volume_normalized':{f:math.sqrt(m[f]['sum2']/cut['exact_volume']) for f in fields}}
 summary['resolutions'][n]=rec
 for ax,f in zip(axes[0],['H','M']):
  bins=prod['radial_bins'];r=np.array([(b['lo']+b['hi'])/2 for b in bins]);y=np.array([math.sqrt(b['measured'][f]['sum2']/ext[f]['sum2']) for b in bins])
  mask=np.array([b['measured']['volume']>0 for b in bins]) & (y>0)
  ax.semilogy(r[mask],y[mask],'.',ms=4,color=color,label=f'N{n}')
  ax.set(xlim=(.063,1),ylim=(1e-14,2),xlabel='Compactified r',ylabel=f'Fractional {f} norm per radial bin',title=f'Initial {f}: where the error lies')
 for ax,f in zip(axes[1],['H','M']):
  regnames=list(prod['regions'])[1:];base=[prod['regions'][k][f]['rms'] for k in regnames];delta=[v['exact_interior_halos']['difference_from_production'][k][f]['rms'] for k in regnames]
  ax.semilogy(range(len(regnames)),np.where(np.array(base)>0,base,np.nan),'o-',color=color,label=f'N{n}: production')
  ax.semilogy(range(len(regnames)),np.where(np.array(delta)>0,delta,np.nan),'x--',color=color,label=f'N{n}: exchange difference')
  ax.set_xticks(range(len(regnames)),['Core','Overlap','Seams','Initial\ntransition','Last 3\nrows','0.775–0.9'],fontsize=9)
  ax.set(ylabel=f'{f} RMS',title=f'Initial {f}: exact-halo substitution control')
for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Native initial-data patch audit: interpolation and projection are subdominant',fontsize=14)
fig.tight_layout(rect=(0,.06,1,.95));fig.text(.5,.012,'Common cutoff r > 0.063; regions overlap. Empty bins and exact zeros omitted. Scri radial donor interpolation and outer ghost lift retained.',ha='center',fontsize=9)
fig.savefig(p/'patch-controls.png',dpi=160)
# Norm orders belong to the production grid, while cutoff variations are separate controls.
summary['effective_orders']={}
for f in fields:
 vals=[summary['resolutions'][n]['maximum_locations'][f]['rms'] for n in ns]
 summary['effective_orders'][f]=[math.log(a/b)/math.log(j/i) for a,b,i,j in zip(vals,vals[1:],ns,ns[1:])]
(p/'patch-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for n,rec in summary['resolutions'].items():
 print(n,'projection',rec['projection_max_field_change'],'exchange',rec['controls']['exact_interior_halos'])
 print('cutoff',rec['cutoff_control'])
 print('maxima',{f:(r['max'],r['max_xyz']) for f,r in rec['maximum_locations'].items()})
