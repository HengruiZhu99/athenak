#!/usr/bin/env python3
"""Plot saved native clock pilot diagnostics; no evolution or acceptance change."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('evidence',type=Path);a=p.parse_args();review=json.loads((a.evidence/'native-oracle-pilot-review.json').read_text());controls=json.loads((a.evidence/'paired-grid-timestep-review.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=['#235b8e','#b35136','#548049'];names=['fields','first','configuration_second'];labels=['field','first derivative','configuration Hessian']
# Original coordinate-volume histories are retained and explicitly labelled.
history={}
for case in ['clock-pulse4','clock-pulse-t002','clock-interface-segment1']:
 rows=list(csv.DictReader((a.work/case/'cubed_sphere_native_oracle_errors.csv').open()))
 for row in rows:
  if row['subset']=='all':history.setdefault(round(float(row['time']),14),{})[row['region']]=row
x=[];curves=[[],[],[]]
for t,regions in sorted(history.items()):
 if len(regions)!=2:continue
 x.append(t)
 for j,key in enumerate(['field_RMS','first_RMS','configuration_second_RMS']):curves[j].append(np.sqrt(sum(float(r['volume'])*float(r[key])**2for r in regions.values())/sum(float(r['volume'])for r in regions.values())))
for y,color,label in zip(curves,colors,labels):
 xx=np.array(x);yy=np.array(y);mask=yy>0;axes[0,0].semilogy(xx[mask],yy[mask],color=color,label=label)
axes[0,0].axhline(.01,color='0.5',ls=':',lw=1);axes[0,0].set(xlabel='Coordinate time',ylabel='Normalized RMS error',title='Full native coarse pilot: mapped-volume histories');axes[0,0].legend(fontsize=8)
for name,color,label in zip(names,colors,labels):
 vals=[review['cases'][case]['native_exact_errors']['fixed_reference_Penrose']['all'][name]for case in ['clock-pulse-t002','clock-fine-t002-complete']]
 axes[0,1].loglog([12,18],[v['rms']for v in vals],'o-',color=color,label=label+' RMS');axes[0,1].loglog([12,18],[v['linf']for v in vals],'s--',color=color,alpha=.75,label=label+' max')
axes[0,1].axhline(.01,color='0.5',ls=':',lw=1);axes[0,1].set(xlabel='Bulk cells per axis (angular/radial refined with it)',ylabel='Normalized error',title='Two pilot grids at t=.02: reference-volume norms');axes[0,1].set_xticks([12,18],labels=['12','18']);axes[0,1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator());axes[0,1].legend(fontsize=7,ncol=2)
pairs=list(controls['timestep_self_differences'].values())
for name,color,label in zip(names,colors,labels):axes[1,0].loglog([4e-4,2e-4],[v[name]['rms']for v in pairs],'o-',color=color,label=label)
axes[1,0].set(xlabel='Larger timestep in each adjacent comparison',ylabel='Normalized RMS self-difference',title='Actual timestep trend is about first order');axes[1,0].legend(fontsize=8);axes[1,0].text(.03,.04,'All temporal/fine spatial ratios < 8.5e-5\nNo RK3 order assertion',transform=axes[1,0].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
constraint_names=['H_raw','M_conformal','M_physical','Z_conformal','Z_physical','Theta_physical'];short=['H','M conf','M phys','Z conf','Z phys','Theta'];pos=np.arange(6)
for j,(case,label,color)in enumerate([('clock-pulse-t002','coarse','#235b8e'),('clock-fine-t002-complete','fine','#b35136')]):
 vals=review['cases'][case]['constraints']['fixed_reference_Penrose']['all']['dimensionless'];axes[1,1].bar(pos+(j-.5)*.35,[vals[k]['rms']for k in constraint_names],width=.35,label=label,color=color)
axes[1,1].axhline(.01,color='0.5',ls=':',lw=1);axes[1,1].set(yscale='log',xticks=pos,xticklabels=short,ylabel='Dimensionless RMS',title='Physical and conformal constraints, normalized, t=.02');axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(True,alpha=.2)
fig.suptitle('Native second-order Z4c: finite-cut RWM clock pilots (spatial FD4)',fontsize=14)
fig.supxlabel('Radial oracle only. Coarse interface Hessian excess retained; causal, angular/tensor and numerical exact-scri gates remain open.',fontsize=9)
fig.savefig(a.evidence/'native-clock-pilots.png',dpi=160);fig.savefig(a.evidence/'native-clock-pilots.svg');plt.close(fig)
meta={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{n:hashlib.sha256((a.evidence/n).read_bytes()).hexdigest()for n in ['native-oracle-pilot-review.json','paired-grid-timestep-review.json']},'scope':'Saved pilot plots only; not three-resolution scientific convergence or Goal1 completion.'};(a.evidence/'plot-receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
