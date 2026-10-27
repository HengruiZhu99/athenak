#!/usr/bin/env python3
"""Retain failed gates, physical-volume diagnostics, raw angular scatter and 1D controls."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radial_reference import interpolate
p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
ref=np.load(a.reference/'radial-n513.npz');fields=['alpha','chi','g_rr','A_rr','Khat','Theta','Lambda_r','beta_r']
summary=dict(scope='coarse failed research pilots, no late-time or production claim',charts={},reference_controls={})
for n in [129,257]:
 old=np.load(a.reference/f'radial-n{n}.npz');r=np.linspace(.02,.98,97)
 summary['reference_controls'][str(n)]={f:float(np.max(abs(interpolate(old['r'],old[f],r)-interpolate(ref['r'],ref[f],r)))) for f in fields}
for mode in [0,1,2]:
 out=a.matrix/f'chart-{mode}';manifest=json.loads((out/'manifest.json').read_text());rows=[]
 for case in manifest['cases']:
  name=case['name']
  if not name.startswith('wormhole-'):continue
  row=dict(name=name,exit_code=case['exit_code'],result=case.get('result',{}));data=np.genfromtxt(out/(name+'.csv'),delimiter=',',names=True)
  errors={};w=data['weight'];r=data['r']
  for f in fields:
   d=data[f]-interpolate(ref['r'],ref[f],r);errors[f]=dict(weighted_rms=float(np.sqrt(np.average(d*d,weights=w))),point_max=float(max(abs(d))))
  row['comparison_to_t0002_radial']=errors if case['result']['steps'] else None
  row['constraint_regions']={}
  for label,mask in [('all',np.ones(len(r),bool)),('puncture_r_lt_01',r<.1),('bulk_r_01_03',(r>=.1)&(r<.3)),('transition_r_03_08',(r>=.3)&(r<.8)),('outer_r_ge_08',r>=.8),('scri_r_ge_09',r>=.9)]:
   row['constraint_regions'][label]=dict(cells=int(sum(mask)),volume_quadrature=float(sum(w[mask])),
    H_rms=float(np.sqrt(np.average(data['H'][mask]**2,weights=w[mask]))),H_max=float(max(abs(data['H'][mask]))),
    M_rms=float(np.sqrt(np.average(data['M_norm'][mask]**2,weights=w[mask]))),M_max=float(max(data['M_norm'][mask])))
  if case['result']['steps']:
   spheres=np.genfromtxt(out/(name+'.csv.spheres.csv'),delimiter=',',names=True);rad=np.unique(spheres['r']);angular={}
   for f in fields:
    values=spheres[f].reshape(-1,64);truth=interpolate(ref['r'],ref[f],rad)
    angular[f]=dict(radii=rad.tolist(),mean=values.mean(axis=1).tolist(),scatter=values.std(axis=1).tolist(),
     mean_error=(values.mean(axis=1)-truth).tolist(),maximum_scatter=float(max(values.std(axis=1))))
   row['raw_angular_comparison']=angular
  rows.append(row)
 base=np.genfromtxt(out/'wormhole-n8-steps2.csv',delimiter=',',names=True)
 control={}
 names=[f'state_{i}' for i in range(22)]+['delta_beta_x','delta_beta_y','delta_beta_z']
 for label in ['dt-half','tight-tolerance']:
  other=np.genfromtxt(out/(label+'.csv'),delimiter=',',names=True)
  assert np.array_equal(base['r'],other['r'])
  control[label]=dict(maximum_stored_field_or_shift_difference=float(max(np.max(abs(base[f]-other[f])) for f in names)))
 summary['charts'][str(mode)]=dict(source_stable=manifest.get('sources_unchanged'),binary_stable=manifest.get('binary_unchanged'),
  zero_exits=sum(c['exit_code']==0 for c in manifest['cases']),failed_exits=sum(c['exit_code']!=0 for c in manifest['cases']),
  cases=rows,independent_controls=control)
(a.matrix/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for mode,label in [(0,'Generic powers'),(1,'Rejected compact blend'),(2,'Smooth AF factors')]:
 rows=summary['charts'][str(mode)]['cases'];initial=[r['result'] for r in rows if r['result']['steps']==0];final=[r['result'] for r in rows if r['result']['steps']==2]
 for ax,key,values,title in [(axes[0,0],'H_rms',initial,'Initial H RMS, all cells'),(axes[0,1],'M_rms',initial,'Initial M RMS, all cells'),
    (axes[1,0],'scri_tensor_pole',final,'Tensor scri pole at t=0.0002'),(axes[1,1],'scri_raw_clock_rate',final,'Unforced scri clock rate at t=0.0002')]:
  ax.loglog([x['n'] for x in values],[x[key] for x in values],'o-',label=label);ax.set_title(title);ax.set_xlabel('N per block axis');ax.grid(alpha=.25)
axes[1,0].axhline(1e-7,color='k',ls='--',label='unchanged pole target');axes[0,0].legend(fontsize=8)
fig.suptitle('Required M=1/3 wormhole: exploratory 3D pilots fail coupled gates')
fig.savefig(a.matrix/'wormhole-failed-convergence.png',dpi=160);plt.close(fig)
row=next(c for c in summary['charts']['2']['cases'] if c['name']=='wormhole-n12-steps2')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,f in zip(axes.flat,['alpha','chi','Khat','beta_r']):
 q=row['raw_angular_comparison'][f];r=np.array(q['radii']);mean=np.array(q['mean']);sd=np.array(q['scatter'])
 ax.plot(ref['r'],ref[f],color='k',label='Independent spherical N513');ax.plot(r,mean,'o-',ms=3,label='3D angular mean, N12')
 ax.fill_between(r,mean-sd,mean+sd,alpha=.25,label='Raw angular scatter');ax.set(xlabel='compact radius',ylabel=f);ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8);fig.suptitle('t=0.0002 comparison; coarse 3D run is not accepted')
fig.savefig(a.matrix/'wormhole-radial-comparison.png',dpi=160);plt.close(fig)
print(json.dumps(dict(charts={k:dict(zero_exits=v['zero_exits'],failed_exits=v['failed_exits'],controls=v['independent_controls']) for k,v in summary['charts'].items()},reference_controls=summary['reference_controls'])))
