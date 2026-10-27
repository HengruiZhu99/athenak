#!/usr/bin/env python3
"""Regionwise finite-Frechet versus continuum errors, with every sample retained."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ['MPLCONFIGDIR']=str(a.output/'.mplconfig');os.environ['XDG_CACHE_HOME']=str(a.output/'.cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((a.input/'manifest.json').read_text());report={'scope':'Independent arbitrary nonspherical nontracefree metric Frechet and analytic continuum consistency; no solve/evolution/operator adoption','input_manifest_sha256':sha(a.input/'manifest.json'),'source_sha256':sha(Path(__file__)),'all_samples_retained':True,'weights':'map_volume*sqrt(det g) for bulk; chi=1 synthetic; uniform point weights on surface','cases':{},'inputs':{}}
for case,summary in m['result'].items():
 path=a.input/(case+'.csv.gz');report['inputs'][str(path)]=sha(path);q=np.genfromtxt(path,delimiter=',',names=True,deletechars='');n=int(case.split('-')[0][1:]);bulk=q['surface']==0;surf=~bulk;pid=q['point'].astype(int);j=pid%n;k=pid//n%n
 regions={'all_volume':bulk,'cartesian':bulk&(q['face']<0),'shells':bulk&(q['face']>=0),'refined_cartesian':bulk&(q['face']<0)&(q['refinement']>0),'coarse_cartesian':bulk&(q['face']<0)&(q['refinement']==0),'surface_all':surf,'surface_edges_and_corners':surf&((j==0)|(j==n-1)|(k==0)|(k==n-1)),'surface_corners':surf&((j==0)|(j==n-1))&((k==0)|(k==n-1))}
 for layer in range(7):regions['shell_layer_'+str(layer)]=bulk&(q['layer']==layer)
 names=['continuum_error']+[x for x in summary['errors'] if x.startswith('finite_epsilon')];output={}
 for region,mask in regions.items():
  if not mask.any():continue
  w=q['conformal_weight'][mask] if region.startswith('surface')==False else np.ones(mask.sum());data={'count':int(mask.sum()),'weight_sum':float(w.sum()),'fields':{}}
  for name in names:
   v=np.stack([q[name+'_'+str(i)] for i in range(3)],axis=1);v2=np.sum(v*v,axis=1);vmax=np.max(np.abs(v),axis=1);indices=np.flatnonzero(mask);top=indices[np.argsort(vmax[mask])[-10:][::-1]];data['fields'][name]={'vector_rms':float(np.sqrt(np.sum(w*v2[mask])/w.sum())),'component_max':float(vmax[mask].max()),'top10':[{'point':int(pid[i]),'block':int(q['block'][i]),'refinement':int(q['refinement'][i]),'face':int(q['face'][i]),'layer':int(q['layer'][i]),'xyz':[float(q[x][i]) for x in ['x','y','z']],'components':v[i].tolist()} for i in top]}
  output[region]=data
 report['cases'][case]=output
report['observed_continuum_order']={}
for direction in [0,1]:
 vals=[report['cases'][f'n{n}-refine0-direction{direction}']['all_volume']['fields']['continuum_error']['vector_rms'] for n in [6,8,10]];report['observed_continuum_order'][str(direction)]=[float(np.log(vals[i]/vals[i+1])/np.log([8/6,10/8][i])) for i in [0,1]]
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for direction in [0,1]:
 for n in [6,8,10]:
  case=f'n{n}-refine0-direction{direction}';data=report['cases'][case];eps=np.array([1e-3,1e-4,1e-5,1e-6,1e-7]);names=[f'finite_epsilon_{e:g}_error' for e in eps]
  for ax,region in zip(axs,['all_volume','surface_all']):ax.loglog(eps,[data[region]['fields'][x]['vector_rms'] for x in names],marker='o',label=f'N{n}, direction{direction}')
for ax,title in zip(axs,['Volume: same-halo Frechet error','Scri: same-surface Frechet error']):ax.set_title(title);ax.set_xlabel('Central perturbation epsilon');ax.set_ylabel('Vector RMS error');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7)
fig.savefig(a.output/'finite-frechet-error.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for d in [0,1]:
 for ax,region in zip(axs,['all_volume','surface_all']):
  ns=np.array([6,8,10]);ys=[report['cases'][f'n{n}-refine0-direction{d}'][region]['fields']['continuum_error']['vector_rms'] for n in ns];ax.loglog(ns,ys,'o-',label=f'Direction{d}');ax.loglog(ns,ys[0]*(6/ns)**4,':',label=f'N^-4 guide{d}')
for ax,title in zip(axs,['Volume: continuum AD error','Scri: continuum AD error']):ax.set_title(title);ax.set_xlabel('Cells per block axis N');ax.set_ylabel('Vector RMS error');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
fig.savefig(a.output/'continuum-consistency.png',dpi=160);plt.close(fig)
report['plots_sha256']={f.name:sha(f) for f in a.output.glob('*.png')};(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.output/'.gitignore').write_text('/.cache/\n/.mplconfig/\n');print(json.dumps({'observed_continuum_order':report['observed_continuum_order'],'cases':len(report['cases'])}))
