#!/usr/bin/env python3
"""Plot complete initial-state evidence, retaining the scri-rate failure."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
inputs={}
def read(path):
    inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return np.load(path) if path.suffix=='.npz' else json.loads(path.read_text())
cases=[]
for name in ['compatible-wormhole-initial-n10-v3-20261027','compatible-wormhole-initial-n12-20261027']:
    d=a.evidence/name/'result';cases.append((read(d/'assessment.json'),read(d/'profile.npz')))
reference=read(a.evidence/'takeover-spherical-reference-20261027/short-n2049/profile-0001.npz')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axes=plt.subplots(2,2,figsize=(10.7,7.2),layout='constrained')
colors=['#0072B2','#D55E00']
axes[0,0].plot(reference['r'],reference['beta'],'-',color='black',lw=1.4,label='1D N=2049, t=0')
all_node=[]
for (result,profile),color in zip(cases,colors):
    label=f"3D block N={result['n']}"
    axes[0,0].scatter(profile['group_radius'],profile['group_mean'],s=7,alpha=.65,color=color,label=label)
    axes[0,1].scatter(profile['group_radius'],profile['group_scatter'],s=7,alpha=.65,color=color,label=label)
    h=result['history'];axes[1,0].semilogy([r['iteration'] for r in h],[r['original_rms'] for r in h],color=color,label=label)
    W=profile['weights'];rate=profile['actual_metric_rate'];all_node.append(float(np.sqrt(np.sum(W[...,None]*rate**2)/np.sum(W))))
axes[0,0].set(title='Initial shift compared with spherical solver',xlabel='Compactified radius r',ylabel='Radial shift βʳ',xlim=(0,1))
axes[0,0].legend(frameon=False,fontsize=9)
axes[0,1].set(title='Actual equal-radius angular scatter',xlabel='Compactified radius r',ylabel='Standard deviation of βʳ',xlim=(0,1))
axes[0,1].legend(frameon=False,fontsize=9)
axes[1,0].axhline(1e-9,ls='--',color='black',lw=.8,label='RMS solve target')
axes[1,0].set(title='Original bulk + Dirichlet-row solve',xlabel='GMRES iteration',ylabel='Recomputed residual RMS')
axes[1,0].legend(frameon=False,fontsize=9)
ns=[r['n'] for r,_ in cases]
axes[1,1].semilogy(ns,[r['raw_scri_metric_rate_max'] for r,_ in cases],'o-',color='#CC79A7',label='Scri maximum (failure)')
axes[1,1].semilogy(ns,all_node,'s-',color='#D55E00',label='All-node RMS, including scri')
axes[1,1].semilogy(ns,[r['bulk_actual_metric_rate_rms'] for r,_ in cases],'^-',color='#0072B2',label='Bulk RMS')
axes[1,1].set(title='Metric-connection rate: boundary still fails',xlabel='Nodes per block direction',ylabel='Absolute metric-connection rate',xticks=ns)
axes[1,1].legend(frameon=False,fontsize=8)
for ax in axes.ravel():ax.grid(True,alpha=.2)
fig.suptitle('Physical Schwarzschild initial gauge — no time evolution',fontsize=13)
fig.savefig(a.output/'wormhole-initial-gauge.png');fig.savefig(a.output/'wormhole-initial-gauge.pdf');plt.close(fig)
manifest={'scope':__doc__,'inputs_sha256':inputs,'all_node_metric_rate_rms':dict(zip(map(str,ns),all_node)),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
