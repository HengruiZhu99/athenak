#!/usr/bin/env python3
"""Physical initial-state refinement and cost; no evolution acceptance claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
base=Path(__file__).resolve().parents[2]/'docs/elliptic-shift-3d/evidence';hashes={}
def read(case,file='result/assessment.json',binary=False):
    path=base/case/file;hashes[str(path.relative_to(base))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return np.load(path) if binary else json.loads(path.read_text())
def record(case):
    manifest=read(case,'manifest.json')
    if 'exit_code' not in manifest:raise ValueError('unfinished run must not be plotted as final')
    row=read(case);row['case']=case;return row
two=[record(f'compatible-endpoint-fd-two-layer-n{n}'+('-continued' if n==14 else '')+'-20261027') for n in [10,12,14]]
three=[record(f'compatible-endpoint-fd-wormhole-n{n}-continued-20261027') for n in [10,12]]
old=[record('compatible-wormhole-initial-n10-v3-20261027'),record('compatible-wormhole-initial-n12-20261027')]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold','savefig.dpi':180})
fig,axs=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
ax=axs[0,0]
for rows,label,color,marker in [(old,'Lobatto, 2 layers','#ba4c38','s'),(two,'Local FD, 2 layers','#2878a5','o'),(three,'Local FD, 3 layers','#2d8264','^')]:
    ax.semilogy([r['n'] for r in rows],[r['raw_scri_metric_rate_max'] for r in rows],marker+'-',color=color,label=label)
ax.set(xlabel='Nodes per block axis N',ylabel='Scri maximum |metric-connection rate|',title='A  Boundary error decreases, but still fails',xticks=[10,12,14],ylim=(.08,500))
ax.text(.03,.04,'Acceptance target: 1e−7 (below axis)',transform=ax.transAxes,fontsize=9);ax.legend(fontsize=9)
ax=axs[0,1]
for rows,label,color,marker in [(two,'2 layers','#2878a5','o'),(three,'3 layers','#2d8264','^')]:
    x=[r['n'] for r in rows]
    ax.semilogy(x,[r['radial_shift_reference_rms'] for r in rows],marker+'-',color=color,label=label+', all cells')
    ax.semilogy(x,[r['inner_shift_reference_rms'] for r in rows],marker+'--',color=color,label=label+', r < 0.15')
ax.set(xlabel='Nodes per block axis N',ylabel='Radial shift RMS error vs frozen 1D',title='B  All-cell and puncture-neighborhood errors',xticks=[10,12,14]);ax.legend(fontsize=8)
ax=axs[1,0]
for r,label,color in [(two[0],'N=10, 2 layers','#9bc1d4'),(two[-1],'N=14, 2 layers','#2878a5'),(three[-1],'N=12, 3 layers','#2d8264')]:
    profile=read(r['case'],'result/profile.npz',True)
    X=profile['coordinates'];radius=profile['radius'];W=profile['weights'];beta=profile['beta']
    radial=np.sum(beta*X,axis=-1)/radius;error=radial-profile['reference_beta']
    edges=np.linspace(0,1+1e-12,41);group=np.clip(np.digitize(radius.ravel(),edges)-1,0,39)
    weights=np.bincount(group,weights=W.ravel(),minlength=40)
    sums=np.bincount(group,weights=(W*error**2).ravel(),minlength=40)
    y=np.full(40,np.nan);valid=weights>0;y[valid]=np.sqrt(sums[valid]/weights[valid])
    ax.semilogy((edges[:-1]+edges[1:])/2,y,'o-',ms=3,color=color,label=label)
ax.set(xlabel='Compactified radius r',ylabel='Volume-weighted radial-bin shift error',title='C  Spatial error distribution, all nodes included',xlim=(0,1),ylim=(1e-6,.2));ax.legend(fontsize=8)
ax.text(.03,.04,'N=10 boundary-only bin is at roundoff (below axis)',transform=ax.transAxes,fontsize=8)
ax=axs[1,1]
for r,color in zip(two,['#9bc1d4','#4d91b3','#174b76']):
    history=[];offset=0
    if 'linear_continuation' in r:
        prior=read(r['linear_continuation']['case']);history+=prior['history'];offset=r['linear_continuation']['prior_iterations']
    history += [dict(row,iteration=row['iteration']+offset) for row in r['history']]
    ax.semilogy([x['iteration'] for x in history],[x['original_rms'] for x in history],color=color,label=f"N={r['n']}")
    ax.scatter(history[-1]['iteration'],history[-1]['original_rms'],color=color,s=20)
ax.axhline(1e-9,color='0.3',ls='--',lw=1,label='Bulk solve RMS target')
ax.set(xlabel='Total Krylov iterations, including continuation',ylabel='Original bulk/Dirichlet residual RMS',title='D  Solve cost remains high');ax.legend(fontsize=8)
for ax in axs.flat:ax.grid(alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Local endpoint shell study at t=0 — stable 3D evolution remains unverified',fontsize=13,fontweight='semibold')
for ext in ['png','pdf']:fig.savefig(args.output/f'local-endpoint-study.{ext}')
summary={'scope':__doc__,'families':{},'input_sha256':hashes}
for key,rows in [('two_layers',two),('three_layers',three)]:
    summary['families'][key]=[{k:r[k] for k in ['case','n','cells','original_rms','raw_scri_metric_rate_max','all_node_metric_rate_rms','radial_shift_reference_rms','inner_shift_reference_rms','gates']} for r in rows]
    summary[key+'_observed_orders']={field:[float(np.log(rows[i-1][field]/rows[i][field])/np.log((rows[i]['n']-1)/(rows[i-1]['n']-1))) for i in range(1,len(rows))] for field in ['raw_scri_metric_rate_max','radial_shift_reference_rms','inner_shift_reference_rms']}
(args.output/'assessment.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
