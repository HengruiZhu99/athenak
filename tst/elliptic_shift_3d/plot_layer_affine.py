#!/usr/bin/env python3
"""Distinguish sampled-reference improvement, physical solve error and cost."""
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
    if 'exit_code' not in manifest:raise ValueError('cannot plot a nonterminal solve')
    r=read(case);r['case']=case;return r
global_case=record('compatible-endpoint-fd-wormhole-n10-continued-20261027')
layer=record('compatible-layer-affine-wormhole-n10-20261027')
concentric=record('compatible-concentric-pcg-n10-20261027')
layer_cg=record('compatible-layer-affine-pcg-n10-20261027')
concentric_gmres=record('compatible-concentric-wormhole-n10-20261027')
trace_cg=record('compatible-layer-trace-mass-pcg-n10-20261027')
layout=read('compatible-layer-affine-layout-audit-20261027')['rows'][1]
concentric_reference=read('compatible-concentric-reference-audit-20261027')['rows'][0]
sampled=[layout['sampled_reference'][1]['scri_max'],layout['sampled_reference'][2]['scri_max'],concentric_reference['sampled_reference'][2]['scri_max']]
rows=[global_case,layer,concentric];labels=['Global correction\n3 blended layers','Layer correction\n3 blended layers','Layer correction\n4 concentric layers']
colors=['#ad5545','#2878a5','#2d8264']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold','savefig.dpi':180})
fig,axs=plt.subplots(2,2,figsize=(11.7,8.4),layout='constrained')
ax=axs[0,0];x=np.arange(3)
ax.semilogy(x,sampled,'o--',color='#817584',label='Sampled 1D field, before solving')
ax.semilogy(x,[r['raw_scri_metric_rate_max'] for r in rows],'s-',color='#ad5545',label='Actual 3D solved field')
ax.axhline(1e-7,color='0.35',ls=':',label='Unchanged scri target')
ax.set(xticks=x,xticklabels=labels,ylabel='Maximum scri metric-connection rate',title='A  Reference accuracy is not solve accuracy',ylim=(1e-13,30));ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
ax=axs[0,1]
ax.bar(x-.17,[r['radial_shift_reference_rms'] for r in rows],width=.32,color=colors,label='All cells')
ax.bar(x+.17,[r['inner_shift_reference_rms'] for r in rows],width=.32,color=colors,alpha=.4,hatch='//',label='r < 0.15')
ax.set(xticks=x,xticklabels=labels,yscale='log',ylabel='Radial shift RMS error vs frozen 1D',title='B  Radial agreement improves with layout');ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
ax=axs[1,0]
for r,color,label in zip(rows,colors,['Global, blended','Layer, blended','Layer, concentric']):
    f=read(r['case'],'result/profile.npz',True);X=f['coordinates'];radius=f['radius'];W=f['weights']
    error=np.sum(f['beta']*X,axis=-1)/radius-f['reference_beta']
    edges=np.linspace(0,1+1e-12,41);group=np.clip(np.digitize(radius.ravel(),edges)-1,0,39)
    weights=np.bincount(group,weights=W.ravel(),minlength=40)
    sums=np.bincount(group,weights=(W*error**2).ravel(),minlength=40)
    y=np.full(40,np.nan);valid=weights>0;y[valid]=np.sqrt(sums[valid]/weights[valid])
    ax.semilogy((edges[:-1]+edges[1:])/2,y,'o-',ms=3,color=color,label=label)
ax.axvspan(.3667508104,.6919875107,color='0.4',alpha=.08,label='Initial slice transition')
ax.set(xlabel='Compactified radius r',ylabel='Volume-weighted radial-bin shift error',title='C  Every active core and shell node retained',xlim=(0,1),ylim=(1e-6,.1));ax.legend(fontsize=8)
ax.text(.03,.04,'Concentric boundary-only bin is at roundoff (below axis)',transform=ax.transAxes,fontsize=8)
ax=axs[1,1]
for r,color,style,label in [(layer,'#2878a5','-','Blended GMRES'),(layer_cg,'#2878a5','--','Blended PCG'),(concentric_gmres,'#2d8264','-','Concentric GMRES (capped)'),(concentric,'#2d8264','--','Concentric PCG'),(trace_cg,'#ad5545',':','Blended trace-weight PCG')]:
    ax.semilogy([h['iteration'] for h in r['history']],[h['original_rms'] for h in r['history']],style,color=color,label=label)
ax.axhline(1e-9,color='0.35',ls=':',label='Bulk RMS target')
ax.set(xlabel='Krylov iterations',ylabel='Original bulk/Dirichlet residual RMS',title='D  Solver control on the same physical equations');ax.legend(fontsize=8)
for ax in axs.flat:ax.grid(alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Initial gauge at t=0, block N=10 — boundary failure persists; no evolution claim',fontsize=13,fontweight='semibold')
for ext in ['png','pdf']:fig.savefig(args.output/f'layer-affine-study.{ext}')
summary={'scope':__doc__,'input_sha256':hashes,'physical_cases':[{k:r[k] for k in ['case','cells','iterations','original_rms','raw_scri_metric_rate_max','all_node_metric_rate_rms','radial_shift_reference_rms','inner_shift_reference_rms','gates']} for r in rows],
         'sampled_reference_scri_max':sampled,'solver_control_cases':[r['case'] for r in [layer,layer_cg,concentric_gmres,concentric,trace_cg]]}
(args.output/'assessment.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
