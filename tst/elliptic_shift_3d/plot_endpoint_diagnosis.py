#!/usr/bin/env python3
"""Plot retained endpoint failures and bounded enriched-solver results."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
base=Path(__file__).resolve().parents[2]/'docs/elliptic-shift-3d/evidence';inputs={}
def read(case,file='result/assessment.json'):
    path=base/case/file;raw=path.read_bytes();inputs[str(path.relative_to(base))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold','savefig.dpi':180})
fig,axs=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
ax=axs[0,0]
cases=['compatible-wormhole-initial-n10-v3-20261027','compatible-wormhole-affine-n10-20261027','compatible-wormhole-affine-buffer-n10-20261027']
labels=['Original pair','Affine correction','Correction +\nouter layer']
records=[read(c) for c in cases];x=np.arange(3)
ax.plot(x,[r['bulk_actual_metric_rate_rms'] for r in records],'o-',color='#2878a5',label='Bulk RMS')
ax.plot(x,[r['raw_scri_metric_rate_max'] for r in records],'s-',color='#ba4c38',label='Scri maximum')
ax.set(yscale='log',xticks=x,xticklabels=labels,ylabel='Actual metric-connection rate',title='A  Initial physical solve: boundary still fails',ylim=(1e-11,2e3),xlim=(-.3,2.3));ax.legend(fontsize=9,loc='center')
ax=axs[0,1];r=read('compatible-scri-high-mode-20261027')
v=[r['original_scri_max'],r['highest_contribution_scri_max'],r['remainder_scri_max']]
bars=ax.bar(['Full failure','Highest radial\nmode contribution','Remaining\ncontribution'],v,color=['#ba4c38','#d69a39','#677b87'])
ax.bar_label(bars,fmt='%.1f',padding=3);ax.set(ylabel='Scri maximum |connection rate|',title='B  Saved 3D failure: diagnostic decomposition',ylim=(0,225))
ax.text(.03,.94,'No contribution removed from a physical solve',transform=ax.transAxes,va='top',fontsize=9)
ax=axs[1,0];r=read('compatible-endpoint-enrichment-20261027')
rows=[v for v in r['rows'] if v['n']==10 and v['source']=='highest_legendre' and v['penalty_scale'] in ['unit','n','n_squared','enriched_shift_control']]
ax.bar(['Unit penalty','10× penalty','100× penalty','Extra shift\ndegree'],[v['endpoint_rate_max'] for v in rows],color=['#ba4c38']*3+['#2878a5'])
ax.set(yscale='log',ylabel='Endpoint divergence maximum',title='C  Scalar design control (not a 3D pass)',ylim=(1e-14,1e4))
ax.text(.03,.94,'The prescribed highest-mode source is unchanged',transform=ax.transAxes,fontsize=9,va='top')
ax=axs[1,1]
for case,label,color in [('compatible-enriched-wormhole-n10-20261027','Boundary block','#ba4c38'),('compatible-enriched-coarse-wormhole-n10-20261027','Expanded coarse space','#d69a39'),('compatible-enriched-line-wormhole-n10-20261027','Coupled radial lines','#2878a5')]:
    manifest=read(case,'manifest.json')
    if 'exit_code' not in manifest:raise ValueError('do not plot an unfinished probe as terminal')
    path=base/case/'result/progress.jsonl';raw=path.read_bytes();inputs[str(path.relative_to(base))]=hashlib.sha256(raw).hexdigest()
    rows=[json.loads(line) for line in raw.decode().splitlines()];rows=[v for v in rows if 'iteration' in v]
    if not rows:continue
    ax.semilogy([v['iteration'] for v in rows],[v['original_rms'] for v in rows],label=label,color=color)
    ax.scatter(rows[-1]['iteration'],rows[-1]['original_rms'],s=20,color=color)
ax.axhline(1e-9,color='0.35',ls='--',lw=1,label='RMS target')
ax.set(xlabel='Krylov iterations',ylabel='All-node original residual RMS',title='D  Enriched full-tensor physical solve',ylim=(1e-11,1e3));ax.legend(fontsize=8,loc='center left')
for ax in axs.flat:ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('Outer-boundary diagnosis — no stable 3D evolution claimed',fontsize=14,fontweight='semibold')
for ext in ['png','pdf']:fig.savefig(a.output/f'endpoint-diagnosis.{ext}')
(a.output/'inputs.json').write_text(json.dumps({'scope':__doc__,'sha256':inputs},indent=2)+'\n')
