#!/usr/bin/env python3
"""Plot retained component evidence without rerunning or filtering simulations."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
inputs={}
def read(name):
    path=a.evidence/name/'result/assessment.json';inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
mapped=read('compatible-mapped-shells-thin-20261027')
baseline=read('compatible-cartesian-linear-coarse-20261027')
extended=read('compatible-cartesian-extended-solve-20261027')
upwind=read('compatible-cartesian-upwind-extended-20261027')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axes=plt.subplots(1,3,figsize=(13.2,3.7),layout='constrained')
rows=mapped['rows'];ns=[r['n'] for r in rows]
axes[0].semilogy(ns,[r['rms'] for r in rows],'o-',color='#0072B2',label='All-cell RMS')
axes[0].semilogy(ns,[r['maximum'] for r in rows],'s--',color='#D55E00',label='Maximum')
axes[0].set(title='Mapped derivative accuracy',xlabel='Nodes per block direction',ylabel='Absolute derivative error',xticks=ns)
axes[0].legend(frameon=False)
for r,label,color,style in [(baseline,'Penalty block: 400-step cap','#999999',':'),(extended,'Penalty block: extended cap','#0072B2','-'),(upwind,'Interface-based block','#D55E00','-')]:
    h=r['history'];axes[1].semilogy([v['iteration'] for v in h],[v['original_rms'] for v in h],style,color=color,label=label)
axes[1].axhline(1e-9,color='black',ls='--',lw=.8,label='Unchanged RMS target')
axes[1].set(title='Periodic Cartesian AMR gauge solve',xlabel='PCG iteration',ylabel='Recomputed original residual RMS')
axes[1].legend(frameon=False,fontsize=8)
axes[2].semilogy(ns,[r['surface_discrete_normal_error'] for r in rows],'o-',color='#CC79A7')
axes[2].set(title='Boundary geometry limitation',xlabel='Nodes per block direction',ylabel='Maximum unit-normal error',xticks=ns)
axes[2].text(.04,.08,'Outer nodes: |Ω| ≤ 2.3 × 10⁻¹⁶\nPhysical scri evolution: untested',transform=axes[2].transAxes,fontsize=9)
for ax in axes:ax.grid(True,which='major',alpha=.2)
fig.suptitle('Spatial component evidence — not a Schwarzschild evolution',fontsize=13)
fig.savefig(a.output/'compatible-progress.png');fig.savefig(a.output/'compatible-progress.pdf');plt.close(fig)
manifest={'scope':'Plots of complete retained component assessments, not evolution results','inputs_sha256':inputs,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
