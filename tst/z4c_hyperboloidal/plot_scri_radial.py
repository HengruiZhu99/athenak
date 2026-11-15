#!/usr/bin/env python3
"""Plot the complete five-region exterior partition from validated histories."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scri_radial_partition import partition

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',action='append',required=True,help='label=directory')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
runs={label:partition(Path(path)) for label,path in (entry.split('=',1) for entry in args.run)}
(args.output/'radial-partition.json').write_text(json.dumps(runs,indent=2)+'\n')
labels=['AH exterior, r<.80','.80 ≤ r < .88','.88 ≤ r < .94','.94 ≤ r < .97','.97 ≤ r ≤ 1']
units=[r'$M_0^2 H_{\rm RMS}$',r'$M_0^2\mathcal{M}_{\rm RMS}$',r'$M_0\Theta_{\rm RMS}$',r'$M_0 Z_{\rm RMS}$']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(4,5,figsize=(16,10),sharex=True,layout='constrained')
for label,r in runs.items():
    values=np.array(r['rms_by_time_region_constraint'])
    for component in range(4):
        for region in range(5):
            ax=axs[component,region]
            ax.semilogy(r['time_M0'],np.ma.masked_less_equal(values[:,region,component],0),label=label,lw=1.2)
for component in range(4):
    for region in range(5):
        ax=axs[component,region];ax.grid(alpha=.2,which='both')
        if component==0:ax.set_title(labels[region])
        if region==0:ax.set_ylabel(units[component])
        if component==3:ax.set_xlabel(r'$t/M_0$')
axs[0,0].legend(fontsize=8)
fig.suptitle('Constraint histories across the full horizon exterior')
fig.supxlabel('Compact-coordinate volume-weighted RMS; the five disjoint regions exhaust the exterior. Exact zeros omitted.',fontsize=9)
for ext in ('pdf','png'):fig.savefig(args.output/f'full-exterior-radial-history.{ext}',dpi=160)
plt.close(fig)

fig,axs=plt.subplots(1,4,figsize=(14,4),layout='constrained')
x=np.arange(5);width=.8/len(runs)
for j,(label,r) in enumerate(runs.items()):
    fractions=100*np.array(r['squared_integral_fractions'])[-1]
    for i,ax in enumerate(axs):
        ax.bar(x+(j-(len(runs)-1)/2)*width,fractions[:,i],width=width,label=label)
for ax,title in zip(axs,['Hamiltonian','Momentum','Theta','Z']):
    ax.set_title(title);ax.set_xticks(x,['inner','shell 0','shell 1','shell 2','shell 3'],rotation=30)
    ax.set_ylabel('Fraction of exterior squared norm (%)');ax.grid(alpha=.2,axis='y')
axs[0].legend(fontsize=8)
fig.suptitle('Which radial regions contribute to the final exterior norm?')
fig.supxlabel('Volume integrals of squared constraints; fractions sum to 100% for each field and run.',fontsize=9)
for ext in ('pdf','png'):fig.savefig(args.output/f'full-exterior-radial-fractions.{ext}',dpi=160)
