#!/usr/bin/env python3
"""Plot actual, unnormalized Theta maxima and stellar-density histories."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_runs import history

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', action='append', nargs=2, metavar=('LABEL','DIRECTORY'), required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for label,directory in a.run:
    found=[]
    for path in Path(directory).glob('*.hst'):
        data,finite=history(path)
        if 'Theta-max' in data:found.append(data)
    assert len(found)==1,directory
    d=found[0];t=d['time'];q=d['Theta-max'];mask=q>0
    if mask.any():axes[0].semilogy(t[mask],q[mask],label=label)
    else:axes[0].plot([],[],label=label+' (exactly zero)',linestyle=':')
    if 'star' in label.lower():
        axes[1].plot(t,d['rho-max']/d['rho-max'][0],label=label)
axes[0].set(xlabel='Time [M]',ylabel='Unexcised max |Theta|',title='Perturbation response; raw amplitudes')
axes[1].set(xlabel='Time [M]',ylabel='Peak density / initial peak',title='Star morphology diagnostic')
for ax in axes:
    ax.grid(alpha=.25)
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
fig.suptitle('R0=M trumpet, residual Z4c; background-adapted gauge; kappa1=0.1, kappa2=0',fontsize=10)
fig.savefig(a.output,dpi=180)
fig.savefig(a.output.with_suffix('.pdf'))
