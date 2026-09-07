#!/usr/bin/env python3
"""Plot independent diagnostic convergence, separately for H and M."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=json.loads(a.input.read_text());fig,axes=plt.subplots(1,2,figsize=(8.2,3.5))
for ax,field in zip(axes,['H','M']):
    for r in rows:
        ax.loglog([16,32,64],r[field+'_errors'],'o-',label=f"FD{r['order']}")
    ax.set(xlabel='Cells in varying direction',ylabel=f'Maximum absolute {field} error')
    ax.set_xticks([16,32,64],['16','32','64']);ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(True,which='both',alpha=.2);ax.legend()
fig.suptitle('Independent snapshot diagnostics: analytic conformal/sheared geometry')
fig.tight_layout();fig.savefig(a.output,dpi=180)
