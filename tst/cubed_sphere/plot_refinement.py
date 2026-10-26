#!/usr/bin/env python3
"""Plot the measured Hamiltonian response to independent grid refinement."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('analysis',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
data=json.loads(args.analysis.read_text())
fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained',sharey=True)
for ax,axis,resolutions,title in zip(axes,['bulk','shell'],[[32,48,64],[16,24,32]],
                                    ['Refine bulk; shell fixed at 16','Refine shell; bulk fixed at 32']):
    for region,marker in [('bulk','o'),('shell','s')]:
        values=data['responses'][axis][region]['H_l2']['values']
        ax.semilogy(resolutions,values,marker=marker,label=region.capitalize())
    ax.set_title(title,fontsize=11);ax.set_xlabel('Cells per direction')
    ax.set_xticks(resolutions);ax.grid(True,alpha=.25,which='both');ax.legend()
axes[0].set_ylabel('Hamiltonian coordinate-volume L2')
fig.suptitle('Regular Z4c angular pulse, t = 2; independent Aurora controls',fontsize=12)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=180)
