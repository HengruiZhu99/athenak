#!/usr/bin/env python3
"""Plot observed full-domain histories from the ongoing puncture screens."""
import argparse,csv,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for name,label in [('uniform32','Uniform h=M/2'),('uniform64','Uniform h=M/4'),('smr32','SMR finest h=M/4')]:
 records=[]
 for f in (a.runs/(name+'-t2-cpu')).glob('intrinsic-diagnostics-*.csv'):
  rows=list(csv.DictReader(f.open()));records.append((float(rows[0]['time']),np.array([float(r['RMS']) for r in rows])))
 final=json.loads((a.runs/(name+'-t2-analysis.json')).read_text());records.append((final['time'],np.array(final['regions']['full']['RMS'])))
 records.sort(key=lambda v:v[0]);t=np.array([r[0] for r in records]);v=np.array([r[1] for r in records])
 curves=[v[:,0],np.sqrt(np.sum(v[:,7:11]**2,axis=1)),np.sqrt(np.sum(v[:,11:41]**2,axis=1)),np.sqrt(np.sum(v[:,41:71]**2,axis=1))]
 for ax,y in zip(axes.flat,curves):ax.plot(t,y,marker='.',label=label)
for ax,title in zip(axes.flat,['Physical Hamiltonian RMS','Combined C/Z RMS','Combined reduction RMS','Combined intrinsic curl RMS']):
 ax.set(title=title,xlabel='Coordinate time / M',xlim=(0,2));ax.grid(alpha=.25);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
axes[0,0].legend(fontsize=9);fig.suptitle('Single puncture: early full-domain screens (convergence not yet qualified)')
fig.savefig(a.output,dpi=160)
