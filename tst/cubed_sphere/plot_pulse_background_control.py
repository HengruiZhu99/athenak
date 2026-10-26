#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('research',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
paths={'Angular pulse':[a.research/'long-stage01/pulse',a.research/'long-stage02/pulse',a.research/'puncture-spatial-diagnosis/pulse-stage03'],'Stationary native background':[a.research/'pulse-growth-diagnosis-v2/stationary-middle',*sorted((a.research/'pulse-growth-continuation01/stationary-middle').glob('segment*'))]}
fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
for label,roots in paths.items():
 series={}
 for root in roots:
  for row in csv.DictReader((root/'cubed_sphere_conformal.csv').open()):series[(row['region'],float(row['time']))]=row
 for r,region in enumerate(['bulk','shell']):
  data=[v for (reg,t),v in sorted(series.items()) if reg==region]
  for c,key in enumerate(['H_l2','M_conformal_l2','Z_conformal_l2']):
   ax=axes[r,c];ax.semilogy([float(x['time']) for x in data],[float(x[key]) for x in data],label=label);ax.set(xlabel='t',title=region+' '+key.replace('_conformal','').replace('_l2',' RMS'));ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Same middle grid and native equations: angular pulse versus undisturbed Minkowski\nBoth shown through t=0.1; immutable reference, no RHS subtraction. Growth is not a stability acceptance.')
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.pdf'))
