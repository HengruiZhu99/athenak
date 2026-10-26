#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('output',type=Path);a=p.parse_args();j=json.loads(a.review.read_text());fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,c in zip(axes.flat,j['collar_H_M']):
 ax.plot([r['nr'] for r in j['runs']],c['values'],'o-');ax.axvline(164,color='grey',ls=':',lw=1);ax.set_title(c['region_field'].split('/')[-1]);ax.set_xlabel('Radial cells (bulk32, angular24 fixed)');ax.set_ylabel('Outer-collar constraint');ax.grid(alpha=.25)
fig.suptitle('Native puncture t=0.1, quintic Rb=0.98, immutable Minkowski reference\nRadial sensitivity only; full convergence and original monitors still fail',fontsize=12)
fig.savefig(a.output.with_suffix('.png'),dpi=170);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
