#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('output',type=Path);a=p.parse_args();j=json.loads(a.review.read_text());checks={x['region_field']:x for x in j['checks']};fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
regions={'bulk/fixed_interior':'Bulk interior','bulk/fixed_interface':'Bulk interface','shell/fixed_inner_shell':'Inner shell','shell/fixed_transition':'Full shell transition','shell/fixed_outer_collar':'Outer collar'}
for col,keys in enumerate([['H_l2','H_linf'],['M_conformal_l2','M_conformal_linf'],['Z_conformal_l2','Z_conformal_linf']]):
 for row,key in enumerate(keys):
  ax=axes[row,col]
  for region,label in regions.items():ax.semilogy([16,24,32],checks[region+'/'+key]['values'],'o-',label=label)
  ax.set(title=key.replace('_conformal','').replace('_l2',' RMS').replace('_linf',' maximum'),xlabel='Bulk N; angular/radial refined together',xticks=[16,24,32]);ax.grid(alpha=.2,which='both')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10);fig.suptitle('Native puncture at t=0.1: balanced refinement, immutable Minkowski reference, Rb=0.98\nSix fixed-region quantities remain nonmonotonic; full convergence criterion fails.',fontsize=12)
fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.pdf'))
