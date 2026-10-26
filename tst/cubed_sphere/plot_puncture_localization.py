#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('output',type=Path);a=p.parse_args();s=json.loads(a.review.read_text());d={x['region_field']:x for x in s['checks']}
keys={'bulk/fixed_interface/M_conformal_linf':'Bulk interface M maximum','shell/fixed_inner_shell/H_linf':'Inner shell H maximum','shell/fixed_inner_shell/M_conformal_linf':'Inner shell M maximum','shell/all/H_l2':'Whole shell H RMS','shell/all/M_conformal_l2':'Whole shell M RMS','shell/all/Z_conformal_l2':'Whole shell Z RMS'}
y=np.arange(len(keys));fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for offset,field,label in [(-.18,'bulk_fractional_change','Bulk only: N24 → N32'),(.18,'shell_fractional_change','Shell only: A18/R123 → A24/R164')]:ax.barh(y+offset,[100*d[k][field] for k in keys],height=.34,label=label)
ax.set(yticks=y,yticklabels=list(keys.values()),xlabel='Constraint change relative to baseline (%) — negative means reduction',title='Native puncture at t=0.1: independent bulk and shell refinement\nSame equations, quintic Rb=0.98 and immutable Minkowski reference');ax.axvline(0,color='k',lw=.8);ax.grid(axis='x',alpha=.2);ax.invert_yaxis();ax.legend(fontsize=9);fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.pdf'))
