#!/usr/bin/env python3
"""Scientific pilot error figures; distinct normalizations and scopes stay explicit."""
from pathlib import Path
import json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();P=a.evidence
old=json.loads((P/'native-tensor-pilot-review.json').read_text());new=json.loads((P/'resolved-entry-v3-review.json').read_text());strict=json.loads((P/'resolved-entry-seedwidth-review.json').read_text());physical=json.loads((P/'old-grid-physical-length-review.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,report_old,report_new,title in [(axes[0],old,strict,'Original seed length L = .15 (failures retained)'),(axes[1],physical,new,'Continuum gradient length L = .012929 (v3)')]:
 for sector,color in [('plus','#14689c'),('cross','#c7522a')]:
  for quantity,marker in [('first','o'),('configuration_second','s')]:
   labels=['tt-'+sector+'-coarse4'+('-valid'if sector=='plus'else''),'tt-'+sector+'-fine4','tt-'+sector+'-resolved4-v3'];values=[]
   for i,label in enumerate(labels):
    c=(report_old if i<2 else report_new)['cases'][label];values.append(c['initial']['regions']['all']['errors']['fixed_reference_Penrose'][quantity]['rms'])
   ax.plot(range(3),values,marker=marker,color=color,linestyle='-'if quantity=='first'else'--',label=sector+' '+('first'if quantity=='first'else'config Hessian'))
 ax.axhline(.01,color='black',linewidth=.7,linestyle=':',label='RMS pilot budget');ax.set_yscale('log');ax.set_xticks(range(3),['12 / 8 / 16','18 / 12 / 24','36 / 16 / 64']);ax.set_xlabel('B / angular / radial (unequal refinements)');ax.set_ylabel('Initial normalized RMS error');ax.set_title(title,fontsize=10);ax.grid(alpha=.18)
axes[0].legend(fontsize=8);fig.suptitle('Native coupled TT initial derivatives: grid diagnosis, not convergence acceptance',fontsize=12);fig.savefig(P/'native-TT-entry-errors.png',dpi=180);plt.close(fig)
