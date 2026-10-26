#!/usr/bin/env python3
"""Plot preserved read-only operator evidence, with its scientific limit."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();rhs=E/'cpu-schwarzschild-commensurate-finer/actual-resolved-rhs-review.json';memory=E/'aurora/cpu-memory-rhs-audit-v1-collected/result/memory-audit-receipt.json';r=json.loads(rhs.read_text());m=json.loads(memory.read_text());stem=E/'frozen-rhs-and-memory-audit-v22-v3'
if stem.with_suffix('.png').exists():raise RuntimeError('Existing plot is immutable')
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140});fig,axes=plt.subplots(1,3,figsize=(12,4.2));ns=r['resolutions']
for ax,norm in zip(axes[:2],['L2','Linf']):
 for field,label,marker in [('RHS','Actual compiled psi_t','o'),('reconstruction','Reconstructed velocity','s')]:
  c=next(c for c in r['checks'] if c['time']==100 and c['field']==field and c['norm']==norm);ax.loglog(ns,c['errors'],marker=marker,label=label);ax.text(.05,.06 if field=='RHS' else .15,f"{label}: p = "+', '.join(f'{x:.2f}' for x in c['orders']),transform=ax.transAxes,fontsize=8)
 ax.set_xticks(ns,labels=ns);ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel('Resolution N');ax.set_title('t = 100 M, '+norm);ax.set_ylabel('Independent-reference error');ax.legend(fontsize=8,loc='upper right')
records=m['records'];fill=[1000*x['metadata']['fill_seconds_per_call'] for x in records];evaluate=[1000*x['metadata']['evaluate_seconds_per_call'] for x in records];axes[2].bar(['Default DDR','Explicit HBM'],fill,label='Interpatch fill');axes[2].bar(['Default DDR','Explicit HBM'],evaluate,bottom=fill,label='Scalar RHS');axes[2].set_ylabel('Milliseconds per call');axes[2].set_title('N60; 200 calls per placement\nIdentical RHS; scalar state and geometry fixed',fontsize=10);axes[2].legend(fontsize=8,loc='upper left');axes[2].set_ylim(0,80)
for i,(a,b) in enumerate(zip(fill,evaluate)):axes[2].text(i,a+b+1,f'{a+b:.1f} ms',ha='center',fontsize=9)
fig.suptitle('Frozen scalar operator audit',fontsize=14);fig.text(.5,.015,'No time stepping. Schwarzschild stage remains unadmitted: late psi L2 convergence still fails.',ha='center',fontsize=10,color='#9c2d25');fig.tight_layout(rect=[0,.055,1,.94])
for suffix in ['.png','.pdf']:fig.savefig(stem.with_suffix(suffix))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();stem.with_suffix('.json').write_text(json.dumps(dict(inputs={str(f):sha(f) for f in [rhs,memory]},source_sha256=sha(Path(__file__)),outputs={suffix:sha(stem.with_suffix(suffix)) for suffix in ['.png','.pdf']},admission=False),indent=2)+'\n');print(stem.with_suffix('.png'))
