#!/usr/bin/env python3
"""Plot a closed early snapshot and its resolved-reference uncertainty."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();R=E/'cpu-schwarzschild-dyadic';old=R/'early-t2-review.json';fine=R/'early-t2-reference4800-review.json';x=json.loads(old.read_text());y=json.loads(fine.read_text());stem=E/'schwarzschild-dyadic-early-reference-v22'
if stem.with_suffix('.png').exists():raise RuntimeError('Existing early plot is immutable')
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');labels=[c['field']+' '+c['norm'] for c in y['checks']];indices=np.arange(len(labels))
for k,label in enumerate(['N24/N48','N48/N96']):axes[0].bar(indices+(k-.5)*.35,[c['orders'][k] for c in y['checks']],.35,label=label)
axes[0].axhline(1.5,color='black',ls='--',lw=1,label='Required1.5');axes[0].set_xticks(indices,labels,rotation=35,ha='right');axes[0].set(ylabel='Measured pair order',title='Closed t = 2M snapshot; 4800-point oracle');axes[0].legend(fontsize=8)
labels=[c['field']+' '+c['norm'] for c in y['reference_uncertainty']];indices=np.arange(len(labels))
for k,(r,label,color) in enumerate([(x,'1200/2400 reference difference','#c44e52'),(y,'2400/4800 reference difference','#22856b')]):axes[1].bar(indices+(k-.5)*.35,[100*c['fraction'] for c in r['reference_uncertainty']],.35,label=label,color=color)
axes[1].axhline(10,color='black',ls='--',lw=1,label='Required10%');axes[1].set_xticks(indices,labels,rotation=35,ha='right');axes[1].set(ylabel='Reference difference / primary error (%)',title='Independent uncertainty at the actual N96 points');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('Early dyadic scalar evidence: no stage admission\nN96 to100M and timestep/boundary controls remain outstanding',fontsize=12)
for suffix in ['.png','.pdf']:fig.savefig(stem.with_suffix(suffix),dpi=170)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();stem.with_suffix('.json').write_text(json.dumps(dict(admission=False,inputs={str(f):sha(f) for f in [old,fine]},source_sha256=sha(Path(__file__)),outputs={suffix:sha(stem.with_suffix(suffix)) for suffix in ['.png','.pdf']}),indent=2)+'\n');print(stem.with_suffix('.png'))
