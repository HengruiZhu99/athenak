#!/usr/bin/env python3
"""Plot retained finite-grid mode diagnoses with source/data hash provenance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',action='append',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
records=[]
for folder in a.input:records.extend(json.loads((folder/'review.json').read_text())['records'])
fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
for method,bias,label in [('quadrature','centered','Quadrature'),('discrete-ode','centered','Centered ODE (failed)'),('discrete-ode','right','Biased ODE candidate')]:
 rows=sorted([r for r in records if r['method']==method and r['bias']==bias and r['epsilon']==1e-7],key=lambda r:r['n'])
 axes[0].plot([r['n'] for r in rows],[r['dominant_real'] for r in rows],'o-',label=label)
axes[0].set_xscale('log',base=2);axes[0].set_yscale('symlog',linthresh=1);axes[0].axhline(0,color='k',lw=.7);axes[0].set_xlabel('N');axes[0].set_ylabel('Largest real eigenvalue');axes[0].legend(fontsize=8)
p=a.input[-1]/'n512-discrete-ode-centered-eps1e-07.npz'
with np.load(p) as z:r=z['r'];mode=z['dominant_mode'];amplitude=np.sqrt(np.sum(abs(mode)**2,axis=1))
axes[1].plot(r,amplitude/max(amplitude));axes[1].set_xlim(.9,1);axes[1].set_xlabel('r');axes[1].set_ylabel('Normalized mode magnitude');axes[1].set_title('Centered ODE: N512 unstable mode',fontsize=10)
for ax in axes:ax.grid(alpha=.25)
fig.savefig(a.output/'modes.png',dpi=160);fig.savefig(a.output/'modes.pdf');plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(a.output/'provenance.json').write_text(json.dumps(dict(scope='Discrete diagnostic, not a continuum stability theorem',source_sha256=sha(Path(__file__)),inputs={str(p.resolve()):sha(p) for p in [*[f/'review.json' for f in a.input],p]}),indent=2)+'\n')
