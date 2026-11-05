#!/usr/bin/env python3
"""Plot derivative growth and competing raw-tail fit residuals."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--boundary',type=Path,required=True);ap.add_argument('--tail',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 boundary=json.loads((a.boundary/'audit.json').read_text());fits=np.load(a.tail/'fit_residuals.npz');fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');rows=boundary['reports'];n=[r['n'] for r in rows]
 for f,name in [(0,'chi'),(5,'Khat'),(8,'lapse'),(9,'shift')]:axes[0].loglog(n,[abs(r['endpoint']['dd'][f]) for r in rows],'o-',label=name)
 axes[0].set(xlabel='Radial points',ylabel='Absolute endpoint second derivative',xticks=n);axes[0].xaxis.set_major_formatter(ScalarFormatter());axes[0].xaxis.set_minor_locator(NullLocator())
 for model in ('fractional','logarithmic','polynomial'):
  tag='n1024-outer0.015-exclude0-lapse-'+model;axes[1].semilogy(fits[tag+'-x'],np.maximum(abs(fits[tag+'-error']),1e-14),'o-',ms=3,label=model)
 axes[1].set(xlabel='Distance to scri (1-r)',ylabel='Absolute lapse-ratio fit residual',title='N1024, 0 < 1-r < .015')
 for ax in axes:ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'boundary-tail.png',dpi=160);plt.close(fig)
 inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.boundary/'audit.json',a.tail/'audit.json',a.tail/'fit_residuals.npz']};(a.output/'summary.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Growing second derivatives and finite-range fits motivate a regularity audit; they do not prove a fractional continuum exponent.'),indent=2)+'\n')
if __name__=='__main__':main()
