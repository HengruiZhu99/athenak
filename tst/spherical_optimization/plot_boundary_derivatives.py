#!/usr/bin/env python3
"""Compare signed endpoint second derivatives from independent boundary audits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',nargs=2,action='append',required=True,metavar=('LABEL','JSON'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={};groups={};fields=[0,5,8]
 fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for label,filename in a.audit:
  p=Path(filename);inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();rows=sorted(json.loads(p.read_text())['reports'],key=lambda r:r['n']);n=np.array([r['n'] for r in rows]);assert len(set(n))==len(n)
  dd=np.array([[r['endpoint']['dd'][i] for i in fields] for r in rows]);groups[label]=dict(n=n.tolist(),second_derivatives=dd.tolist(),adjacent_differences=np.diff(dd,axis=0).tolist())
  for j,ax in enumerate(axes):ax.plot(n,dd[:,j],'o-',label=label)
 for ax,name in zip(axes,['chi','Khat','conformal lapse']):
  ax.set(xscale='log',yscale='symlog',xlabel='Radial points',ylabel='Endpoint radial second derivative',title=name,xticks=[128,256,512,1024]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_locator(NullLocator());ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'endpoint-second-derivatives.png',dpi=160);plt.close(fig)
 out=dict(fields=['chi','Khat','alpha'],groups=groups,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Signed endpoint derivatives on evolved total fields, from the fixed numerical policy. Decreasing differences support convergence of derivatives required by the PDE; a few levels do not prove analyticity or exclude higher-order logarithmic terms.')
 (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups))
if __name__=='__main__':main()
