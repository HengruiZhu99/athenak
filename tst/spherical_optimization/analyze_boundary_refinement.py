#!/usr/bin/env python3
"""Compare boundary and exterior diagnostics across explicitly grouped policies."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--comparison',nargs=2,action='append',required=True,metavar=('LABEL','JSON'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);groups={};inputs={}
 for label,filename in a.comparison:
  p=Path(filename);inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();groups.setdefault(label,[]).extend(json.loads(p.read_text())['runs'])
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');axes=axes.ravel();summary={}
 for label,rows in groups.items():
  rows=sorted(rows,key=lambda r:r['n']);n=np.array([r['n'] for r in rows]);assert len(set(n))==len(n);metrics=np.array([[max(r['physical_rms']),r['clock'],r['auxiliary_probes'][-1]['defect'],r['pole']] for r in rows]);orders=np.log(abs(metrics[:-1]/metrics[1:]))/np.log((n[1:]-.5)/(n[:-1]-.5))[:,None]
  summary[label]=dict(n=n.tolist(),metrics=metrics.tolist(),adjacent_orders=orders.tolist(),fields=['exterior','clock','D_defect','pole'])
  for j,ax in enumerate(axes):ax.loglog(n,abs(metrics[:,j]),'o-',label=label)
 for ax,name in zip(axes,['Exterior constraint indicator','Clock residual','Differential D defect','Maximum pole residual']):
  ax.set(xlabel='Radial points',ylabel='Absolute '+name.lower(),xticks=[64,128,256,512,1024],xlim=(56,1200));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_locator(NullLocator());ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'boundary-refinement.png',dpi=160);plt.close(fig)
 out=dict(groups=summary,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Measured adjacent ratios of absolute residuals; sign changes and a few levels do not establish an asymptotic power law. These are nonlinear evolved diagnostics, not endpoint well-posedness or 3D stability proof.');(a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
