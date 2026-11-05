#!/usr/bin/env python3
"""Compare existing standard and shift-normal boundary evidence at 10M."""
import argparse,json,hashlib
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={};old={}
 def read(name):
  p=a.evidence/name/'comparison.json';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())['runs']
 new=read('shift-normal-four-10M-v21')
 for row in read('shift-normal-comparison-10M-v21')+read('shift-normal-n512-comparison-v21'):
  if row['case'].startswith('chi-r2-'):old[row['n']]=row
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import NullLocator,ScalarFormatter
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for label,rows in [('standard',list(old.values())),('shift-normal',new)]:
  rows=sorted(rows,key=lambda r:r['n']);n=[r['n'] for r in rows]
  for ax,values in zip(axes,[[max(r['physical_rms']) for r in rows],[abs(r['auxiliary_probes'][-1]['defect']) for r in rows],[abs(r['clock']) for r in rows]]):ax.loglog(n,values,'o-',label=label)
 for ax,title in zip(axes,['Exterior constraint indicator','Differential D defect','Clock residual']):
  ax.set(xlabel='Radial points',ylabel=title,xticks=[64,128,256,512]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_locator(NullLocator());ax.grid(which='both',alpha=.25);ax.legend()
 fig.savefig(a.output/'boundary-progress.png',dpi=160);plt.close(fig)
 (a.output/'summary.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Matched characteristic-CFL .4 at 10M; default compiled equations unchanged. Exterior convergence does not establish boundary convergence; the two finest boundary residuals decrease slowly.'),indent=2)+'\n')
if __name__=='__main__':main()
