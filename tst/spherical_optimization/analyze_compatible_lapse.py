#!/usr/bin/env python3
"""Summarize compatible-lapse accuracy, boundary and pulse evidence."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullLocator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={}
 def read(name):
  p=a.evidence/name;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
 old=read('shift-normal-four-10M-v21/comparison.json')['runs'];new=read('compatible-lapse-four-10M-v23/comparison.json')['runs'];pulses=read('compatible-lapse-pulse-response-v23/comparison.json')['rows']
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for label,rows in [('Direct D2',old),('Compatible lapse',new)]:
  n=[r['n'] for r in rows]
  for ax,v in zip(axes,[[max(r['physical_rms']) for r in rows],[abs(r['auxiliary_probes'][-1]['defect']) for r in rows],[abs(r['clock']) for r in rows]]):ax.loglog(n,v,'o-',label=label)
 for ax,title in zip(axes,['Exterior constraint indicator','Differential D defect','Clock residual']):
  ax.set(xlabel='Radial points',ylabel=title,xlim=(56,600),xticks=[64,128,256,512]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_locator(NullLocator());ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'accuracy-boundary.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 groups=[[('original-gauge-N128','Direct D2'),('compatible-gauge-N128','Compatible 20M'),('compatible-gauge-N128-long','Compatible 40M')],[('original-gauge-N256','Direct D2'),('compatible-gauge-N256','Compatible')],[('compatible-constraint-N128','N128'),('compatible-constraint-N256','N256')]]
 for ax,group,title in zip(axes,groups,['Gauge pulse, N128','Gauge pulse, N256','Constraint pulse, compatible']):
  for label,display in group:
   rows=[r for r in pulses if r['label']==label];ax.semilogy([r['t_M'] for r in rows],[r['max_field_response'] for r in rows],'o-',ms=3,label=display)
  ax.set(xlabel='t/M',ylabel='Maximum exterior paired response',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'pulse-comparison.png',dpi=160);plt.close(fig)
 temporal=read('compatible-lapse-temporal-v23/convergence.json');transient=read('compatible-lapse-transient-temporal-v23/convergence.json');spatial=read('compatible-lapse-spatial-v23/convergence.json')
 out=dict(inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),accuracy=new,pulse_finals=[r for r in pulses if r['t_M'] in (20.,40.)],temporal_10M=temporal['adjacent_log2_norm_ratios'],temporal_1M=transient['adjacent_log2_norm_ratios'],spatial=spatial['adjacent_log2_norm_ratios'],qualification='Concurrent runs are not controlled timings; boundary convergence and robust timestep gates remain unresolved.')
 (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
