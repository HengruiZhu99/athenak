#!/usr/bin/env python3
"""Plot saved continuum leading-power and evolved-polarization diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={}
 def read(name):
  p=a.evidence/name/'audit.json';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
 balance=read('scri-indicial-balance-v25');mode=read('scri-indicial-mode-v25');s=balance['default_positive_power'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for name,label in [('scri-indicial-native-v25','Native outer jets'),('scri-indicial-outermetric-v25','Evolved metric value')]:
  rows=read(name)['rows'];roots=[min(r['finite_roots'],key=lambda x:abs(x['real']-s)+abs(x['imag']))['real'] for r in rows];axes[0,0].semilogx([r['distance'] for r in rows],roots,'o-',label=label)
 axes[0,0].axhline(s,color='k',ls='--',label='Limiting positive root');axes[0,0].set(xlabel='Distance to scri',ylabel='Weighted power s');axes[0,0].legend(fontsize=8)
 rows=mode['direct_rhs'];axes[0,1].loglog([r['distance'] for r in rows],[r['scaled_RHS_norm'] for r in rows],'o-',label='Direct scaled RHS')
 rows=balance['constraint_checks'];axes[0,1].loglog([r['distance'] for r in rows],[np.linalg.norm(r['leading_scaled_H_M_Theta_Z']) for r in rows],'s-',label='Scaled physical constraints');axes[0,1].set(xlabel='Distance to scri',ylabel='Residual norm');axes[0,1].legend(fontsize=8)
 xi=np.linspace(.1,1.2,200);power=1+(-7+np.sqrt(41+32/xi))/2;axes[1,0].plot(xi,power,label='Leading-balance prediction');axes[1,0].plot([1,.25],[1+s,4],'o');axes[1,0].axhline(2,color='k',ls='--',label='Field power 2');axes[1,0].set(xlabel='Outer lapse damping xi',ylabel='Predicted metric/lapse power 1+s');axes[1,0].legend(fontsize=8)
 fit=next(r for r in mode['evolved_polarization'] if r['outer']==.03 and r['excluded_cells']==2);x=np.arange(9);axes[1,1].bar(x-.18,mode['predicted_outer_metric_mode'],.36,label='Continuum leading mode');axes[1,1].bar(x+.18,fit['normalized'],.36,label='N1024 finite-window fit');axes[1,1].set(xticks=x,xticklabels=balance['chart_field_order'],ylabel='Amplitude / lapse amplitude');axes[1,1].legend(fontsize=8)
 for ax in axes.flat:ax.grid(alpha=.2)
 fig.savefig(a.output/'indicial-mode.png',dpi=160);plt.close(fig)
 (a.output/'summary.json').write_text(json.dumps(dict(inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Parameter curve is an untested leading continuum prediction. Polarization fit retains the discrepancy of curvature/connection fields and does not establish a unique nonlinear mode.'),indent=2)+'\n')
if __name__=='__main__':main()
