#!/usr/bin/env python3
"""Plot the saved smooth-data propagation audit without rerunning kernels."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audits',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];inputs={}
 for folder in a.audits:
  p=folder/'audit.json';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();rows.extend(json.loads(p.read_text())['rows'])
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');scale=np.array([1/27,1/27,1/9,1/9])
 for family,ax in zip(['areal','compact'],axes):
  cases=[v for v in rows if v['family']==family]
  for R in sorted({v['reference_R_M'] for v in cases}):
   v=sorted([v for v in cases if v['perturbed_gauge'] and v['reference_R_M']==R],key=lambda x:x['derivative_step'])
   ax.loglog([x['derivative_step'] for x in v],[max(abs(np.array(x['continuum_constraint_rate'])*scale)) for x in v],'o-',label=f'perturbed gauge, R/M={R:g}')
  steps=sorted({v['derivative_step'] for v in cases});floor=[max(max(abs(np.array(v['continuum_constraint_rate'])*scale)) for v in cases if not v['perturbed_gauge'] and v['derivative_step']==h) for h in steps]
  ax.loglog(steps,floor,'k--',label='stationary maximum over radii')
  ax.set(xlabel='RHS differentiation spacing',ylabel='Maximum dimensionless constraint rate',title=family+' slicing');ax.grid(which='both',alpha=.25);ax.legend(fontsize=8)
 fig.savefig(a.output/'constraint-rate.png',dpi=160);plt.close(fig)
 out=dict(inputs=inputs,scaling='M=1/3; derivatives of (M²H,M²Momentum,MTheta,MZ) with respect to t/M',max_initial_constraint=float(max(max(v['initial_constraint_max']) for v in rows)),max_stationary_center_RHS=float(max(max(abs(np.array(v['RHS_at_center']))) for v in rows if not v['perturbed_gauge'])),max_finest_dimensionless_rate=float(max(max(abs(np.array(v['continuum_constraint_rate'])*scale)) for v in rows if v['derivative_step']==.000125)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Local constraint preservation on the sampled analytic geometry/gauge families; finite RHS differentiation reaches a roundoff floor. No claim about arbitrary data or evolved boundary regularity.')
 (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
