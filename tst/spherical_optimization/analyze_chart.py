#!/usr/bin/env python3
"""Physical-geometry reanalysis and scri differential-auxiliary probes."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel import Kernel
from evolution import ChartEvolution,DerivativeBoundaryEvolution
from physical_constraints import evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+');ap.add_argument('--library',required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);rows=[];common=np.geomspace(25,100,301)
 cases=a.cases or ['chart-central20-n128-1M-v4','chart-analytic-n128-1M-v4']+[f'chart-reference-n{n}-1M-v6' for n in (64,128,256)]
 duration=None
 for name in cases:
  folder=a.evidence/name;result=json.loads((folder/'result.json').read_text());cfg=json.loads((folder/'manifest.json').read_text())['config']
  if duration is None:duration=cfg['duration']
  assert cfg['duration']==duration
  if result['status']!='completed_unqualified':rows.append(dict(case=name,status=result['status']));continue
  initial=np.load(folder/'initial.npz');last=np.load(sorted(folder.glob('profile-*.npz'))[-1]);r=initial['r'];q=last['q'];om=initial['omega']
  phys=evaluate(r,q,last['d'],last['dd'],om);phys[:,:2]/=9;phys[:,2:]/=3
  mask=(r[:-1]>initial['configuration'][3]/6);R=3*r[:-1][mask]/(om[:-1,0][mask]*np.sqrt(initial['q'][:-1,0][mask]))
  sample=PchipInterpolator(R,phys[mask])(common);linear=np.column_stack([np.interp(common,R,phys[mask,f]) for f in range(4)])
  rms=np.sqrt(np.mean(sample**2,axis=0));interp=np.sqrt(np.mean((sample-linear)**2,axis=0))
  k=Kernel(a.library,cfg['mode'],cfg['radius'],cfg['slicing_ratio'],cfg['separated'])
  e=(DerivativeBoundaryEvolution if cfg.get('derivative_boundary',False) else ChartEvolution)(k,cfg['n'],cfg['grading'],cfg['penalty'],cfg['ko'],cfg['floors'],cfg.get('normal_policy','sampled'),cfg.get('reference_jets',False),cfg.get('reference_step',5e-4),cfg.get('reference_tail','numerical'),cfg.get('grid_map','quadratic'))
  state=q[:,e.independent];rate=e.rhs(state);base=e.last.copy();probe=[]
  for scale in (1e-5,3e-6,1e-6):
   eps=scale/max(1.,np.max(abs(rate)));e.rhs(state+eps*rate);plus=e.last['dependent_D'];e.rhs(state-eps*rate);minus=e.last['dependent_D']
   trace=(plus-minus)/(2*eps);probe.append(dict(epsilon=float(eps),trace_rate=float(trace),pde_rate=float(base['auxiliary_trace_pde']),defect=float(trace-base['auxiliary_trace_pde'])))
  row=dict(case=name,binary_sha256=json.loads((folder/'manifest.json').read_text())['binary_sha256'],n=cfg['n'],seconds=result['stepping_seconds'],physical_rms=rms.tolist(),interpolation_rms=interp.tolist(),auxiliary_probes=probe,
           clock=float(base['clock']),pole=float(max(abs(base['pole']))),trace=float(max(abs(base['trace']))),D_mismatch=float(base['dependent_D']-base['q'][-1,11]),reference_step=cfg.get('reference_step',5e-4),scri_tangency=float(max(abs(base['tangency'][-1]))))
  rows.append(row);k.close()
 timing_same_binary=len({x['binary_sha256'] for x in rows if 'binary_sha256' in x})<=1
 (a.output/'comparison.json').write_text(json.dumps(dict(domain=f'Reference R/M 25–100, uniform log-R RMS, t={duration:g}M',runs=rows,timing_same_binary=timing_same_binary,qualification='No robust or converged candidate admitted.'),indent=2)+'\n')
 fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for row in rows:
  if 'physical_rms' not in row:continue
  label=row['case'].replace('chart-','').replace('-1M-v6','').replace('-1M-v4','').replace('-1M-v8','')
  ax[0].loglog(row['seconds'],max(row['physical_rms']),'o',label=label)
  ax[1].loglog(row['n'],abs(row['auxiliary_probes'][-1]['defect']),'o',label=label)
 ax[0].set(xlabel=f'Measured {duration:g}M CPU seconds',ylabel='Max physical constraint RMS (dimensionless)',title='Common reference exterior' if timing_same_binary else 'Mixed binaries: do not rank runtimes')
 ax[1].set(xlabel='Radial points',ylabel='Differential D trace defect',title='Boundary compatibility remains a gate')
 ax[0].legend(fontsize=7)
 for axes in ax:axes.grid(True,which='both',alpha=.25)
 fig.savefig(a.output/'chart-comparison.png',dpi=160);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
