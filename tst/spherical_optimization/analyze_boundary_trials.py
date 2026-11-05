#!/usr/bin/env python3
"""Reproduce damping response, boundary balance and rejected outflow evidence."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];inputs={}
 def load(path):
  inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  return json.loads(path.read_text()) if path.suffix=='.json' else np.load(path)
 for damping in (1,2,4,8):
  names=[f'chi-r2-{kind}-20M-c0.5-v16' if damping==1 else f'damping{damping}-{kind}-20M-v18' for kind in ('none','gauge')]
  b,p=[a.evidence/x for x in names];cfg=load(b/'manifest.json')['config'];pcfg=load(p/'manifest.json')['config']
  assert cfg['fixed_dt']==pcfg['fixed_dt'] and cfg.get('auxiliary_damping',1)==pcfg.get('auxiliary_damping',1)==damping
  for path in sorted(b.glob('profile-*.npz')):
   bs=load(path);ps=load(p/path.name);assert bs['t_M']==ps['t_M'] and np.array_equal(bs['r'],ps['r'])
   delta=(ps['q']-bs['q'])[bs['r']>.8]
   rows.append(dict(damping=damping,t_M=float(bs['t_M']),max_field_response=float(abs(delta).max()),max_lapse_shift_response=float(abs(delta[:,8:10]).max())))
 balance=load(a.evidence/'boundary-balance-v18/balance.json')['runs'];spectra=[]
 for n,width in [(32,3),(32,5),(32,7),(64,5)]:
  name=f'outflow-w{width}-spectrum-v18' if n==32 else f'outflow-w{width}-n{n}-spectrum-v18'
  s=load(a.evidence/name/'audit.json');v=load(a.evidence/name/'linearization.npz');mode=abs(v['leading_mode']);index=np.unravel_index(mode.argmax(),mode.shape)
  spectra.append(dict(n=n,width=width,max_real_eigenvalue=s['max_real_eigenvalue'],relative_jacobian_difference=s['relative_jacobian_difference'],peak_r=float(v['r'][index[0]]),peak_independent_field=int(index[1]),eigenvalue_times_endpoint_dr=float(s['max_real_eigenvalue']*(v['r'][-1]-v['r'][-2]))))
 out=dict(response=rows,rejected_outflow=spectra,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Paired raw field differences on r>.8, not an energy norm. Damping1 C=.5 versus other C=.4: temporal control is in chi-r2-long-pulses-v16; these are not controlled timings. Positive real outflow eigenvalues occur also in exact semidiscrete flow, and scale with inverse spacing.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for damping in (1,2,4,8):
  v=[x for x in rows if x['damping']==damping]
  axes[0].semilogy([x['t_M'] for x in v],[x['max_field_response'] for x in v],'o-',label=f'damping {damping}')
 axes[0].set(xlabel='t/M',ylabel='Maximum paired exterior field response');axes[0].legend()
 n=[x['n'] for x in balance]
 for key,label in [('D_minus_Lambda_advection','beta (D−Lambda) derivative'),('D_damping','−damping D')]:axes[1].plot(n,[x['bulk_D_terms'][key] for x in balance],'o-',label=label)
 axes[1].plot(n,[x['trace_time_derivative_probes'][-1] for x in balance],'o-',label='dependent D time derivative')
 axes[1].set(xlabel='Radial points',ylabel='Boundary rate contribution');axes[1].legend(fontsize=8)
 for width in (3,5,7):
  v=[x for x in spectra if x['width']==width];axes[2].plot([x['n'] for x in v],[x['max_real_eigenvalue'] for x in v],'o-',label=f'extrapolation width {width}')
 axes[2].set(xlabel='Radial points',ylabel='Rejected mode Re(lambda)');axes[2].legend(fontsize=8)
 for ax in axes:ax.grid(alpha=.25)
 fig.savefig(a.output/'boundary-trials.png',dpi=160);plt.close(fig)
 print(json.dumps(dict(final_responses=[x for x in rows if x['t_M']>19.99],rejected_outflow=spectra)))
if __name__=='__main__':main()
