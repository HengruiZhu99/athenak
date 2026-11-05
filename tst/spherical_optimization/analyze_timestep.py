#!/usr/bin/env python3
"""Short-run RK4 accuracy and failure limits, with no robust admission claim."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_convergence import interpolate
from kernel import Kernel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--sweep',type=Path);ap.add_argument('--cases',nargs='+');ap.add_argument('--evidence',type=Path);ap.add_argument('--linearization',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 if a.cases:
  folders=[a.evidence/name for name in a.cases];runs=[]
  for folder in folders:
   cfg=json.loads((folder/'manifest.json').read_text())['config'];result=json.loads((folder/'result.json').read_text())
   initial=np.load(folder/'initial.npz');dr=np.r_[np.diff(initial['r']),initial['r'][-1]-initial['r'][-2]]
   kk=Kernel.from_config(a.library,cfg);_,_,_,diag=kk.parts(initial['r'],initial['q'],np.zeros_like(initial['q']),np.zeros_like(initial['q']),initial['source']);kk.close()
   runs.append(dict(C_char_target=cfg['fixed_dt']*float(max(diag[:,7]/dr)),result=result))
 else:
  sweep=json.loads((a.sweep/'sweep.json').read_text());runs=sweep['runs'];folders=[a.sweep/f"C{x['C_char_target']:.3f}" for x in runs]
 cfg0=json.loads((folders[0]/'manifest.json').read_text())['config'];duration=cfg0['duration'];k=Kernel.from_config(a.library,cfg0);R=np.geomspace(25,100,301)
 def radius(r):
  q,_,om=k.initial(np.array([r]));return 3*r/(om[0,0]*np.sqrt(q[0,0]))
 target=np.array([brentq(lambda r:radius(r)-x,k.config[2],.99999) for x in R]);k.close()
 ref=np.load(folders[0]/'last-accepted.npz');rows=[]
 assert runs[0]['result']['status']=='completed_unqualified'
 for run,folder in zip(runs,folders):
  cfg=json.loads((folder/'manifest.json').read_text())['config']
  assert set(cfg)==set(cfg0)
  assert all(cfg[key]==cfg0[key] for key in cfg if key not in ('output','fixed_dt')), 'timestep cases differ in formulation or diagnostics'
  result=run['result'];row=dict(target=run['C_char_target'],status=result['status'],time_M=result['time_M'],seconds=result['stepping_seconds'],actual_C_char=result['C_char_max_accepted'],failure=result['failure'])
  row['case']=folder.name;history=json.loads((folder/'history.json').read_text())
  if history:row.update(final_recorded_pole=history[-1]['pole'],final_recorded_clock=history[-1]['clock'])
  if result['status']=='completed_unqualified':
   q=np.load(folder/'last-accepted.npz')['q'];delta=interpolate(ref['r'],q-ref['q'],target,7);row['field_difference_rms']=np.sqrt(np.mean(delta**2,axis=0)).tolist();row['max_field_difference_rms']=float(max(row['field_difference_rms']))
  rows.append(row)
 linear=np.load(a.linearization/'linearization.npz');values=linear['eigenvalues'];decaying=values[values.real<-1e-8];linear_info=json.loads((a.linearization/'audit.json').read_text());speed=linear_info['speed_over_dr']
 def amplitude(cfl):
  z=cfl/speed*decaying;return max(abs(1+z+z*z/2+z**3/6+z**4/24))-1
 threshold=brentq(amplitude,.1,2)
 out=dict(duration_M=duration,n=cfg0['n'],linearization=str(a.linearization),linearization_n=linear_info['n'],same_resolution=cfg0['n']==linear_info['n'],runs=rows,decaying_eigenvalue_RK4_C_char_bound=float(threshold),qualification='Frozen decaying-eigenspace absolute-stability bound and unperturbed nonlinear runs only; positive growth modes, nonnormal transients, boundary incompatibility and perturbation accuracy prevent robust timestep admission.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for row in rows:
  if 'max_field_difference_rms' in row:axes[0].semilogy(row['target'],max(row['max_field_difference_rms'],1e-15),'o',color='C0')
  axes[1].plot(row['target'],row['time_M'],'o',color='C0' if row['status']=='completed_unqualified' else 'C3')
 for ax in axes:ax.axvline(threshold,color='k',ls='--',label=f"N{linear_info['n']} frozen decaying-spectrum RK4 bound");ax.grid(alpha=.25);ax.set_xlabel('Initial characteristic CFL target')
 axes[0].set(ylabel=f"Exterior field RMS difference from C={rows[0]['target']:.3g}",title=f'Temporal differences at {duration:g}M')
 axes[1].set(ylabel='Last accepted time / M',title='Completion is not robust stability');axes[1].legend(fontsize=7)
 fig.savefig(a.output/'timestep.png',dpi=160);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
