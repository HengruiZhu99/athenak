#!/usr/bin/env python3
"""RK4 bounds and nonnormal amplification on saved, scaled Jacobians."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import svdvals
from scipy.optimize import brentq
from audit_semidiscrete import product,exponential

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--linearization',nargs='+',type=Path,required=True);ap.add_argument('--targets',nargs='+',type=float,default=[.4,.5,.55,.6,.625,.65]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);results=[];inputs={}
 if any(not np.isfinite(c) or c<=0 for c in a.targets):raise ValueError('positive finite targets required')
 for folder in a.linearization:
  for name in ['audit.json','linearization.npz']:inputs[str(folder/name)]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
  info=json.loads((folder/'audit.json').read_text());data=np.load(folder/'linearization.npz');J=data['J'];values=data['eigenvalues'];speed=info['speed_over_dr'];decaying=values[values.real<-1e-8];I=np.eye(len(J));started=time.perf_counter()
  def polynomial(z):return 1+z+z*z/2+z**3/6+z**4/24
  def excess(c):return float(max(abs(polynomial(c/speed*decaying)))-1)
  bound=brentq(excess,.1,2.);rows=[]
  for c in a.targets:
   dt=c/speed;A=dt*J;A2=product(A,A);rk=I+A+A2/2+product(A2,A)/6+product(A2,A2)/24;exact=exponential(A)
   rows.append(dict(C_char=c,dt=dt,dt_M=3*dt,decaying_RK_spectral_radius=float(max(abs(polynomial(dt*decaying)))),full_RK_spectral_radius=float(max(abs(polynomial(dt*values)))),exact_spectral_radius=float(np.exp(dt*max(values.real))),RK_scaled_norm=float(svdvals(rk)[0]),exact_scaled_norm=float(svdvals(exact)[0]),RK_minus_exact_scaled_norm=float(svdvals(rk-exact)[0])))
  results.append(dict(linearization=str(folder),n=info['n'],max_real_eigenvalue=info['max_real_eigenvalue'],speed_over_dr=speed,decaying_RK4_C_char_bound=float(bound),rows=rows,seconds=time.perf_counter()-started));print(json.dumps(results[-1]),flush=True)
 out=dict(results=results,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Frozen scaled semidiscrete operators only. The decaying-eigenspace bound excludes positive real modes and does not control nonnormal transient amplification. Saved local scaling is not a continuum energy norm. Robust nonlinear accuracy margins require the separate evolved sweeps and perturbation controls.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for v in results:
  x=[r['C_char'] for r in v['rows']];label=f"N{v['n']}: "+Path(v['linearization']).name.replace('outer-lapse025-','').replace('-v26','');axes[0].plot(x,[r['decaying_RK_spectral_radius'] for r in v['rows']],'o-',label=label);axes[1].semilogy(x,[r['RK_scaled_norm'] for r in v['rows']],'o-',label=label+' RK4');axes[1].semilogy(x,[r['exact_scaled_norm'] for r in v['rows']],'--',label=label+' exact')
 axes[0].axhline(1,color='k',ls=':');axes[0].set(xlabel='Characteristic CFL',ylabel='Decaying-sector spectral radius');axes[1].set(xlabel='Characteristic CFL',ylabel='Saved-scaled operator 2-norm')
 for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.savefig(a.output/'rk-margin.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
