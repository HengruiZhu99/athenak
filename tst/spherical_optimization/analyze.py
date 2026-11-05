#!/usr/bin/env python3
"""Reproduce provisional radial accuracy/cost plots from immutable run artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);Rcommon=np.geomspace(25,100,301);rows=[];fields={}
 for p in sorted(a.evidence.glob('*1M-v2')):
  if not (p/'result.json').exists():continue
  result=json.loads((p/'result.json').read_text());manifest=json.loads((p/'manifest.json').read_text());config=manifest['config']
  if result['status']!='completed_unqualified':rows.append(dict(case=p.name,status=result['status']));continue
  initial=np.load(p/'initial.npz');r=initial['r'];q0=initial['q'];omega=initial['omega'][:,0];k=initial['configuration'][3]
  mask=(r>k/6)&(r<1);R=3*r[mask]/(omega[mask]*np.sqrt(q0[mask,0]))
  last=np.load(sorted(p.glob('profile-*.npz'))[-1]);q=last['q'];diag=last['diag'];chi=q[:,0];gr=q[:,1]
  # Physical norm of the radial momentum covector, plus dimensionless M factors.
  physical=np.sqrt(chi/gr)*omega
  metrics=np.column_stack((diag[:,0]/9,diag[:,1]*physical/9,diag[:,2]/3,diag[:,3]*physical/3))
  samples=PchipInterpolator(R,metrics[mask])(Rcommon)
  linear=np.column_stack([np.interp(Rcommon,R,metrics[mask,f]) for f in range(4)])
  rms=np.sqrt(np.mean(samples**2,axis=0));interpolation=np.sqrt(np.mean((samples-linear)**2,axis=0))
  fields[p.name]=PchipInterpolator(R,q[mask])(Rcommon)
  row=dict(case=p.name,status=result['status'],n=config['n'],slicing_ratio=config['slicing_ratio'],separated=config['separated'],time_M=result['time_M'],
           cpu_seconds=result['stepping_seconds'],rhs_calls=result['rhs_calls'],dimensionless_H_M_Theta_Z_rms=rms.tolist(),
           pchip_linear_disagreement_rms=interpolation.tolist(),error_indicator=float(max(rms)),interpolation_indicator=float(max(interpolation)),
           C_char=result['C_char_max_accepted'],floor_calls=result['floor_calls'])
  rows.append(row)
 comparisons={}
 names=[f'central20-separated-n{n}-1M-v2' for n in (64,128,256)]
 if all(n in fields for n in names):
  d1=np.sqrt(np.mean((fields[names[0]]-fields[names[1]])**2,axis=0));d2=np.sqrt(np.mean((fields[names[1]]-fields[names[2]])**2,axis=0))
  comparisons=dict(fields=['chi','gr','gt','Ar','At','Khat','Theta','Lambda','alpha','beta','B','D'],difference_64_128=d1.tolist(),difference_128_256=d2.tolist(),
                   pairwise_order=np.log2(d1/np.maximum(d2,1e-300)).tolist(),qualification='One triplet only, interpolation uncertainty not removed; not asymptotic convergence.')
 summary=dict(domain='R_reference/M in [25,100], uniform log-R RMS; initial outer branch',duration_M=1,
  qualification='Provisional 1M evidence only. No robust-stability or fixed-accuracy 10M admission.',runs=rows,field_comparison=comparisons)
 (a.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for row in rows:
  if row['status']!='completed_unqualified':continue
  label=f"N={row['n']}, slice×{row['slicing_ratio']:g}"+(' separated' if row['separated'] else ' constant')
  marker='s' if row['slicing_ratio']!=2 else ('o' if row['separated'] else 'x')
  axes[0].loglog(row['cpu_seconds'],row['error_indicator'],marker,label=label)
  axes[1].loglog(row['n'],row['dimensionless_H_M_Theta_Z_rms'][0],marker,label=label)
 axes[0].set(xlabel='Measured 1M CPU stepping time (s)',ylabel='Maximum dimensionless constraint RMS',title='Provisional accuracy–cost comparison')
 axes[1].set(xlabel='Radial points',ylabel='M² Hamiltonian RMS',title='Fixed reference exterior, t=1M')
 for ax in axes:ax.grid(True,which='both',alpha=.25)
 axes[0].legend(fontsize=7);fig.suptitle('Spherical pilots: completion is not stability or convergence',fontsize=11)
 fig.savefig(a.output/'accuracy-cost.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,layout='constrained')
 for p in sorted(a.evidence.glob('*1M-v2')):
  if not (p/'history.json').exists():continue
  history=json.loads((p/'history.json').read_text());t=[v['t_M'] for v in history]
  axes[0].semilogy(t,[max(v['exterior_H'],1e-16) for v in history],label=p.name)
  axes[1].semilogy(t,[max(v['clock'],1e-16) for v in history])
 axes[0].set(ylabel='H RMS, compact r=.65–.995',title='Preserved transients and boundary residuals')
 axes[1].set(xlabel='t/M',ylabel='Unprojected scri clock residual')
 axes[0].legend(fontsize=6)
 for ax in axes:ax.grid(True,alpha=.25)
 fig.savefig(a.output/'transients.png',dpi=160);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
