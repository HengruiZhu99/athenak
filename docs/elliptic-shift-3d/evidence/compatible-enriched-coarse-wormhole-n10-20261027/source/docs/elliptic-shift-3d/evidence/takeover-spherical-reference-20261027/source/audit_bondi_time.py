#!/usr/bin/env python3
"""Replay scalar snapshots with their exact frozen RHS to audit time affinity.

The news and metric rate come from the preserved numerical source. This is
sampled postprocessing, not an all-stage maximum or an evolution restart.
An affine reparameterization alone does not repair a failed Einstein boundary
phase or establish a Bondi mass-loss law.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bondi import nonaffinity

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.run=a.run.resolve()
a.output.mkdir(parents=True,exist_ok=False)
m=json.loads((a.run/'manifest.json').read_text());frozen=a.run/'frozen-source'
if not frozen.is_dir():raise ValueError('Exact frozen numerical source is required')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
for name,digest in m['binding']['source_sha256'].items():
    if sha(frozen/name)!=digest:raise ValueError('Frozen source mismatch: '+name)
if not m['arguments']['scalar'] or m['arguments']['backend']!='spherical-g1':raise ValueError('This audit requires the coupled g=1 scalar experiment')
sys.path.insert(0,str(frozen/'tst/elliptic_shift_1d'))
from evolve import CoupledEvolution
args=SimpleNamespace(**m['arguments']);library=frozen/'runtime'/Path(args.library).name
if sha(library)!=m['binding']['executable_sha256']:raise ValueError('Frozen native library mismatch')
args.library=str(library);ev=CoupledEvolution(args);rows=[]
for profile in sorted(a.run.glob('profile-*.npz')):
    with np.load(profile,allow_pickle=False) as z:q=z['q'].copy();t=float(z['t'])
    rate=ev.rhs(q);geom=ev.geometry_state;d=ev.geometry_first
    chi=1+geom[-1,0];alpha=1+geom[-1,6]
    beta=-1+ev.shift_deviation[-1];db=beta+1.5*alpha*geom[-1,2]
    kappa=nonaffinity(chi,alpha,d[-1,0],1+d[-1,6],db,rate[-1,0])
    rows.append(dict(t=t,kappa=float(kappa),f=float(q[-1,8]),f_t=float(rate[-1,8]),
        chi=float(chi),chi_t=float(rate[-1,0]),profile_sha256=sha(profile)))
def integral(values,t):return np.r_[0.,np.cumsum((values[:-1]+values[1:])*np.diff(t)/2)]
t=np.array([row['t'] for row in rows]);kap=np.array([row['kappa'] for row in rows])
news=np.array([row['f_t'] for row in rows]);lograte=integral(kap,t);affine_rate=np.exp(lograte)
affine_time=integral(affine_rate,t);energy=integral(4*np.pi*news*news,t)
reparameterized=integral(4*np.pi*news*news/affine_rate,t)
coarse_lograte=integral(kap[::2],t[::2])
common_last=2*((len(t)-1)//2)
record=dict(run=str(a.run),binding=m['binding'],analysis_sha256=sha(__file__),
    frame='g_B=chi*g_bar; Omega_B=sqrt(chi)*Omega; unit-sphere cuts; alpha(scri)=1',
    max_sampled_abs_nonaffinity=float(max(abs(kap))),
    max_sampled_abs_integrated_nonaffinity=float(max(abs(lograte))),
    final_affine_parameter=float(affine_time[-1]),coordinate_duration=float(t[-1]-t[0]),
    final_affine_rate=float(affine_rate[-1]),sampled_coordinate_flux_integral=float(energy[-1]),
    sampled_affine_reparameterized_flux_integral=float(reparameterized[-1]),
    stored_RK_coordinate_flux_integral=m.get('radiated_energy'),
    cadence_coarsening_lograte_change=float(abs(lograte[common_last]-coarse_lograte[-1])),
    scope='Sampled exact-source replay; no all-stage bound. Affine reparameterization is a diagnostic, not a repair or proof of Bondi mass balance.',rows=rows)
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n')
np.savez(a.output/'signals.npz',t=t,nonaffinity=kap,affine_time=affine_time,
    affine_rate=affine_rate,coordinate_news=news,coordinate_flux=energy,affine_reparameterized_flux=reparameterized)
print(json.dumps({key:value for key,value in record.items() if key not in ['rows','binding']}))
