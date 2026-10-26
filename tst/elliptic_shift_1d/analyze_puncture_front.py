#!/usr/bin/env python3
"""Immutable saved-snapshot diagnostics of the inward lapse front.

A running trajectory may supply already-written profiles; no claim about its
future completion or cumulative constraint extrema is inferred from snapshots.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--case-paths',type=Path,nargs='+',required=True);p.add_argument('--times',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(len(a.times),4,figsize=(14,3*len(a.times)),squeeze=False,layout='constrained');records=[]
for root in a.case_paths:
 manifest=json.loads((root/'manifest.json').read_text());n=manifest['arguments']['n']
 (a.output/f'manifest-n{n}.json').write_text(json.dumps(manifest,indent=2)+'\n');profiles={}
 for path in sorted(root.glob('profile-*.npz')):
  with np.load(path,allow_pickle=False) as z:t=float(z['t'])
  for requested in a.times:
   if abs(t-requested)<1e-10:profiles[requested]=path
 if set(profiles)!=set(a.times):raise ValueError('No exact saved profile for every requested time')
 for row,t in enumerate(a.times):
  path=profiles[t]
  with np.load(path,allow_pickle=False) as z:
   r=z['r'];diag=z['diagnostic'];q=z['q_physical'];beta=z['beta'];alpha=diag[:,7];chi=diag[:,8]
  mask=(r>.03)&(r<.3);ids=np.flatnonzero(mask);O=(1-r*r)/2
  physical_lapse=alpha[mask]/O[mask];physical_gradient=np.gradient(physical_lapse,r[mask],edge_order=2)
  F=alpha*alpha+4*manifest['arguments']['slicing']*O*O*alpha
  family=manifest['arguments']['slicing_family'];scale=manifest['arguments']['slicing_scale'];A=(1+r*r)/2
  if family=='rational':F=alpha*alpha+4*manifest['arguments']['slicing']*O*O*alpha*(A+scale)/(alpha+scale)
  if family=='blended-log':F=(1-4*O*O)*alpha*alpha+8*manifest['arguments']['slicing']*O**3*alpha
  Hindex=ids[np.argmax(abs(diag[mask,0]))];Mindex=ids[np.argmax(abs(diag[mask,1]))];j=np.argmax(abs(physical_gradient));k=ids[j]
  records.append(dict(n=n,time=t,profile=str(path.resolve()),profile_sha256=sha(path),manifest_snapshot=f'manifest-n{n}.json',status_at_capture=manifest['status'],h=float(r[1]-r[0]),maximum_physical_lapse_gradient=float(abs(physical_gradient[j])),gradient_location=float(r[k]),H_peak=float(abs(diag[Hindex,0])),H_location=float(r[Hindex]),M_peak=float(abs(diag[Mindex,1])),M_location=float(r[Mindex]),inward_lapse_characteristic_at_gradient=float(-beta[k]-np.sqrt(F[k]*chi[k])),theta_peak=float(max(abs(diag[mask,2]))),Z_peak=float(max(abs(diag[mask,3])))))
  for ax,values,label in zip(axes[row],[physical_lapse,q[mask,3]-3,diag[mask,0],diag[mask,1]],['Physical lapse','Khat','Hamiltonian','Momentum']):
   ax.plot(r[mask],values,label=f'N{n}');ax.set_title(f't={t}: {label}',fontsize=10);ax.set_xlabel('r');ax.grid(alpha=.25)
axes[0,0].legend();fig.savefig(a.output/'front.png',dpi=160);plt.close(fig)
record=dict(records=records,scope='Local saved-state gradients/constraints on .03<r<.3. Second-order gradient diagnostic, not an evolution operator or a shock theorem. Running-trajectory manifests are captured here and do not certify later states.',analysis_sha256=sha(Path(__file__)))
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
