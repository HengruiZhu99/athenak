#!/usr/bin/env python3
"""Locate retained-domain constraint errors without omitting early transients."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from trajectory import read_trajectory

p=argparse.ArgumentParser()
p.add_argument('--case-paths',type=Path,nargs='+',required=True)
p.add_argument('--times',type=float,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(len(a.times),4,figsize=(14,3*len(a.times)),squeeze=False,layout='constrained')
records=[];runs=[]
for root in a.case_paths:
    m,rows,stages,segments=read_trajectory(root,allow_domain_transfer=True)
    if m['status']!='finished':raise ValueError('This assessment requires completed trajectories')
    name=root.name;arg=m['arguments'];label=f"{name}: {arg['metric_variable']}, N{arg['n']}"
    snapshot=name+'-manifest.json'
    (a.output/snapshot).write_text(json.dumps(m,indent=2)+'\n')
    runs.append(dict(case=str(root.resolve()),manifest_sha256=sha(root/'manifest.json'),manifest_snapshot=snapshot,
                     all_stage_extrema=stages,segments=segments))
    for t in a.times:
        matches=[]
        for segment in segments:
            for path in sorted(Path(segment['folder']).glob('profile-*.npz')):
                with np.load(path,allow_pickle=False) as z:
                    if abs(float(z['t'])-t)<1e-10:matches.append(path)
        if not matches:raise ValueError('No exact saved profile at requested time')
        path=matches[-1]
        with np.load(path,allow_pickle=False) as z:r=z['r'];diag=z['diagnostic'];q=z['q_physical']
        mask=r<.3;ids=np.flatnonzero(mask);O=(1-r[mask]**2)/2
        lapse=diag[mask,7]/O
        gradient=np.gradient(lapse,r[mask],edge_order=2)
        hi=ids[np.argmax(abs(diag[mask,0]))];mi=ids[np.argmax(abs(diag[mask,1]))]
        gi=ids[np.argmax(abs(gradient))]
        records.append(dict(case=name,n=arg['n'],metric_variable=arg['metric_variable'],time=t,
            profile=str(path.resolve()),profile_sha256=sha(path),H_peak=float(abs(diag[hi,0])),H_location=float(r[hi]),
            M_peak=float(abs(diag[mi,1])),M_location=float(r[mi]),H_left=float(diag[0,0]),M_left=float(diag[0,1]),
            physical_lapse_gradient_peak=float(max(abs(gradient))),gradient_location=float(r[gi]),
            near_cutoff_H_max=float(max(abs(diag[:7,0]))),near_cutoff_M_max=float(max(abs(diag[:7,1])))))
        row=a.times.index(t)
        for ax,values,title in zip(axes[row],[lapse,q[mask,3]-3,diag[mask,0],diag[mask,1]],
                                  ['Physical lapse','Khat','Hamiltonian','Momentum']):
            ax.plot(r[mask],values,label=label);ax.set_title(f't={t}: {title}',fontsize=10)
            ax.set_xlabel('r');ax.grid(alpha=.25)
axes[0,0].legend(fontsize=7)
fig.savefig(a.output/'retained-front.png',dpi=160);fig.savefig(a.output/'retained-front.pdf');plt.close(fig)
record=dict(records=records,runs=runs,analysis_sha256=sha(Path(__file__)),
    scope='Exact saved states, r<.3, plus entire-segment stage extrema. Peak locations distinguish cutoff and reference-profile/front errors. Second-order lapse-gradient diagnostic is not an evolution operator or a shock theorem. These retained-domain diagnostics do not admit the original full prefix or the wormhole experiment.')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(records=records)))
