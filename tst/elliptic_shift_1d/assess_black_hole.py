#!/usr/bin/env python3
"""Read-only invariant assessment of saved black-hole/collapse profiles.

Includes failed-stage manifest extrema. Profile-only exterior constraints are
explicitly sampled diagnostics and cannot certify all-stage safety.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from horizon import apparent_horizons
from trajectory import read_trajectory


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
    for root in a.input:
      for path in sorted(root.glob('*/manifest.json')):
        folder=path.parent;m,rows,stages,segments=read_trajectory(folder)
        if m['status']=='running':raise ValueError('Read completed or failed trajectories only')
        expected=m['arguments']['mass'] if not m['arguments']['scalar'] else None
        sampled=[]
        for segment in segments:
          for profile in sorted(Path(segment['folder']).glob('profile-*.npz')):
            with np.load(profile,allow_pickle=False) as z:
                r=z['r'];c=z['diagnostic'];t=float(z['t']);ah=apparent_horizons(r,c[:,8],c[:,11]);beta=z['beta']
            radius=ah.get('apparent_horizon_r',0.);mask=r>max(.03,1.2*radius)
            exterior={field:float(np.sqrt(np.sum(r[mask]**2*c[mask,i]**2)/np.sum(r[mask]**2))) for i,field in enumerate(['H','M','Theta','Z'])}
            sampled.append(dict(time=t,profile_sha256=sha(profile),profile=str(profile),horizon=ah,
                exterior_mask_lower_radius=max(.03,1.2*radius),exterior_constraint_rms=exterior))
        peak={field:max(max(row[field+'_rms'] for row in rows),stages.get('max_'+field+'_rms',0)) for field in ['H','M','Theta','Z']}
        mass=[v['horizon']['apparent_horizon_mass'] for v in sampled if 'apparent_horizon_mass' in v['horizon']]
        summary=dict(name=folder.name,folder=str(folder.resolve()),status=m['status'],time=m.get('time'),failure=m.get('failure'),
            arguments=m['arguments'],binding=m['binding'],segments=segments,all_RK_stages=stages,
            peak_constraints=peak,sampled_horizon_profiles=sampled,expected_vacuum_mass=expected,
            max_absolute_horizon_mass_error=None if expected is None or not mass else max(abs(np.array(mass)-expected)),
            relative_horizon_mass_drift=None if not mass else float(max(abs(np.array(mass)-mass[0]))/mass[0]),
            admission=False,scope='Read-only component evidence; no successful collapse, trumpet or wormhole-to-trumpet admission inferred.')
        if mass:summary['final_extracted_horizon_mass']=mass[-1]
        records.append(summary)
    result=dict(analysis_sha256=sha(__file__),horizon_analysis_sha256=sha(Path(__file__).with_name('horizon.py')),records=records)
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for record in records:
        points=[v for v in record['sampled_horizon_profiles'] if 'apparent_horizon_mass' in v['horizon']]
        if points:axes[0].plot([v['time'] for v in points],[v['horizon']['apparent_horizon_mass'] for v in points],label=record['name'])
        for ax,field in zip(axes[1:],['H','M']):
            points=record['sampled_horizon_profiles'];ax.semilogy([v['time'] for v in points],[v['exterior_constraint_rms'][field] for v in points],label=record['name'])
    for ax,title in zip(axes,['Polynomial apparent-horizon mass','Sampled exterior Hamiltonian RMS','Sampled exterior momentum RMS']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Coordinate time');ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.savefig(a.output/'invariants.png',dpi=160);fig.savefig(a.output/'invariants.pdf');plt.close(fig)
    print(json.dumps([{key:r.get(key) for key in ['name','status','time','peak_constraints','max_absolute_horizon_mass_error','relative_horizon_mass_drift','admission']} for r in records]))


if __name__=='__main__':main()
