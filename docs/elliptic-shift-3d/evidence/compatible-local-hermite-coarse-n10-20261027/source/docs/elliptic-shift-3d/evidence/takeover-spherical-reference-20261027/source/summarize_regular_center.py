#!/usr/bin/env python3
"""Preserve evidence from passing pilots and rejected origin/endpoint variants."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/elliptic-shift-1d'))
p.add_argument('--output',type=Path,default=Path('docs/elliptic-shift-1d/evidence/regular-center-review-v1'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
names=[f'g1-xi1-ko01-matrix-v1/n{n}-t{t}' for t in [1,3] for n in [256,512,1024]]
names+=['node-origin-n1025-t3-v1','coupled-scalar-a001-n256-t3-v3',
        'half-ko05-n1024-t6-v1','half-fd4-n1024-t6-v1',
        'ko05-gauge-matrix-v1/n256-t6','ko05-gauge-matrix-v1/n512-t6',
        'ko05-gauge-a01-n512-t6-v1','ko05-scalar001-n512-t3-v1',
        'ko05-scalar001-matrix-v1/n256-t3','ko05-scalar001-matrix-v1/n1024-t3',
        'ko05-scalar0058-n256-t6-v1']
records=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for name in names:
    folder=a.root/name
    if not (folder/'manifest.json').exists():continue
    manifest=json.loads((folder/'manifest.json').read_text())
    rows=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
    if not rows:continue
    last=rows[-1];record=dict(name=name,status=manifest['status'],arguments=manifest['arguments'],
        binding=manifest['binding'],last=last,
        manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
    record['maxima']={key:max(x[key] for x in rows) for key in
        ['H_rms','M_rms','Theta_rms','Z_rms','scri_pole','scri_lapse_pole','regularity_trace_max']}
    if (folder/'frozen-source').exists():
        mismatches=[]
        for relative,expected in manifest['binding']['source_sha256'].items():
            source=folder/'frozen-source'/relative
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:mismatches.append(relative)
        library=folder/'frozen-source/runtime'/Path(manifest['arguments']['library']).name
        if not library.exists() or hashlib.sha256(library.read_bytes()).hexdigest()!=manifest['binding']['executable_sha256']:mismatches.append('library')
        record['frozen_source_mismatches']=mismatches
    failure=folder/'failure.json'
    if failure.exists():record['failure']=json.loads(failure.read_text())
    if manifest['arguments'].get('scalar'):
        initial=manifest.get('scalar_initial_data',{}).get('Bondi_mass')
        record['initial_Bondi_mass']=initial
        if initial is not None:record['initial_mass_minus_integrated_flux']=initial-last['radiated_energy']
    records.append(record)
    if ('-t3' in name or 'half-ko05' in name) and 'scalar' not in name:
        label=name.replace('g1-xi1-ko01-matrix-v1/','')
        for ax,key in zip(axes,['H_rms','M_rms']):
            ax.semilogy([x['t'] for x in rows],[max(x[key],1e-20) for x in rows],label=label)
for ax,title in zip(axes,['Hamiltonian','Momentum']):
    ax.set(xlabel='Hyperboloidal coordinate time',ylabel='Volume-weighted RMS',title=title)
    ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.savefig(a.output/'origin-failure.png',dpi=160);fig.savefig(a.output/'origin-failure.pdf');plt.close(fig)
order={}
for key in ['H_rms','M_rms','H_endpoint_max','M_endpoint_max']:
    rows=[x for x in records if x['name'].startswith('g1-xi1-ko01-matrix-v1/') and x['name'].endswith('-t1')]
    values=[x['last'][key] for x in rows]
    order[key]=dict(values=values,orders=np.log2(np.array(values[:-1])/np.array(values[1:])).tolist())
(a.output/'summary.json').write_text(json.dumps(dict(scope='Pilots and failed controls; stronger KO has separate evidence. No black-hole or long-time acceptance.',
    records=records,t1_convergence=order),indent=2)+'\n')
print(json.dumps(dict(runs=len(records),t1_convergence=order)))
