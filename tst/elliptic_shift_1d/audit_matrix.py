#!/usr/bin/env python3
"""Persist whole-trajectory gates, including stage peaks and failed controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--suite',choices=['general','noise','scalar'],default='general');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
records=[]
for path in sorted(a.input.glob('*/manifest.json')):
    m=json.loads(path.read_text());folder=path.parent
    rows=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()] if (folder/'history.jsonl').exists() else []
    stages=m.get('all_RK_stages',{})
    record=dict(name=folder.name,status=m['status'],time=m.get('time'),arguments=m['arguments'],binding=m['binding'],
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_RK_stages=stages,failure=m.get('failure'))
    if not rows:records.append(record);continue
    gates={}
    for field,stage,tolerance in [('scri_pole','max_scri_pole',1e-7),('scri_lapse_pole','max_lapse_pole',1e-7),
        ('scri_clock','max_scri_clock',1e-13),('scri_null','max_scri_null',1e-12)]:
        value=max(max(abs(r[field]) for r in rows),stages.get(stage,0))
        gates[field]=dict(maximum=value,tolerance=tolerance,pass_gate=value<=tolerance)
    for field in ['chi','alpha']:
        value=min(min(r['min_'+field] for r in rows),stages.get('min_'+field,float('inf')))
        gates['positive_'+field]=dict(minimum=value,pass_gate=value>0)
    record['gates']=gates;record['diagnostic_coverage']='all RK stages plus reports' if stages else 'report samples only'
    record['complete_component_gates']=m['status']=='finished' and all(g['pass_gate'] for g in gates.values())
    if a.suite=='noise':
        ratios={}
        for field in ['H','M','Theta','Z']:
            initial=rows[0][field+'_rms'];peak=max(max(r[field+'_rms'] for r in rows),stages.get('max_'+field+'_rms',0))
            ratios[field]=dict(initial=initial,maximum=peak,peak_ratio=peak/initial,final_ratio=rows[-1][field+'_rms']/initial)
        record['component_ratios']=ratios
        record['amplification_and_decay_gates']=all(v['peak_ratio']<=100 and v['final_ratio']<=1 for v in ratios.values())
    if a.suite=='scalar':
        initial=m.get('scalar_initial_data',{}).get('Bondi_mass')
        record['initial_mass']=initial;record['integrated_coordinate_time_flux']=rows[-1]['radiated_energy']
        record['mass_minus_flux']=initial-rows[-1]['radiated_energy'] if initial is not None else None
        record['final_mass_r09']=rows[-1].get('mass_r0.9')
        record['horizon_roots_max']=max(r.get('apparent_horizon_count',0) for r in rows)
    records.append(record)
result=dict(scope='Component gates and experiment records only. Convergence, physical extraction, durations and remaining experiments require separate assessment.',suite=a.suite,records=records)
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
if a.suite=='scalar' and records:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for record in records:
        folder=a.input/record['name'];rows=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
        n=record['arguments']['n']
        for ax,field in zip(axes[:2],['H_rms','M_rms']):
            ax.semilogy([r['t'] for r in rows],[max(r[field],1e-20) for r in rows],label=str(n))
    ordered=sorted((r for r in records if r.get('mass_minus_flux') is not None),key=lambda r:r['arguments']['n'])
    axes[2].loglog([r['arguments']['n'] for r in ordered],[abs(r['mass_minus_flux']) for r in ordered],'o-')
    for ax,title in zip(axes,['Hamiltonian RMS (sampled)','Momentum RMS (sampled)','Initial mass minus integrated flux']):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.25)
    for ax in axes[:2]:ax.set_xlabel('Coordinate time');ax.legend(title='N')
    axes[2].set_xlabel('N')
    fig.savefig(a.output/'scalar-refinement.png',dpi=160);fig.savefig(a.output/'scalar-refinement.pdf');plt.close(fig)
print(json.dumps([dict(name=r['name'],status=r['status'],component_gates=r.get('complete_component_gates'),noise_gates=r.get('amplification_and_decay_gates'),mass_minus_flux=r.get('mass_minus_flux')) for r in records]))
