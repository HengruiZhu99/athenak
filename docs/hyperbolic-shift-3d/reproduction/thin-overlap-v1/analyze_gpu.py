#!/usr/bin/env python3
"""Report a communication-only GPU pair at identical discretization/end time."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=[];reports=[]
for name in ['puncture-full','puncture-sparse']:
 r=json.loads((a.raw/(name+'.report.json')).read_text());reports.append(r)
 assert r['status']=='completed' and r['t']==.05 and r['steps']==100 and r['mpi_size']==12 and r['backend']=='SYCL'
 phases=np.asarray(r['phase_seconds_by_rank']);interface=np.asarray(r['interface_seconds_by_rank']);rank=int(np.argmax(phases.sum(axis=1)+interface))
 values=dict(zip(['Bulk RHS','Communication','Boundary/checks','Diagnostics','Bulk halo'],phases[rank]))
 values['Paired penalty']=float(interface[rank]);values['Other / startup']=float(r['seconds']-sum(values.values()));assert min(values.values())>=0
 rows.append(dict(name=name,wall_seconds=r['seconds'],representative_rank=rank,seconds=values,
  rhs_share=float(values['Bulk RHS']/r['seconds']),trace_setup_seconds=max(r['trace_setup_seconds_by_rank']),
  trace_application_seconds_max=max(r['interface_seconds_by_rank']),cells=r['cells'],floor=r['lapse_floor_node_interventions'],
  max_null=max(r['max_null_surface'],r['max_null_ghost'])))
for key in ['cells','n','dt','t','core_evolved_layers','shifted_overlap_donors','trace_kappa','trace_normal_ratio']:
 assert reports[0][key]==reports[1][key]
sparse=reports[1];received=sum(int(row[4]) for row in sparse['sparse_exchange_by_rank']);full=sparse['cells']*11
summary=dict(scope='Single sequential same-node 12-tile pair at t=0.15M; communication-only comparison. Other/startup includes unseparated synchronization. No accuracy-matched discretization claim.',
 rows=rows,wall_speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'],
 transfer=dict(logical_remote_bytes_per_stage_sparse=received*31*8,logical_remote_bytes_per_stage_full=full*31*8,
               reduction_fraction=1-received/full,scope='Logical remote payload, not measured network traffic.'))
(a.out/'performance.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained');bottom=np.zeros(2)
for key in rows[0]['seconds']:
 values=[r['seconds'][key] for r in rows];axes[0].bar([0,1],values,bottom=bottom,label=key);bottom+=values
axes[0].set_xticks([0,1],['Full gather','Cached sparse']);axes[0].set(ylabel='Wall seconds',title='L1 paired puncture: 100 steps');axes[0].legend(fontsize=7)
for kappa,style in [(0,'o-'),(1,'s-')]:
 data=[json.loads((a.raw/f'minkowski-l{layer}-k{kappa}.report.json').read_text()) for layer in (1,2,3)]
 assert all(r['status']=='completed' and r['lapse_floor_node_interventions']==0 for r in data)
 axes[1].semilogy([1,2,3],[r['max_state_deviation'] for r in data],style,label=f'kappa={kappa}')
axes[1].set(xlabel='Extra Cartesian evolved layers',ylabel='Maximum state deviation',title='Short Minkowski preservation',xticks=[1,2,3]);axes[1].legend()
for axis in axes:axis.grid(alpha=.2)
fig.suptitle('GPU preflight: short tests, not 3M stability or convergence')
fig.savefig(a.out/'performance.png',dpi=180);fig.savefig(a.out/'performance.pdf')
print(json.dumps(summary,indent=2))
