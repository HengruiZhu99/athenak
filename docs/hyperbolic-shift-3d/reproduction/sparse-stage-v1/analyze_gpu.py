#!/usr/bin/env python3
"""Analyze a verified same-node sparse/control pair without changing acceptance."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'output/hyperbolic-core-shell/aurora/preflight-sparse-stage-v1'
v=json.loads((raw/'verification.json').read_text())
assert v['passed'] and v['execution_equivalence']['streaming_byte_equality']
labels=['Bulk RHS','Communication / waiting','Boundary / checks','Diagnostics','Bulk halo','Other / startup']
rows=[]
for name in ['puncture-control','puncture-fast']:
    r=json.loads((raw/'output'/(name+'.report.json')).read_text())
    h=np.atleast_1d(np.genfromtxt(raw/'output'/(name+'.history.csv'),delimiter=',',names=True))
    p=np.asarray(r['phase_seconds_by_rank']);rank=int(np.argmax(p.sum(axis=1)))
    seconds=np.r_[p[rank],r['seconds']-sum(p[rank])]
    assert seconds[-1]>=0 and r['timing_per_rank_exclusive']
    step_seconds=float((h['wall_seconds'][-1]-h['wall_seconds'][0])/(h['steps'][-1]-h['steps'][0]))
    rows.append(dict(name=name,wall_seconds=r['seconds'],steps=r['steps'],rank=rank,
                     seconds_per_step_after_initial_diagnostic=step_seconds,
                     seconds=dict(zip(labels,seconds.tolist())),rhs_seconds_by_rank=p[:,0].tolist(),
                     bulk_rhs_share=float(p[rank,0]/r['seconds']),
                     rhs_max_over_min=float(p[:,0].max()/p[:,0].min()),
                     collective_calls=r['collective_calls_by_rank'][rank]))
    if name=='puncture-fast':
        sparse=np.asarray(r['sparse_exchange_by_rank']);ranks=r['mpi_size'];cells=r['cells']
        assert sparse.shape==(ranks,7) and np.all(sparse[:,6]==1)
        assert sum(sparse[:,3])==sum(sparse[:,4])
        full_remote_points=cells*(ranks-1)
        transfer=dict(scope='Logical remote values per stage, not physical network-link traffic or bandwidth.',
                      sparse_remote_points=int(sum(sparse[:,4])),full_remote_points=full_remote_points,
                      sparse_remote_bytes=int(sum(sparse[:,4]))*31*8,
                      full_remote_bytes=full_remote_points*31*8,
                      value_reduction_fraction=1-float(sum(sparse[:,4]))/full_remote_points,
                      receive_points_by_rank=sparse[:,4].tolist(),
                      timing_by_rank=sparse[:,:3].tolist(),
                      setup_by_rank=r['mesh_plan_setup_seconds_by_rank'])
result=dict(scope='One sequential same-node 12-tile pair; whole checkpoints equal. Startup included in native wall time; after-initial-diagnostic estimate includes final diagnostics. No long-time or convergence claim.',
            rows=rows,transfer=transfer,wall_speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'],
            step_speedup=rows[0]['seconds_per_step_after_initial_diagnostic']/rows[1]['seconds_per_step_after_initial_diagnostic'])
out=Path(__file__).resolve().parent
(out/'gpu-performance.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(a,b,c)=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
left=np.zeros(2)
for label,color in zip(labels,['#0072B2','#E69F00','#CC79A7','#777777','#009E73','#BBBBBB']):
    values=[r['seconds'][label] for r in rows];a.barh(['Frozen V11','Sparse exchange'],values,left=left,label=label,color=color);left+=values
for row,label in zip(rows,['Frozen V11','Sparse exchange']):b.plot(range(ranks),row['rhs_seconds_by_rank'],'.-',label=label)
b.set(xlabel='MPI rank',ylabel='Bulk RHS seconds',title='Remaining compute imbalance');b.legend()
c.bar(range(ranks),transfer['receive_points_by_rank'],color='#0072B2',label='Sparse donors')
remote=[cells-(cells*(r+1)//ranks-cells*r//ranks) for r in range(ranks)]
c.plot(range(ranks),remote,'--',color='#777777',label='Full gather')
c.set(xlabel='MPI rank',ylabel='Remote points per stage',title='Logical data requested');c.legend(loc='upper right',bbox_to_anchor=(1,.87))
a.set(xlabel='Seconds, longest accounted rank',title='Exclusive phase costs');a.invert_yaxis()
a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2)
fig.suptitle('100-step puncture: same node, 12 GPU tiles, identical checkpoints')
fig.savefig(out/'gpu-performance.png',dpi=170);fig.savefig(out/'gpu-performance.pdf')
print(json.dumps(result,indent=2))
