#!/usr/bin/env python3
"""Plot independently verified boundary subphases; never add overlapping timers."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'output/hyperbolic-core-shell/aurora/preflight-boundary-profile-v1'
v=json.loads((raw/'verification.json').read_text());assert v['passed'] and v['execution_equivalence']['streaming_byte_equality']
rows=[]
for name in ['puncture-control','puncture-fast']:
    r=json.loads((raw/'output'/(name+'.report.json')).read_text())
    h=np.atleast_1d(np.genfromtxt(raw/'output'/(name+'.history.csv'),delimiter=',',names=True))
    rows.append(dict(name=name,wall_seconds=r['seconds'],step_seconds=float((h['wall_seconds'][-1]-h['wall_seconds'][0])/(h['steps'][-1]-h['steps'][0])),report=r))
r=rows[1]['report'];phases=np.array(r['phase_seconds_by_rank']);boundary=np.array(r['boundary_phase_seconds_by_rank']);errors=np.array(r['error_sync_seconds_by_rank'])
assert phases.shape==(12,5) and boundary.shape==(12,6) and errors.shape==(12,)
assert np.all(np.isfinite(boundary)) and np.all(boundary>=0) and np.all(boundary.sum(axis=1)<=phases[:,2]+1e-5)
labels=['Null projection','Normal halo','Normal numerator RHS','Angular work','Scri RHS','Boundary error/statistics MPI']
rank=int(np.argmax(phases.sum(axis=1)))
result=dict(scope='One same-node 100-step puncture pair on 12 GPU tiles. Timers only; whole checkpoints identical. Boundary subphases subdivide closure time. All-error-sync overlaps boundary/other buckets and must not be added to exclusive totals.',
            cases=[{k:value for k,value in row.items() if k!='report'} for row in rows],
            profiled_over_control_step_ratio=rows[1]['step_seconds']/rows[0]['step_seconds'],
            selected_rank=rank,boundary_columns=labels,boundary_seconds_by_rank=boundary.tolist(),
            boundary_max_seconds=dict(zip(labels,boundary.max(axis=0).tolist())),
            all_error_sync_seconds_by_rank=errors.tolist(),
            outside_boundary_error_sync_lower_bound_by_rank=np.maximum(0,errors-boundary[:,5]).tolist(),
            outside_error_bound_note='Lower bound only: subtracting the boundary error/statistics bucket also subtracts its statistics reductions.',
            selected_rank_rhs_compute_seconds=dict(bulk=float(phases[rank,0]),normal_numerator=float(boundary[rank,2]),scri=float(boundary[rank,4])),
            selected_rank_unclassified_boundary_seconds=float(phases[rank,2]-boundary[rank].sum()))
out=Path(__file__).resolve().parent;(out/'gpu-profile.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(a,b)=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
x=np.arange(12);bottom=np.zeros(12)
for i,(label,color) in enumerate(zip(labels,['#56B4E9','#009E73','#0072B2','#CC79A7','#332288','#E69F00'])):
    a.bar(x,boundary[:,i],bottom=bottom,label=label,color=color);bottom+=boundary[:,i]
a.set(xlabel='MPI rank',ylabel='Boundary seconds',title='Boundary cost by operation',xticks=x)
a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2)
b.plot(x,phases[:,0],'.-',label='Bulk RHS')
b.plot(x,errors,'.-',label='All error-check MPI (overlaps other buckets)')
b.plot(x,boundary[:,5],'.-',label='Boundary error/statistics MPI')
b.set(xlabel='MPI rank',ylabel='Seconds',title='Computation and synchronization',xticks=x);b.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16))
fig.suptitle('100-step puncture: detailed boundary profile, same numerical evolution')
fig.savefig(out/'gpu-profile.png',dpi=170);fig.savefig(out/'gpu-profile.pdf')
print(json.dumps(result,indent=2))
