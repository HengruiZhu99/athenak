#!/usr/bin/env python3
"""Summarize a verified same-node radial batching pair; no cross-job speedup product."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'output/hyperbolic-core-shell/aurora/preflight-radial-field-batch-v2'
v=json.loads((raw/'verification.json').read_text())
assert v['passed'] and v['execution_equivalence']['streaming_byte_equality']
artifacts={Path(x['remote_path']).name:x for x in v['artifacts']}
def checked(name):
 p=raw/'output'/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==artifacts[name]['sha256']
 return p
rows=[]
for name in ['puncture-control','puncture-fast']:
 r=json.loads(checked(name+'.report.json').read_text())
 h=np.atleast_1d(np.genfromtxt(checked(name+'.history.csv'),delimiter=',',names=True))
 assert r['steps']==100 and h['steps'][-1]-h['steps'][0]==100
 phases=np.array(r['phase_seconds_by_rank']);boundary=np.array(r['boundary_phase_seconds_by_rank'])
 assert phases.shape==(12,5) and boundary.shape==(12,6)
 assert np.all(np.isfinite(phases)) and np.all(phases>=0) and np.all(np.isfinite(boundary)) and np.all(boundary>=0)
 assert np.all(boundary.sum(axis=1)<=phases[:,2]+1e-5)
 rank=int(np.argmax(phases.sum(axis=1)))
 rows.append(dict(name=name,wall_seconds=r['seconds'],
  step_seconds=float((h['wall_seconds'][-1]-h['wall_seconds'][0])/100),
  selected_rank=rank,exclusive_phase_seconds_by_rank=phases.tolist(),
  boundary_seconds_by_rank=boundary.tolist(),
  selected_rank_rhs_seconds=float(phases[rank,0]+boundary[rank,2]+boundary[rank,4]),
  max_bulk_rhs_seconds=float(phases[:,0].max()),max_normal_rhs_seconds=float(boundary[:,2].max()),
  min_normal_rhs_seconds=float(boundary[:,2].min()),
  maximum_null=max(r['max_null_ghost'],r['max_null_surface']),floor_interventions=r['lapse_floor_node_interventions']))
result=dict(scope='Single same-node 100-step n16 puncture comparison, all 12 GPU tiles; whole checkpoints byte-identical. Not a long-time stability/convergence result. No multiplication by speedups measured in other jobs.',
 job=v['terminal']['job_id'],cases=rows,
 control_over_candidate_step_speedup=rows[0]['step_seconds']/rows[1]['step_seconds'],
 control_over_candidate_native_wall_speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'],
 timer_note='Boundary subphases subdivide closure. All-error-sync overlaps other timers and is excluded from stacks; phase maxima across ranks are not summed.')
out=Path(__file__).resolve().parent
(out/'gpu-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(a,b)=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
labels=['Inline radial derivatives','Field-batched radial derivatives']
a.bar([0,1],[r['step_seconds'] for r in rows],color=['#777777','#0072B2'])
for i,r in enumerate(rows):a.text(i,r['step_seconds'],f"{r['step_seconds']:.4f}",ha='center',va='bottom')
a.set(xticks=[0,1],xticklabels=['Inline','Field-batched'],ylabel='Seconds per step',title='Elapsed time after initial diagnostics')
a.set_ylim(0,max(r['step_seconds'] for r in rows)*1.18)
x=np.arange(12)
for row,label,style,color in zip(rows,labels,['--','-'],['#777777','#0072B2']):
 phase=np.array(row['exclusive_phase_seconds_by_rank']);boundary=np.array(row['boundary_seconds_by_rank'])
 b.plot(x,phase[:,0],style,marker='o',color=color,label=label+': bulk RHS')
 b.plot(x,boundary[:,2],style,marker='s',color=color,alpha=.65,label=label+': normal RHS')
b.set(xticks=x,xlabel='MPI rank',ylabel='Seconds over 100 steps',title='Physical RHS computation')
b.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16))
fig.suptitle('Radial field batching: identical puncture trajectory, same node')
fig.savefig(out/'gpu-comparison.png',dpi=170);fig.savefig(out/'gpu-comparison.pdf')
print(json.dumps({k:value for k,value in result.items() if k!='cases'},indent=2))
