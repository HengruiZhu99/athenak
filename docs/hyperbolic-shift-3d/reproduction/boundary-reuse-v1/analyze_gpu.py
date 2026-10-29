#!/usr/bin/env python3
"""Analyze the same-allocation, bitwise-equal GPU pair; one pair, not a scaling study."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'output/hyperbolic-core-shell/aurora/preflight-boundary-reuse-v1'
v=json.loads((raw/'verification.json').read_text());assert v['passed'] and v['execution_equivalence']['streaming_byte_equality']
rows=[]
labels=['Bulk RHS','Timed MPI gathers','Boundary / checks','Diagnostics','Bulk halo','Other / startup']
for name in ['puncture-control','puncture-fast']:
 r=json.loads((raw/'output'/(name+'.report.json')).read_text())
 h=np.genfromtxt(raw/'output'/(name+'.history.csv'),delimiter=',',names=True)
 p=np.asarray(r['phase_seconds_by_rank']);rank=int(np.argmax(p.sum(axis=1)))
 seconds=np.r_[p[rank],r['seconds']-sum(p[rank])];assert seconds[-1]>=0
 step_seconds=(h['wall_seconds'][-1]-h['wall_seconds'][0])/(h['steps'][-1]-h['steps'][0])
 rows.append(dict(name=name,wall_seconds=r['seconds'],steps=r['steps'],seconds_per_step_after_initial_diagnostic=float(step_seconds),rank=rank,seconds=dict(zip(labels,seconds.tolist())),percent=dict(zip(labels,(100*seconds/r['seconds']).tolist())),collective_seconds=r['collective_seconds_by_rank'][rank],collective_calls=r['collective_calls_by_rank'][rank],boundary_cache_hits=r['boundary_cache_hits']))
result=dict(scope='One sequential same-node pair, identical n16 initial data/dt/diagnostics; full checkpoint byte equality independently verified.',rows=rows,wall_speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'],step_speedup=rows[0]['seconds_per_step_after_initial_diagnostic']/rows[1]['seconds_per_step_after_initial_diagnostic'])
out=Path(__file__).resolve().parent;(out/'gpu-performance.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,ay)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
colors=['#0072B2','#E69F00','#CC79A7','#777777','#009E73','#BBBBBB']
left=np.zeros(2)
for label,color in zip(labels,colors):
 values=[r['seconds'][label] for r in rows];ax.barh(['Original execution','Boundary reuse'],values,left=left,label=label,color=color);left+=values
ax.set(xlabel='Seconds, longest accounted rank',title='Exclusive phase costs');ax.invert_yaxis();ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3)
vals=[r['seconds_per_step_after_initial_diagnostic'] for r in rows]
ay.bar(['Original execution','Boundary reuse'],vals,color=['#777777','#0072B2']);ay.set(ylabel='Seconds per step',title='After initial diagnostic; includes final diagnostic')
for i,x in enumerate(vals):ay.text(i,x,f'{x:.3f} s',ha='center',va='bottom')
ay.set_ylim(0,max(vals)*1.18)
fig.suptitle('100-step puncture comparison: same node / 12 GPU tiles, identical checkpoints',fontsize=12)
fig.savefig(out/'gpu-performance.png',dpi=170);fig.savefig(out/'gpu-performance.pdf')
print(json.dumps(result,indent=2))
