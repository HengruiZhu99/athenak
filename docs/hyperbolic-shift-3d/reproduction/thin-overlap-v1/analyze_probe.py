#!/usr/bin/env python3
"""Compare matched-time native runs without treating the coarse control as truth."""
import argparse,json,struct,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--coordinates',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;out=root/'oblique-probe';out.mkdir(exist_ok=True)
rows=json.loads((a.raw/'results.json').read_text());coords=np.genfromtxt(a.coordinates,delimiter=',',names=True)
comparison=[];states={};histories={}
for row in rows:
 name=row['name'];report=row['report'];header=report['checkpoint_header_bytes'];cp=a.raw/(name+'.checkpoint');raw=cp.read_bytes()
 info=struct.unpack('<16s4i3Q2d',raw[:72]);assert info[0].rstrip(b'\0')==b'HYP3D_MPI_V12'
 assert info[5]==report['cells'] and info[8]==report['t']
 nominal=26*report['n']**3;states[name]=np.frombuffer(raw,dtype='<f8',offset=header,count=31*report['cells']).reshape(-1,31)[:nominal].copy()
 histories[name]=np.genfromtxt(a.raw/(name+'.history.csv'),delimiter=',',names=True)
 for suffix in ['.report.json','.history.csv','.log']:shutil.copy2(a.raw/(name+suffix),out/(name+suffix))
 comparison.append(dict(name=name,status=report['status'],t=report['t'],steps=report['steps'],seconds=report['seconds'],cells=report['cells'],
   H_L2=report['H_L2'],M_L2=report['M_L2'],null_max=max(report['max_null_surface'],report['max_null_ghost']),
   floor_interventions=report['lapse_floor_node_interventions'],checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
   interface_seconds=report['interface_seconds_by_rank'][0]))
assert len(rows)==3
common_time=all(row['report']['t']==.04 and row['report']['status']=='completed' for row in rows)
if common_time:
 nominal=len(states['control']);xyz=np.column_stack([coords[k][:nominal] for k in ['x','y','z']]);r=coords['r'][:nominal];volume=coords['volume'][:nominal]
 for record in comparison:
  delta=states[record['name']]-states['control'];metrics={}
  for label,mask in [('whole',r<1),('transition',(r>.14)&(r<.27)),('inner',r<.14)]:
   w=volume*mask;norm=lambda values:float(np.sqrt(np.sum(w*values)/np.sum(w)))
   beta=delta[:,22:25];radial=np.sum(beta*xyz/r[:,None],axis=1)
   metrics[label]=dict(lapse=norm(delta[:,18]**2),chi=norm(delta[:,0]**2),radial_shift=norm(radial**2),
     tangential_shift=norm(np.maximum(0,np.sum(beta**2,axis=1)-radial**2)))
  record['difference_from_coarse_control']=metrics
result=dict(common_target_completed=common_time,scope='single-resolution oblique probe; differences from control are not convergence errors or reflected characteristic amplitudes',cases=comparison)
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(a.raw/'results.json',out/'commands.json')
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');colors=['#4477aa','#228833','#cc6677']
for record,color in zip(comparison,colors):
 name=record['name'];h=histories[name];axes[0,0].plot(h['t'],h['H_L2'],label=name,color=color)
 axes[0,1].plot(h['t'],h['M_L2'],label=name,color=color)
 axes[1,0].semilogy(h['t'],h['core_trace_value_rms'],label=name,color=color)
axes[0,0].set(xlabel='Coordinate time',ylabel='Hamiltonian L2');axes[0,1].set(xlabel='Coordinate time',ylabel='Momentum L2')
axes[1,0].set(xlabel='Coordinate time',ylabel='Core trace mismatch RMS',title='Trace surfaces move with layer count')
a=axes[1,1];x=np.arange(3);bottom=np.zeros(3)
for label,key in [('Bulk RHS','rhs_seconds'),('Ghosts','bulk_halo_seconds'),('Boundary/checks','closure_seconds'),('Penalty','interface'),('Remaining','remaining')]:
 values=[]
 for row in rows:
  r=row['report'];interface=r['interface_seconds_by_rank'][0]
  value=interface if key=='interface' else r['seconds']-sum(r[k] for k in ['rhs_seconds','bulk_halo_seconds','closure_seconds'])-interface if key=='remaining' else r[key]
  values.append(value)
 a.bar(x,values,bottom=bottom,label=label);bottom+=values
a.set_xticks(x,[r['name'] for r in comparison],rotation=15);a.set(ylabel='Seconds',title='Serial CPU, equal n/dt/end time');a.legend(fontsize=8)
for a in axes.flat:a.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Bounded oblique evolution: no accuracy-matched speedup claim')
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');print(json.dumps(result,indent=2))
