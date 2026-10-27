"""Export immutable duration probes and compare nested-grid puncture profiles.

A plot or a completed trajectory is not a convergence certificate.
"""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from puncture import PunctureEvolution

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--cases',nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--compact-labels',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
profiles,axp=plt.subplots(1,2,figsize=(10,4),layout='constrained')
failure_fig,axf=plt.subplots(1,2,figsize=(10,4),layout='constrained')
records=[];fields=[]
for name in a.cases:
 src=a.root/name;m=json.loads((src/'manifest.json').read_text());h=json.loads((src/'history.json').read_text())
 if m['status']=='running':raise ValueError('Only terminal immutable evidence is exported')
 dest=a.output/name;dest.mkdir(parents=True)
 for fn in ['manifest.json','history.json','profiles.npz','failed-stage.npz','continuation_harness.py']:
  if (src/fn).exists():shutil.copy2(src/fn,dest/fn)
 z=np.load(src/'profiles.npz');n=len(z['r']);t=np.array([x['t'] for x in h]);mass=np.array([x['apparent_horizon_mass'] for x in h])
 label=f"N={n}, CFL={m['arguments']['cfl']:g}"
 source=m['arguments'].get('shift_source','reference')
 if source!='reference':label+=', '+('inner source' if source=='lapse-inner' else 'global source')
 if m['arguments'].get('connection_driver')=='metric':label+=', metric driver'
 if a.compact_labels:
  label=f"N={n}, "+{'reference':'original','lapse-adapted':'global source','lapse-inner':'inner source'}[source]
  if m['arguments'].get('connection_driver')=='metric':label+=' + V driver'
 if m['arguments'].get('shift_gauge')=='moving-puncture':label=f'N={n}, bulk Gamma driver'
 if m['arguments'].get('inner_shift',0):label+=f", b={m['arguments']['inner_shift']:g}"
 if m['status']!='completed':label+=' (stopped)'
 color=None
 for ax,key in [(axes[0,0],'H_exterior_rms'),(axes[0,1],'M_exterior_rms'),(axes[1,1],'pole')]:
  line,=ax.semilogy(t*3,[max(x[key],1e-16) for x in h],label=label,color=color);color=line.get_color()
  if m['status']!='completed':ax.plot(t[-1]*3,h[-1][key],'x',color=color,ms=8)
 axes[1,0].plot(t*3,1e6*(mass/(1/3)-1),label=label,color=color)
 e=PunctureEvolution(n,a.library,ratio=m['arguments']['ratio'],damping=m['arguments']['damping'],ko=m['arguments']['ko'],restoring=m['arguments']['restoring'],inner_shift=m['arguments'].get('inner_shift',0.),lapse_floor=m['arguments'].get('lapse_floor',1e-10),shift_source=m['arguments'].get('shift_source','reference'),connection_driver=m['arguments'].get('connection_driver','evolved'),shift_gauge=m['arguments'].get('shift_gauge','separated'),gamma_coefficient=m['arguments'].get('gamma_coefficient',1.),grid_stretch=m['arguments'].get('grid_stretch',16.),shift_inner_stencil=m['arguments'].get('shift_inner_stencil','positive'),inner_z4_damping=m['arguments'].get('inner_z4_damping',0.))
 e.rhs(z['last']);v=e.last
 axf[0].semilogy(t*3,[max(x['H_interior_max'],1e-16) for x in h],color=color,label=label)
 axf[1].plot(t*3,[x['g_min'] for x in h],color=color,label=label)
 axf[1].plot(3*m['t'],min(v['g']),'o',ms=3,color=color)
 if (src/'failed-stage.npz').exists():
  failed=np.load(src/'failed-stage.npz');axf[1].plot(3*m['attempted_step']*m['dt'],min(1+failed['q'][:,1]),'x',color=color,ms=7)
 snapshot_label=label+f", t/M={3*m['t']:g}"
 axp[0].plot(e.r[1:],v['alpha'][1:],color=color,label=snapshot_label)
 axp[1].plot(e.r[1:-1],v['alpha'][1:-1]/e.O[1:-1],color=color,label=snapshot_label)
 physical=v['physical'][:,:7].copy();physical[:,0]=v['chi'];physical[:,1]=v['g'];physical[:,6]=v['alpha']
 physical=np.c_[physical,v['beta'],z['last'][:,8]]
 fields.append((name,e,z,physical,m))
 records.append(dict(case=name,status=m['status'],stop_reason=m['error'],t=m['t'],t_over_M=3*m['t'],n=n,
  diagnostic_sample_time=float(t[-1]),
  final_relative_mass_error=float(mass[-1]*3-1),peak_relative_mass_drift=float(max(abs(mass/mass[0]-1))),
  final_exterior={k:h[-1][k] for k in ['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms']},
  final_interior={k:h[-1][k] for k in ['H_interior_max','M_interior_max','Theta_interior_max','Z_interior_max']},
  stage_stats=m['stage_stats'],final_clock_rate=v['raw_clock_rate']))
for ax,title in zip(axes.flat,['Exterior Hamiltonian RMS','Exterior momentum RMS','AH mass error relative to analytic M [ppm]','Scri numerator maximum']):
 ax.set(xlabel='Time / M',title=title);ax.grid(alpha=.2)
axes[1,1].axhline(.01,ls=':',color='black',lw=1,label='Guard')
axes[0,0].legend(fontsize=8);axes[1,0].axhline(0,color='black',lw=.7)
axes[1,0].set_yscale('symlog',linthresh=.1)
axp[0].set(xlabel='Compact radius r',ylabel='Conformal lapse',title='Final lapse profiles')
axp[1].set(xlabel='Compact radius r',ylabel='Physical lapse',title='Puncture and interior (all positive nodes)',xscale='log',yscale='log',xlim=(1e-9,.8),ylim=(1e-11,10))
axp[1].axhline(1e-10,ls=':',color='black',lw=1)
for ax in axp:ax.grid(alpha=.2);ax.legend(fontsize=8)
axf[0].set(xlabel='Time / M',ylabel='Interior maximum |H|',title='Interior errors retained')
axf[1].set(xlabel='Time / M',ylabel='Minimum radial metric g',title='Accepted states and failed-stage crosses')
axf[1].axhline(0,color='black',ls=':',lw=1)
for ax in axf:ax.grid(alpha=.2)
axf[0].legend(fontsize=7)
for f,name in [(fig,'duration-constraints'),(profiles,'duration-lapse'),(failure_fig,'duration-interior')]:
 f.savefig(a.output/(name+'.pdf'));f.savefig(a.output/(name+'.png'),dpi=170)
comparisons=[]
for (na,ea,za,qa,ma),(nb,eb,zb,qb,mb) in zip(fields,fields[1:]):
 if not np.isclose(ma['t'],mb['t'],rtol=0,atol=1e-12):continue
 if len(qa)==len(qb):stride=1
 elif len(qb)==2*len(qa)-1:stride=2
 else:continue
 if not np.allclose(ea.r,eb.r[::stride],rtol=0,atol=1e-14):continue
 d=qa-qb[::stride];weight=ea.r**2*ea.jac;weight[[0,-1]]*=.5
 diff=np.sqrt(np.sum(weight[:,None]*d*d,axis=0)/sum(weight))
 comparisons.append(dict(coarse=na,fine=nb,t=ma['t'],field_order=['chi','g','Arr','p+3','Theta','Lambda','alpha','beta','B'],field_rms=diff.tolist(),combined_rms=float(np.linalg.norm(diff)),scope='Same-gauge coordinate self-difference; full radial weights, no horizon masking'))
result=dict(runs=records,profile_comparisons=comparisons,goal_complete=False,
 caveat='Short probes only. Single survivors, unequal terminal times, and decreasing final errors alone do not certify 100M convergence.')
(a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()}
(a.output/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(result,indent=2))
