"""Export terminal Gamma-driver evidence, verify provenance, plot measured data."""
import argparse,hashlib,inspect,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from puncture import PunctureEvolution

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--long-cases',nargs='+',required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records=[];data={}
for family in ['moving-puncture-v1','moving-puncture-odd-v1']:
 for mp in sorted((a.root/family).glob('*/manifest.json')):
  src=mp.parent;m=json.loads(mp.read_text());name=f'{family}/{src.name}'
  if m['status']=='running':raise ValueError('Only terminal evidence may be exported: '+name)
  for rel,sha in m['sources'].items():assert digest(src/'source'/rel)==sha,(name,rel)
  assert digest(src/'source'/Path(m['arguments']['library']).name)==m['library_sha256']
  for fn,sha in m['checkpoint_hashes'].items():assert digest(src/fn)==sha,(name,fn)
  if 'parent' in m:
   for fn,sha in m['parent']['hashes'].items():assert digest(Path(m['parent']['path'])/fn)==sha
   assert digest(src/'continuation_harness.py')==m['continuation_harness_sha256']
  dest=a.output/name;dest.mkdir(parents=True)
  for fn in ['manifest.json','history.json','profiles.npz','failed-stage.npz','continuation_harness.py']:
   if (src/fn).exists():shutil.copy2(src/fn,dest/fn)
  h=json.loads((src/'history.json').read_text());z=np.load(src/'profiles.npz');data[name]=(m,h,z)
  record=dict(case=name,status=m['status'],t=m['t'],error=m['error'],source_checkpoint_parent_hashes_verified=True,final=h[-1])
  if m['arguments']['puncture']:
   mass=np.array([x['apparent_horizon_mass'] for x in h]);record.update(peak_relative_mass_drift=float(max(abs(mass/mass[0]-1))),final_relative_mass_error=float(mass[-1]*3-1))
  records.append(record)

def evolution(m):
 kwargs={k:v for k,v in m['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
 return PunctureEvolution(m['arguments']['n'],a.library,**kwargs)

plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,axes=plt.subplots(3,2,figsize=(10,10),layout='constrained');shift_window=[];lapse_window=[]
for name in a.long_cases:
 m,h,z=data[name];n=m['arguments']['n'];t=3*np.array([x['t'] for x in h]);label=f'N={n}'
 line,=axes[0,0].semilogy(t,[x['H_exterior_rms'] for x in h],label=label);color=line.get_color()
 axes[0,1].semilogy(t,[x['M_exterior_rms'] for x in h],color=color,label=label)
 axes[1,0].plot(t,1e6*(3*np.array([x['apparent_horizon_mass'] for x in h])-1),color=color,label=label)
 axes[1,1].plot(t,[x['g_min'] for x in h],color=color,label=label)
 e=evolution(m);e.rhs(z['last']);v=e.last
 shift_window.extend(v['beta'][e.r<=.4]);lapse_window.extend(v['alpha'][(e.r>0)&(e.r<=.4)])
 snapshot=f'{label}, t/M={3*m["t"]:g}'
 axes[2,0].plot(e.r,v['beta'],color=color,label=snapshot)
 axes[2,1].plot(e.r[1:],v['alpha'][1:],color=color,label=snapshot)
 if m['status']!='completed':
  for ax,key in [(axes[0,0],'H_exterior_rms'),(axes[0,1],'M_exterior_rms'),(axes[1,1],'g_min')]:ax.plot(t[-1],h[-1][key],'x',color=color,ms=8)
for ax,title in zip(axes.flat,['Exterior Hamiltonian RMS','Exterior momentum RMS','AH mass error [ppm]','Minimum radial metric','Final shift profiles','Final conformal lapse profiles']):
 ax.set(title=title);ax.grid(alpha=.2);ax.legend(fontsize=11)
for ax in axes[:2].flat:ax.set_xlabel('Time / M')
for ax in axes[2]:ax.set_xlabel('Compact radius r')
axes[1,0].axhline(0,color='black',lw=.7);axes[1,0].set_yscale('symlog',linthresh=.01)
axes[2,0].axhline(0,color='black',lw=.7);axes[2,0].set_ylim(min(shift_window)-.01,max(shift_window)+.01);axes[2,0].set_xlim(0,.4)
axes[2,1].set_xlim(0,.4);axes[2,1].set_ylim(0,max(lapse_window)*1.05)
for ext in ['png','pdf']:fig.savefig(a.output/('gamma-progress.'+ext),dpi=160)

f,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for family,case,style in [('moving-puncture-v1','puncture-n513','--'),('moving-puncture-v1','puncture-n513-half-dt',':'),* [('moving-puncture-odd-v1',f'puncture-n{n}','-') for n in [257,513,1025]]]:
 name=f'{family}/{case}'
 if name not in data:continue
 m,h,z=data[name];t=3*np.array([x['t'] for x in h]);label=f'N={m["arguments"]["n"]}, '+('odd shift' if 'odd' in family else 'positive-only')
 if case.endswith('half-dt'):label+=', half dt'
 line,=ax[0].semilogy(t,[x['H_interior_max'] for x in h],style,label=label);color=line.get_color()
 ax[1].plot(t,[x['g_min'] for x in h],style,color=color,label=label)
 if m['status']=='failed':
  qf=np.load(a.root/name/'failed-stage.npz')['q'];ax[1].plot(3*m['attempted_step']*m['dt'],min(1+qf[:,1]),'x',color=color,ms=7)
ax[0].set(title='Interior Hamiltonian maximum',ylabel='Maximum |H|');ax[1].set(title='Inner closure and metric positivity',ylabel='Minimum g')
ax[1].axhline(0,color='black',lw=.7)
for x in ax:x.set_xlabel('Time / M');x.grid(alpha=.2)
ax[0].legend(fontsize=10)
for ext in ['png','pdf']:f.savefig(a.output/('gamma-inner-closure.'+ext),dpi=160)

profiles=[]
for n in [257,513,1025]:
 name=f'moving-puncture-odd-v1/puncture-n{n}'
 if name not in data:continue
 m,h,z=data[name];assert m['status']=='completed' and abs(m['t']-.2)<1e-14
 e=evolution(m);e.rhs(z['last']);v=e.last;q=v['physical'][:,:7].copy();q[:,0]=v['chi'];q[:,1]=v['g'];q[:,6]=v['alpha'];q=np.c_[q,v['beta'],z['last'][:,8]]
 profiles.append((e,q))
comparisons=[]
for (e,q),(ef,qf) in zip(profiles,profiles[1:]):
 assert np.array_equal(e.r,ef.r[::2])
 weight=e.r**2*e.jac;weight[[0,-1]]*=.5;d=q-qf[::2]
 errors=np.sqrt(np.sum(weight[:,None]*d*d,axis=0)/sum(weight))
 comparisons.append(dict(n_coarse=len(q),n_fine=len(qf),field_rms=errors.tolist(),combined_rms=float(np.linalg.norm(errors))))
result=dict(runs=records,short_puncture_comparisons=comparisons,field_order=['chi','g','Arr','p+3','Theta','Lambda','alpha','beta','B'],long_cases=a.long_cases,goal_complete=False)
if len(comparisons)==2:result['short_mixed_profile_order']=float(np.log2(comparisons[0]['combined_rms']/comparisons[1]['combined_rms']))
(a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
hashes={str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file()};(a.output/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
