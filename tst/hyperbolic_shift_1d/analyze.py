"""Portable evidence export and comparison figures. No trajectory mutation."""
import json,gzip,hashlib,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from numerics import weights
root=Path(__file__).resolve().parents[2];raw=root/'output/hyperbolic-shift-1d'
out=root/'docs/hyperbolic-shift-1d/evidence';fig=root/'docs/hyperbolic-shift-1d/report/figures';fig.mkdir(parents=True,exist_ok=True)
export=out/'runs';export.mkdir(exist_ok=True);cases={};digests={}
for mp in sorted(raw.rglob('manifest.json')):
 m=json.loads(mp.read_text())
 if m['status']=='running':raise RuntimeError(f'Unfinished: {mp}')
 name=str(mp.parent.relative_to(raw)).replace('/','__');h=json.loads((mp.parent/'history.json').read_text())
 dest=export/name;dest.mkdir(exist_ok=True)
 for file in ['manifest.json','history.json']:
  src=mp.parent/file;shutil.copy2(src,dest/file);digests[str(src.relative_to(root))]=hashlib.sha256(src.read_bytes()).hexdigest()
 with np.load(mp.parent/'profiles.npz') as z:
  # Keep full saved fields for scientific comparisons, not image-derived curves.
  np.savez_compressed(dest/'profiles.npz',r=z['r'],t=z['t'],q=z['q'],last=z['last'],last_time=z['last_time'])
 cases[name]=(m,h,dest)
def get(name):return cases[name]
def profile(name):
 with np.load(get(name)[2]/'profiles.npz') as z:return z['r'].copy(),z['last'].copy()
def interp(r,q,target):
 result=[]
 for x in target:
  j=max(0,min(np.searchsorted(r,x)-3,len(r)-7));result.append(weights(r[j:j+7],x)@q[j:j+7])
 return np.asarray(result)
def norm(q,r):return float(np.sqrt(np.sum(r[:,None]**2*q*q)/np.sum(r*r)/q.shape[1]))
spatial_names=['qualification-v1__scan-r2-d1-s4-k0.5','strong-restoring-v1__strong-n129','strong-restoring-v1__strong-n257']
r0,q0=profile(spatial_names[0]);r1,q1=profile(spatial_names[1]);r2,q2=profile(spatial_names[2])
fields=[0,1,2,3,4,5,6,7,8] # dependent D endpoint excluded; same gauge across grids
d01=norm(q0[:,fields]-interp(r1,q1[:,fields],r0),r0)
d12=norm(interp(r1,q1[:,fields],r0)-interp(r2,q2[:,fields],r0),r0)
time_names=['strong-time-base','strong-time-half','strong-time-quarter'];tq=[profile(n)[1] for n in time_names];tr=profile(time_names[0])[0]
dt01=norm(tq[0][:,fields]-tq[1][:,fields],tr);dt12=norm(tq[1][:,fields]-tq[2][:,fields],tr)
summary=dict(spatial_t1=dict(difference_coarse_medium=d01,difference_medium_fine=d12,approx_order=float(np.log2(d01/d12)),
  scope='Same hyperbolic gauge, seven-point interpolation to coarse grid; mixed field norm, not a universal order.'),
 time_t015=dict(difference_dt_half=dt01,difference_half_quarter=dt12,observed_order=float(np.log2(dt01/dt12))),
 maximum_relative_null=max(x[0]['stage_stats']['max_relative_null'] for x in cases.values()),
 puncture_qualified=False,long_puncture_runs_launched=False,
 cases={name:dict(status=m['status'],t=m['t'],error=m['error'],final=h[-1]) for name,(m,h,_) in cases.items()},
 scope='Exploration with a rejected unqualified puncture candidate; no improved robustness or long-duration claim.')
(out/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'export-hashes.json').write_text(json.dumps(digests,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'lines.linewidth':1.6,'savefig.dpi':180})
colors=['#4477AA','#EE6677','#228833']
def save(f,name):
 f.savefig(fig/(name+'.png'),bbox_inches='tight');f.savefig(fig/(name+'.pdf'),bbox_inches='tight');plt.close(f)
f,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for name,col,n in zip(spatial_names,colors,[65,129,257]):
 m,h,_=get(name);t=[x['t'] for x in h]
 for ax,key in [(axs[0,0],'H'),(axs[0,1],'M')]:ax.semilogy(t,np.maximum([x[key] for x in h],1e-16),color=col,label=f'Hyperbolic N={n}')
for ax,key in [(axs[0,0],'H'),(axs[0,1],'M')]:
 _,h,_=get('strong-restoring-v1__elliptic-n129-long');ax.semilogy([x['t'] for x in h],np.maximum([x[key] for x in h],1e-16),'--',color='black',label='Elliptic N=129')
 ax.set(xlabel='Coordinate time',ylabel='Radial-weighted RMS',title='Hamiltonian constraint' if key=='H' else 'Momentum constraint');ax.set_ylim(1e-8,.05)
axs[0,0].legend(fontsize=8)
for name,label,style in [(spatial_names[1],'Hyperbolic N=129','-'),('strong-restoring-v1__elliptic-n129-long','Elliptic N=129','--')]:
 r,q=profile(name);axs[1,0].plot(r,q[:,6],style,label=label);axs[1,1].plot(r,q[:,7],style,label=label)
axs[1,0].set(title='Lapse deviation at t=1',xlabel='Compactified radius',ylabel=r'$\alpha-(1+r^2)/2$')
axs[1,1].set(title='Shift deviation at t=1',xlabel='Compactified radius',ylabel=r'$\beta^r+r$');axs[1,1].legend(fontsize=8)
f.suptitle('Finite gauge pulse: stronger restoring source (ξβ=4)\nCoordinate profiles may differ; constraints test the same vacuum geometry',fontsize=12)
save(f,'pulse-comparison')
f,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for name,label,c in [('pilot-n65','No restoring, N=65',colors[0]),('restored-v1__restored-n129','ξβ=1, N=129',colors[1]),(spatial_names[2],'ξβ=4, N=257',colors[2])]:
 m,h,_=get(name);t=[x['t'] for x in h];axs[0].semilogy(t,np.maximum([x['pole'] for x in h],1e-16),label=label,color=c)
 axs[1].plot(t,[x['max_relative_null']/np.finfo(float).eps for x in h],label=label,color=c)
 if m['status']=='failed':axs[0].plot(t[-1],h[-1]['pole'],'x',color=c,ms=8)
axs[0].axhline(.01,color='black',ls=':',label='Exploratory stop threshold');axs[0].set(xlabel='Coordinate time',ylabel='Maximum geometric/lapse numerator',title='Regularity failures are retained');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Coordinate time',ylabel='Maximum relative nullness / machine epsilon',title='Exact nullness survives those failures',ylim=(-.05,1.3))
save(f,'boundary-failures')
baseline=json.loads(gzip.decompress((root/'docs/elliptic-shift-1d/evidence/telegraph-attenuated-n1025-terminal-v1/summary.json.gz').read_bytes()))
bh=[x for x in baseline['rows'] if x['t']<=.2+1e-12];(out/'elliptic-reference-short.json').write_text(json.dumps(dict(n=1025,rows=bh,source='telegraph-attenuated-n1025-terminal-v1/summary.json.gz',resolution_caveat='Reference is finer; this is not a matched-resolution improvement test.'),indent=2)+'\n')
f,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for name,c in [('puncture-n129',colors[0]),('puncture-n257',colors[1])]:
 m,h,_=get(name);t=[x['t'] for x in h];label=f"Hyperbolic N={m['arguments']['n']}"
 for ax,key in [(axs[0,0],'H_exterior_rms'),(axs[0,1],'M_exterior_rms')]:ax.semilogy(t,[x[key] for x in h],color=c,label=label)
 axs[1,0].semilogy(t,np.maximum([abs(3*x['apparent_horizon_mass']-1) for x in h],1e-16),color=c,label=label)
 axs[1,1].semilogy(t,[max(x['pole'],1e-16) for x in h],color=c,label=label)
for ax,key in [(axs[0,0],'H_exterior_rms'),(axs[0,1],'M_exterior_rms')]:ax.semilogy([x['t'] for x in bh],[x[key] for x in bh],'k--',label='Elliptic reference N=1025')
axs[1,0].semilogy([x['t'] for x in bh],[max(3*x['AH_mass_error'],1e-16) for x in bh],'k--')
axs[1,1].axhline(.01,color='black',ls=':');axs[0,0].legend(fontsize=8)
for ax,title,ylabel in zip(axs.flat,['Exterior Hamiltonian','Exterior momentum','Apparent-horizon mass','Scri regularity'],['Exterior-weight-normalized RMS','Exterior-weight-normalized RMS',r'$|M_{AH}/(1/3)-1|$','Maximum numerator']):ax.set(title=title,xlabel='Coordinate time',ylabel=ylabel)
f.suptitle('Short wormhole trials: no long-time qualification\nN=129 stops at t≈0.194; N=257 reaches 0.2. Elliptic reference uses a finer grid.',fontsize=12)
save(f,'puncture-comparison')
print(json.dumps({k:summary[k] for k in ['spatial_t1','time_t015','maximum_relative_null']}))
