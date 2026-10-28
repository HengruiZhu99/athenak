"""Plots and terminal diagnostics for the verified completed 100M campaign."""
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from continue_series import verify
from continue_puncture import digest

p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--rerender',action='store_true');a=p.parse_args()
c=json.loads(a.cases.read_text());out=a.evidence/'plots';out.mkdir(exist_ok=a.rerender)
runs=[Path(x) for x in c['spatial']];manifests=[verify(x) for x in runs];histories=[json.loads((x/'history.json').read_text()) for x in runs]
t=np.array([x['t'] for x in histories[0]])
assert all(np.array_equal(t,[x['t'] for x in h]) for h in histories)
fields=['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms','apparent_horizon_mass','apparent_horizon_r','g_min','H_interior_max']
data={f:np.array([[x[f] for x in h] for h in histories]) for f in fields};mass=data['apparent_horizon_mass'];radius=data['apparent_horizon_r'];err=3*mass-1
np.savez_compressed(out/'histories.npz',t=t,**data)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
colors=['#2463a6','#d77816','#268654'];labels=['N = 257','N = 513','N = 1025']
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=190,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,6.7));fig.subplots_adjust(hspace=.33,wspace=.25)
for ax,f,title in zip(axes.flat,fields[:4],['Hamiltonian constraint','Momentum constraint',r'$\Theta$ constraint',r'$Z_r$ constraint']):
 for y,color,label in zip(data[f],colors,labels):ax.semilogy(3*t,np.ma.masked_less_equal(y,0),color=color,label=label)
 ax.set(xlabel='Time / M',ylabel='Exterior RMS',title=title,xlim=(0,100.0002))
axes[0,0].legend(fontsize=9);save(fig,'exterior-constraints-100M')
fig=plt.figure(figsize=(10,7.2));grid=fig.add_gridspec(2,2,hspace=.38,wspace=.3);axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
for i,(color,label) in enumerate(zip(colors,labels)):
 axes[0].semilogy(3*t,np.ma.masked_less_equal(abs(err[i]),0),color=color,label=label)
 axes[1].plot(3*t,radius[i],color=color,label=label)
for i in range(2):axes[2].semilogy(3*t,np.ma.masked_less_equal(abs(radius[i]-radius[i+1]),0),color=colors[i],label=['257 vs 513','513 vs 1025'][i])
for ax,title,y in zip(axes,['Horizon mass error','Horizon coordinate radius','Radius self-differences'],[r'$|M_{AH}/M-1|$',r'$r_{AH}$',r'$|\Delta r_{AH}|$']):ax.set(xlabel='Time / M',ylabel=y,title=title,xlim=(0,100.0002))
axes[0].legend(fontsize=9);axes[2].legend(fontsize=9);save(fig,'horizon-100M')
s=json.loads((a.evidence/'profiles/summary.json').read_text());sp=json.loads((a.evidence/'profiles/spatial.json').read_text())['rows'];tp=json.loads((a.evidence/'profiles/temporal.json').read_text())['rows']
fig,axes=plt.subplots(2,2,figsize=(10,6.7));fig.subplots_adjust(hspace=.32,wspace=.28)
region_colors={'full':'#2463a6','exterior':'#268654','puncture':'#b53343','scri_collar':'#8757a4'}
for name,color in region_colors.items():
 axes[0,0].plot([r['t_over_M'] for r in sp if r['t']>0],[r['orders'][name] for r in sp if r['t']>0],label=name.replace('_',' '),color=color)
axes[0,0].axhline(0,color='.3',lw=.6)
for name,color in [('full',colors[0]),('exterior',colors[2])]:
 for i,ls in [(0,'-'),(1,'--')]:axes[0,1].semilogy([r['t_over_M'] for r in sp],[r['pairs'][i][name]['mixed_rms'] for r in sp],ls,color=color,label=f'{name}, '+['257/513','513/1025'][i])
for i,ls in [(0,'-'),(1,'--')]:axes[1,0].semilogy([r['t_over_M'] for r in tp if r['t']>0],[r['pairs'][i]['exterior']['mixed_rms'] for r in tp if r['t']>0],ls,color=colors[i],label=['CFL .05 vs .025','CFL .025 vs .0125'][i])
rr=s['temporal_spatial_comparison']['rows']
for name,color in [('full',colors[0]),('exterior',colors[2])]:axes[1,1].semilogy([3*r['t'] for r in rr],[r['ratios'][name] for r in rr],label=name,color=color)
for ax,title in zip(axes.flat,['Spatial order (ten fields)','Spatial self-differences','Exterior timestep self-differences','Temporal / spatial difference']):ax.set(title=title,xlabel='Time / M',xlim=(0,100.0002));ax.legend(fontsize=8)
save(fig,'convergence-100M')
terminal=[]
for i,(m,h) in enumerate(zip(manifests,histories)):
 terminal.append(dict(n=m['arguments']['n'],t=m['t'],last=h[-1],signed_relative_mass_error=float(err[i,-1]),peak_relative_mass_error=float(max(abs(err[i]))),peak_relative_mass_drift=float(max(abs(mass[i]/mass[i,0]-1))),minimum_saved_g=float(min(data['g_min'][i])),stage_stats=m['stage_stats'],manifest_sha256=digest(runs[i]/'manifest.json')))
radius_orders=np.log2(abs(radius[0]-radius[1])/abs(radius[1]-radius[2]))
summary=dict(runs=terminal,horizon_radius=dict(final=radius[:,-1].tolist(),initial=radius[:,0].tolist(),final_self_difference_order=float(radius_orders[-1]),nonpositive_order_samples=int(np.sum(radius_orders<=0)),scope='Coordinate radius is gauge dependent; mass uses the outermost AH areal radius.'),plot_scope='All 1368 shared diagnostic samples; lines connect samples without smoothing. Nonpositive values are masked only on logarithmic axes. Horizon interior excluded from constraints, with normalization over exterior radial weights.')
(out/'terminal.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,out/'plot-source.py')
print(json.dumps(dict(final_radius=summary['horizon_radius'],mass_error=err[:,-1].tolist()),indent=2))
