"""Apply the exact-time refinement audit over a prescribed shared trajectory."""
import argparse,json,subprocess,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
for name in ['coarse','medium','fine-early','fine-late','output']:
 p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--split-time',type=float,required=True)
p.add_argument('--stem',default='gamma2-short-convergence')
p.add_argument('--times',type=float,nargs='+',required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
for i,t in enumerate(a.times):
 dest=a.output/f'frame-{i:02d}'
 cmd=[sys.executable,str(Path(__file__).with_name('assess_refinement.py')),'--kind','spatial','--time',str(t),
      '--cases',str(a.coarse),str(a.medium),str(a.fine_early if t<=a.split_time+1e-13 else a.fine_late),'--output',str(dest)]
 result=subprocess.run(cmd,capture_output=True,text=True)
 if result.returncode:raise RuntimeError(result.stdout+result.stderr)
 rows.append(json.loads((dest/'assessment.json').read_text()))
summary=dict(times=a.times,minimum_orders={scope:min(r['orders'][scope] for r in rows) for scope in ['full','exterior']},
             maximum_orders={scope:max(r['orders'][scope] for r in rows) for scope in ['full','exterior']},
             rows=rows,scope=f'{len(rows)} exact saved times on the short shared trajectory; positive measured orders do not establish convergence at longer durations.')
(a.output/'trajectory.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained');t=3*np.array(a.times)
grid_sizes=[r['n'] for r in rows[0]['runs']]
for k,n in enumerate(grid_sizes):
 line,=ax[0,0].semilogy(t,[r['runs'][k]['diagnostics']['H_exterior_rms'] for r in rows],'-o',label=f'N={n}')
 mass=[abs(3*r['runs'][k]['diagnostics']['apparent_horizon_mass']-1) for r in rows]
 ax[0,1].semilogy(t,np.maximum(mass,1e-16),'-o',color=line.get_color(),label=f'N={n}')
for k,label in enumerate([f'{grid_sizes[i]} vs {grid_sizes[i+1]}' for i in range(2)]):
 line,=ax[1,0].semilogy(t,[r['comparisons'][k]['exterior']['mixed_rms'] for r in rows],'-o',label=label+', exterior')
 ax[1,0].semilogy(t,[r['comparisons'][k]['full']['mixed_rms'] for r in rows],'--',color=line.get_color(),label=label+', full')
for scope,style in [('full','--'),('exterior','-')]:ax[1,1].plot(t,[r['orders'][scope] for r in rows],style,label=scope)
for panel,title in zip(ax.flat,['Exterior Hamiltonian RMS','Absolute relative AH mass error','Mixed-profile self-differences','Measured spatial order']):
 panel.set(title=title,xlabel='Time / M');panel.grid(alpha=.2)
ax[0,0].legend(fontsize=10);ax[1,0].legend(fontsize=9);ax[1,1].legend(fontsize=10)
for ext in ['pdf','png']:fig.savefig(a.output/f'{a.stem}.{ext}',dpi=170)
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
