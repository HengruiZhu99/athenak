"""Compact manifest and white-background comparison of the bounded screen."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=[]
for path in sorted((root/'results').glob('*.json')):
    z=json.loads(path.read_text())
    if 'real' not in z or 'parameters' not in z:continue
    p=z['parameters'];h=p['h'];n=p['n'];g=z['real']
    rows.append(dict(file=path.name,n=n,dx=h,L=n*h,width=h*p['sponge_cells'],rate=p['sponge_rate'],kappa=p['kappa'],eta=p.get('eta',2),lapse_damping=p.get('lapse_damping',.1),order=p.get('sponge_order','before'),k=p.get('angle_y',0)/h if 'angle_y' in p else None,kh_pi=p.get('angle_y',0)/np.pi if 'angle_y' in p else None,gamma=g,factor50000=float(np.exp(min(700,50000*max(0,g)))),taper=p.get('kappa_taper_cells',0),plateau=p.get('sponge_end_cells',0),corner=path.name.startswith('corner')))
with (root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
budget=np.log(2)/50000
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
fig,axs=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
for rate in [.001,.005,.02]:
    rr=[r for r in rows if r['file'].startswith('coarse-h128-L8192-w') and r['kappa']==.1 and r['rate']==rate and abs(r['kh_pi']-1/32)<1e-12]
    rr=sorted(rr,key=lambda r:r['width'])
    axs[0].plot([r['width'] for r in rr],[r['gamma'] for r in rr],'o-',label=f'rate={rate:g}/M')
axs[0].axhline(budget,color='black',ls='--',lw=1)
axs[0].set(xlabel='Sponge width [M]',ylabel='Largest Re λ [1/M]',title='κ₁=0.1; L=8192M, dx=128M')
axs[0].legend(fontsize=8)
colors={2:'#b94a36',.02:'#256ca0'}
for eta in [2.,.02]:
    rr=sorted([r for r in rows if r['file'].startswith('gpu-exact-profile') and r['eta']==eta],key=lambda r:r['kh_pi'])
    axs[1].plot([r['kh_pi'] for r in rr],[max(0,r['gamma'])*1e6 for r in rr],'o-',color=colors[eta],label=f'η={eta:g}, lapse={rr[0]["lapse_damping"]:g}')
axs[1].axhline(budget*1e6,color='black',ls='--',lw=1,label='50,000M doubling limit')
axs[1].set(xlabel='Tangential kh / π',ylabel='Largest Re λ [10⁻⁶/M]',title='κ₁=0; flat normal model of GPU profile',ylim=(-.4,15))
axs[1].legend(fontsize=8)
tr=json.loads((root/'results/candidate-transient-50000.json').read_text())['records']
for k,label in enumerate(['Compact lapse','Measured growing mode']):
    times=[r['time'] for r in tr];v=[r['cases'][k]['weighted_state_norm'] for r in tr];v=np.array(v)/v[0]
    axs[2].plot(times,v,'o-',label=label)
axs[2].axhline(2,color='black',ls='--',lw=1)
axs[2].set(xlabel='t [M]',ylabel='Weighted full-state norm / initial',title='κ₁=0, η=2; width512M, rate0.001/M')
axs[2].legend(fontsize=8)
fig.savefig(root/'screen-summary.png',dpi=180)
manifest=dict(result_count=len(rows),budget=budget,model='Flat frozen α=χ=1, β=0, G1; D6/KO8/linear ghosts; original zero_rate; dense parity-sector spectra',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},reference=json.loads((root/'reference-provenance.json').read_text()))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(results=len(rows),plot=str(root/'screen-summary.png'))))
