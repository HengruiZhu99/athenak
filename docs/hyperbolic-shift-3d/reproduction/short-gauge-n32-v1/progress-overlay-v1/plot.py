"""Plot verified completed coarse histories against an immutable live N32 snapshot."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent/'short-gauge-v1'
snapshot=json.loads((HERE/'snapshot.json').read_text())
histories={};identities={}
for n in (16,24,32):
 p=HERE/'n32.history.csv' if n==32 else BASE/f'n{n}-completed/puncture.history.csv'
 if n==32: expected=snapshot['sha256']
 else:
  worker=json.loads((p.parent/'worker-status.json').read_text())
  expected=next(v['sha256'] for k,v in worker['cases'][0]['artifacts'].items() if k.endswith('.history.csv'))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 identities[n]=expected
 data=np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
 assert np.all(np.diff(data['t_over_M'])>0)
 histories[n]=data
end=float(histories[32]['t_over_M'][-1])
fields=[('H_exterior_L2','Hamiltonian'),('M_exterior_L2','Momentum magnitude'),('Mr_exterior_L2','Radial momentum')]
colors={16:'#b55315',24:'#1769aa',32:'#228844'}
fig,axes=plt.subplots(2,3,figsize=(14,8),sharey='col')
result={'scope':snapshot['scope'],'job':snapshot['job'],'captured_utc':snapshot['captured_utc'],'history_sha256':identities,'cutoff':'Each resolution uses its own complete candidate horizon, including unaccepted candidates; not a common-cutoff comparison.','shared_end_t_over_M':end,'resolutions':{}}
for n,data in histories.items():
 shared=data[data['t_over_M']<=end+1e-9]
 assert abs(shared[-1]['t_over_M']-end)<1e-8
 result['resolutions'][n]={'available_t_over_M':float(data['t_over_M'][-1]),'shared_endpoint':{k:float(shared[-1][k]) for k,_ in fields},'candidate_radius':float(shared[-1]['exterior_cutoff'])}
 for j,(field,label) in enumerate(fields):
  for row,part in [(0,shared),(1,data)]:
   axes[row,j].plot(part['t_over_M'],part[field],'.-',ms=3.8,lw=1.5,color=colors[n],label=f'N{n}'+(' (in progress)' if n==32 else ''))
for j,(_,label) in enumerate(fields):
 axes[0,j].set_title(label)
 for row in range(2):
  ax=axes[row,j];ax.set(xlabel='t / M',ylabel='Exterior RMS',xlim=(0,end if row==0 else 10));ax.grid(alpha=.22)
 axes[1,j].axvline(end,color=colors[32],ls=':',lw=1,alpha=.7)
axes[0,0].legend(fontsize=9)
fig.suptitle(f'SG1 exterior constraints: N16 / N24 / N32\nTop: shared interval through {end:.2f}M   |   Bottom: all available history',fontsize=16)
fig.tight_layout(rect=(0,.09,1,.91))
fig.text(.5,.025,'Same gauge, fresh wormhole data, direct chi/lapse, dt/dx = 0.0444444. N32 is a live snapshot.\nNormalized compactified-coordinate volume norms; each run uses its own candidate horizon. No extrapolation or gap filling.',ha='center',fontsize=10)
fig.savefig(HERE/'constraints.png',dpi=160)
result['ratios_at_shared_endpoint']={f'N{a}/N{b}':{k:result['resolutions'][a]['shared_endpoint'][k]/result['resolutions'][b]['shared_endpoint'][k] for k,_ in fields} for a,b in [(16,24),(24,32)]}
(HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
