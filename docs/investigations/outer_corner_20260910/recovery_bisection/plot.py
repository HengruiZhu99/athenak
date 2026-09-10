from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from criterion import read_history
r=Path(__file__).resolve().parent
s=json.loads((r/'state.json').read_text());fig,ax=plt.subplots(figsize=(11,6),layout='constrained');data=[]
cases=list(s['completed'])
if s.get('active'):cases.append(dict(s['active'],classification='running'))
for c in cases:
 files=list(Path(c['directory']).glob('*.hst'))
 if not files:continue
 h=read_history(files[0]);t=[x['time'] for x in h];v=[x['minLapse'] for x in h]
 ax.plot(t,v,label=f"A={c['amplitude']} ({c['classification']})",lw=1.3)
 data.extend(dict(amplitude=c['amplitude'],classification=c['classification'],time=x['time'],minLapse=x['minLapse']) for x in h)
ax.axhline(.01,color='black',ls='--',lw=1,label='Final-time threshold 0.01')
ax.axhline(1e-5,color='crimson',ls=':',lw=1,label='Early-stop threshold 1e-5')
ax.axhline(.1,color='gray',ls=':',lw=.8,label='Dispersal: first dip <0.1')
ax.axhline(.8,color='green',ls=':',lw=.8,label='Then recover >0.8')
ax.set(xlabel='Coordinate time (existing code units)',ylabel='Global minimum lapse',yscale='log',xlim=(0,200),title='N256 Brill-wave bisection: early lapse collapse/recovery; otherwise classify at t=200')
ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));fig.savefig(r/'lapse_all_runs.png',dpi=180);fig.savefig(r/'lapse_all_runs.pdf')
with (r/'lapse_all_runs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['amplitude','classification','time','minLapse']);w.writeheader();w.writerows(data)
