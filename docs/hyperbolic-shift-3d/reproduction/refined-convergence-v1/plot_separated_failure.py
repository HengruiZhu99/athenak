"""Recorded diagnostics only; no failed-state endpoint or temporal interpolation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parent/'stationary20-separated-failure'
rows=[json.loads(x) for x in (D/'state.history.jsonl').read_text().splitlines()]
t=np.array([r['t_over_M'] for r in rows])
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for key in ['H','M']:
 axs[0,0].semilogy(t,[r[key] for r in rows],'o-',label=key+' full')
 axs[0,0].semilogy(t,[r['fixed'][2][key] for r in rows],'s--',label=key+' r > 0.6')
axs[0,0].set_ylabel('Constraint residual norm');axs[0,0].legend(fontsize=8)
for key,label in [('pole_max','Pole maximum'),('clock_max','Clock maximum')]:
 axs[0,1].semilogy(t,[r[key] for r in rows],'o-',label=label)
axs[0,1].set_ylabel('Cumulative maximum');axs[0,1].legend(fontsize=8)
axs[1,0].semilogy(t,[r['horizon_trial_expansion_rms'] for r in rows],'o-',label='Expansion RMS')
axs[1,0].axhline(.001,color='gray',ls=':',label='Online tolerance')
axs[1,0].set_ylabel('Horizon expansion RMS');axs[1,0].legend(fontsize=8)
for i,label in [(1,'max |grad beta|'),(2,'max |D|'),(3,'max |grad D|')]:
 axs[1,1].semilogy(t,[r['gauge_max'][i] if r['gauge_max'][i]>0 else np.nan for r in rows],'o-',label=label)
axs[1,1].set_ylabel('Full-domain gauge maxima');axs[1,1].legend(fontsize=8)
for ax in axs.flat:
 ax.axvline(.463875,color='red',ls=':',label='Last accepted step')
 ax.set_xlabel('t / M');ax.grid(alpha=.2)
fig.suptitle('Large-center separated-driver N16: numerical failure after 0.463875M\n'
             'Four recorded diagnostics; lines only connect samples; no convergence claim')
fig.savefig(D/'history.png',dpi=170);fig.savefig(D/'history.pdf');plt.close(fig)
