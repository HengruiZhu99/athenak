from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
work=Path(__file__).resolve().parent;r=json.loads((work/'results.json').read_text());e=json.loads((work/'exact-closure-results.json').read_text());cases=r['cases']+e['cases'];lookup={c['name']:c for c in cases}
fig,axes=plt.subplots(1,3,figsize=(13.8,3.8),constrained_layout=True)
for closure,label,color in [('ray','Quadratic ray','#bb342f'),('nearest','Nearest interior','#268664'),('fallback','Inward fallback','#3366a9')]:
 vals=[lookup[f'N{n}-span2.2-{closure}-centered-ko0.1']['spectral']['rightmost_found'][0]['real'] for n in [16,20,24]];axes[0].plot([16,20,24],vals,'o-',label=label,color=color)
axes[0].axhline(0,color='k',lw=.7);axes[0].set(xlabel='N, span 2.2',ylabel='Largest observed Re λ',title='Centered transport spectra');axes[0].legend(frameon=False,fontsize=8)
for name,label,color in [('N24-span2.1-ray-centered-ko0.1','Ray + centered','#bb342f'),('N24-span2.1-ray-upwind-ko0.1','Ray + upwind','#268664'),('N24-span2.1-fallback-centered-ko0.1','Fallback + centered','#3366a9')]:
 hist=json.loads((work/(name+'-pulse.json')).read_text());axes[1].semilogy([x['time'] for x in hist],[x['linf'] for x in hist],label=label,color=color)
axes[1].axhline(1,color='k',ls='--',lw=.8,label='Continuum sup bound');axes[1].set(xlabel='t',ylabel='Sampled max |q|',title='Production geometry N24, span 2.1');axes[1].legend(frameon=False,fontsize=8)
name='N24-span2.1-ray-centered-ko0.1';points=np.genfromtxt(work/(name+'-points.csv'),delimiter=',',names=True);v=np.load(work/(name+'-eigenpairs.npz'))['eigenvectors'][:,0];weights=abs(v)**2;order=np.argsort(points['r']);axes[2].plot(points['r'][order],np.cumsum(weights[order])/weights.sum(),color='#bb342f');axes[2].axvline(.8,color='k',ls='--',lw=.8);axes[2].set(xlabel='r',ylabel='Cumulative squared eigenmode weight',title='Growing mode Re λ = 5.12319',xlim=(0,1),ylim=(0,1));axes[2].text(.08,.65,'96.63% lies at r > .8',transform=axes[2].transAxes,fontsize=9)
fig.savefig(work/'scalar-closure.png',dpi=170)
