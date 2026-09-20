from pathlib import Path
import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
cases=[('pilots-v1/pulse_cubic','Trumpet, v1 cubic'),('pilots-v2/pulse_cubic','Trumpet, v2 cubic'),('pilots-v2/pulse_linear','Trumpet, v2 linear'),('isolation-v2/quadratic_trumpet','Trumpet, v2 quadratic'),('isolation-v2/second_plm_nofofc_trumpet','Trumpet, second order'),('isolation-v2/flat_cubic','Flat, v2 cubic'),('isolation-v2/flat_linear','Flat, v2 linear'),('wormhole-control','Wormhole, v2 cubic')]
plt.rcParams.update({'font.size':10,'axes.facecolor':'white','figure.facecolor':'white'})
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
records=[]
for folder,label in cases:
 f=ROOT/folder/'ks_background.user.hst'
 if not f.exists():continue
 a=np.loadtxt(f); names=re.findall(r'\[\d+\]=([^\s]+)',f.read_text().splitlines()[1]);cols={x:i for i,x in enumerate(names)}
 z=ROOT/folder/'ks_background.z4c.user.hst'; b=np.loadtxt(z);bn=re.findall(r'\[\d+\]=([^\s]+)',z.read_text().splitlines()[1]);bc={x:i for i,x in enumerate(bn)}
 finite=np.isfinite(a[:,cols['Theta-max']])&(a[:,cols['Theta-max']]>0)
 p,=ax[0].semilogy(a[finite,0],a[finite,cols['Theta-max']],label=label)
 finite2=np.isfinite(b[:,bc['Theta-norm']])&(b[:,bc['Theta-norm']]>0)
 ax[1].semilogy(b[finite2,0],np.sqrt(b[finite2,bc['Theta-norm']]),color=p.get_color())
 records.append({'case':folder,'last_history_M':float(a[-1,0]),'last_theta_max':float(a[-1,cols['Theta-max']]),'last_bad_metric':float(a[-1,cols['bad-metric']]),'note':'Final history may precede failure; consult pilot-summary.json.'})
for a in ax:a.set(xlabel=r'$t/M$',xlim=(0,80));a.grid(alpha=.2)
ax[0].set_ylabel(r'$\max|\Theta|$');ax[1].set_ylabel(r'$\left(\int_{\mathrm{included}} \Theta^2\,dV\right)^{1/2}$')
handles,labels=ax[0].get_legend_handles_labels()
fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,.14,1,1))
fig.savefig(ROOT/'pilot-comparison.png',dpi=170)
(ROOT/'pilot-history-summary.json').write_text(json.dumps(records,indent=2)+'\n')
