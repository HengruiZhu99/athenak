"""Plot unsmoothed native shell rays; gradients use production mapped operators."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'comparison';D=P/'slices'
cases={'a':('Original A','#b55315'),'wide':('A: collar 0.7–0.9','#1769aa'),'broad':('A: collar 0.6–0.9 + C6 slice','#b02871')}
raw={k:np.genfromtxt(D/f'{k}.csv',delimiter=',',names=True) for k in cases}
# The innermost shell stores three extra evolved overlap cells (i>=N).
# Preserve them in CSV but plot each primary radial domain only once.
data={k:x[x['i']<json.loads((D/f'{k}.json').read_text())['n']] for k,x in raw.items()}
for ray in [0,1]:
 a=data['a'][data['a']['ray']==ray]
 for k in cases:
  b=data[k][data[k]['ray']==ray]
  assert np.array_equal(a['cell'],b['cell']) and np.array_equal(a['r'],b['r'])
summary={}
for k,arr in data.items():
 summary[k]={}
 for ray in [0,1]:
  x=arr[arr['ray']==ray];x=np.sort(x,order='r');keep=(x['r']>=.6)&(x['r']<=.9)
  xyz=np.array([x['x'][0],x['y'][0],x['z'][0]])/x['r'][0]
  metrics={}
  for field in ['H','Mr','Mnorm','dr_chi','dr_alpha','dr_beta_dev_r','dr_D_r']:
   z=x[keep];i=np.argmax(abs(z[field]));metrics[field]={'max_abs':float(abs(z[field][i])),'r_at_max':float(z['r'][i])}
  summary[k][str(ray)]={'direction':xyz.tolist(),'number_of_points':len(x),'region_0_6_to_0_9':metrics}
def decorate(ax,label):
 ax.axvspan(.7,.775,color='#999999',alpha=.14)
 for boundary in [.45,.75]:ax.axvline(boundary,color='gray',lw=.7,ls=':')
 ax.set(xlabel='compactified radius r',ylabel=label,xlim=(.16,1));ax.grid(alpha=.2)
for filename,fields,title in [
 ('field-slices.png',[('chi',r'$\chi$'),('alpha',r'Conformal lapse $\alpha$'),('metric_rr',r'$\widetilde\gamma_{rr}$'),('beta_dev_r',r'Radial shift deviation $(\beta+x)\cdot n$'),('D_r',r'$D\cdot n$'),('K',r'Conformal trace $K$')],'Native radial field slices at 10M — near face center'),
 ('gradient-slices.png',[('dr_chi',r'$\partial_r\chi$'),('dr_alpha',r'$\partial_r\alpha$'),('dr_beta_dev_r',r'$\partial_r[(\beta+x)\cdot n]$'),('dr_D_r',r'$\partial_r(D\cdot n)$')],'Native radial gradients at 10M — near face center')]:
 fig,axes=plt.subplots(2,3 if len(fields)==6 else 2,figsize=(13,7.5))
 for ax,(field,label) in zip(axes.flat,fields):
  for k,(name,color) in cases.items():
   x=np.sort(data[k][data[k]['ray']==0],order='r');ax.plot(x['r'],x[field],'.-',ms=4,lw=1.1,label=name,color=color)
  decorate(ax,label)
 axes.flat[0].legend(fontsize=8);fig.suptitle(title);fig.tight_layout(rect=(0,.055,1,.95));fig.text(.5,.012,'Primary N16 shell cell centers; no smoothing/interpolation. Dotted: shell interfaces 0.45, 0.75. Shaded: 0.7–0.775.\nDifferent gauges/slices can change field amplitudes; these profiles do not prove regularity or convergence.',ha='center',fontsize=9)
 fig.savefig(P/filename,dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for ray in [0,1]:
 for ax,(field,label) in zip(axes[ray],[('H','Hamiltonian constraint H (signed)'),('Mr','Radial momentum constraint (signed)'),('Mnorm','Conformal momentum norm')]):
  for k,(name,color) in cases.items():
   x=np.sort(data[k][data[k]['ray']==ray],order='r');ax.plot(x['r'],x[field],'.-',ms=4,lw=1.1,label=name,color=color)
  decorate(ax,label);ax.set_title('Near face center' if ray==0 else 'Near face corner / seam')
axes[0,0].legend(fontsize=8);fig.suptitle('Native constraint slices at 10M — same grid points and original derivative operators');fig.tight_layout(rect=(0,.055,1,.94));fig.text(.5,.012,'Native shell rays, r ≥ 0.16; both lie outside all final horizons. Shaded band: 0.7–0.775.\nExtra evolved overlap cells excluded from plotted lines. Dotted: shell interfaces. Native volume sums cross-checked.',ha='center',fontsize=9);fig.savefig(P/'constraint-slices.png',dpi=180);plt.close(fig)
(P/'slice-comparison.json').write_text(json.dumps({'scope':'Primary cells i<N plotted; extra evolved overlap cells remain in raw CSV. Two unsmoothed shell rays at one N16 endpoint. Maxima are descriptive and gauge-dependent; not a regularity norm or convergence test.','cases':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2))
