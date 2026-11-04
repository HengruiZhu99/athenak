"""Isolate radial and angular FD truncation using exact native stencil values."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());RAW=R/'output/refined-convergence/boosted-shell-derivative-control';O=D/'boosted-shell-derivative-control';O.mkdir(exist_ok=True)
a=np.genfromtxt(RAW/'constraints.csv',delimiter=',',names=True);cases=[]
for nr,na in [(12,12),(24,12),(48,12),(96,12),(12,24),(12,48),(12,96),(24,24),(48,48),(96,96)]:
 q=a[(a['Nr']==nr)&(a['Na']==na)];assert len(q)==48
 d=dict(Nr=nr,Na=na)
 for key in ['H','M','Mr']:
  j=np.argmax(abs(q[key]));d[key]=float(abs(q[key][j]));d[key+'_peak']=dict(r=float(q[j]['r']),ray=int(q[j]['ray']))
 cases.append(d)
base=cases[0];radial=[q for q in cases if q['Na']==12];angular=[q for q in cases if q['Nr']==12];both=[q for q in cases if q['Nr']==q['Na']]
result=dict(cases=cases,radial_8x_max_residual_ratio={k:radial[-1][k]/base[k] for k in ['H','M']},angular_8x_max_residual_ratio={k:angular[-1][k]/base[k] for k in ['H','M']},both_orders={k:[float(np.log(both[i][k]/both[i+1][k])/np.log(2)) for i in range(3)] for k in ['H','M']},qualification='Fixed physical points at twelve shell-interior radii and four rays. Same production cubed-sphere maps/Jacobians/Hessians and fourth-order bulk stencil coefficients, exact native values at every stencil point. Independently vary radial/angular logical spacing. This intentionally excludes halo interpolation, seams, KO, boundary closure, quadrature and time integration; not a full evolved-grid convergence test or a new production discretization.')
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['constraints.csv','build.json','compile.log','link.log','native.log']:shutil.copy2(RAW/name,O/name)
fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for k,axis in zip(['H','M'],ax):
 for data,label,xkey in [(radial,'Radial only; angular N=12','Nr'),(angular,'Angular only; radial N=12','Na'),(both,'Radial and angular together','Nr')]:axis.loglog([q[xkey] for q in data],[q[k] for q in data],'o-',label=label)
 axis.set(xlabel='Logical sampling resolution',ylabel=f'Maximum sampled |{k}|',title='Exact boosted initial data: '+k);axis.set_xticks([12,24,48,96],['12','24','48','96']);axis.minorticks_off();axis.grid(alpha=.25);axis.legend(fontsize=8)
fig.suptitle('Independent bulk-stencil control; analytic stencil values, no halo or evolution')
for ext in ['png','pdf']:fig.savefig(O/('radial-angular-isolation.'+ext),dpi=170)
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
