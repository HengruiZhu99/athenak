"""Pointwise derivative refinement and spacelike/map audit of exact boosted data."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());RAW=R/'output/refined-convergence/boosted-initial-point-constraints';O=D/'boosted-initial-point-constraints';O.mkdir(exist_ok=True)
a=np.genfromtxt(RAW/'constraints.csv',delimiter=',',names=True);hs=sorted(set(a['h']),reverse=True);summary=[]
for h in hs:
 q=a[a['h']==h];v=dict(h=float(h))
 for key in ['H','M','Mr']:
  j=int(np.argmax(abs(q[key])));v[key]=float(abs(q[key][j]));v[key+'_peak']={'r':float(q['r'][j]),'ray':int(q['ray'][j])}
 summary.append(v)
orders={k:[float(np.log(summary[i][k]/summary[i+1][k])/np.log(2)) for i in range(len(hs)-1)] for k in ['H','M','Mr']}
result=dict(samples=summary,orders=orders,minimum_spacelike_margin=float(a['Q'].min()),minimum_relative_spacelike_margin=float((a['Q']/a['Q0']).min()),maximum_relative_spacelike_margin=float((a['Q']/a['Q0']).max()),minimum_boost_map_determinant=float(a['detJ'].min()),csv_sha256=hashlib.sha256((RAW/'constraints.csv').read_bytes()).hexdigest(),qualification='Six rays and eleven exterior radii .3-.98 in the exact v=.1 native20-40M embedding transformed with the configured Omega. Independent Cartesian fourth-order first/second derivatives, no mesh interpolation/KO/RK. Residual decay supports spatial truncation rather than nonvacuum data at these points; not a uniform continuum proof or evolved convergence. Small positive Q near scri is expected; Q/Q0 and map positivity are reported separately.')
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['constraints.csv','build.json','compile.log','link.log','native.log']:shutil.copy2(RAW/name,O/name)
fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for key in ['H','M']:
 y=[x[key] for x in summary];ax[0].loglog(hs,y,'o-',label=key)
x=np.array(hs);ax[0].loglog(x,summary[0]['H']*(x/x[0])**4,'k:',label='Fourth order')
ax[0].set(xlabel='Independent Cartesian derivative spacing',ylabel='Maximum sampled absolute residual',title='Initial exact-boost constraints');ax[0].legend();ax[0].grid(alpha=.25,which='both')
q=a[a['h']==hs[0]]
for ray in sorted(set(q['ray'])):
 v=q[q['ray']==ray];ax[1].plot(v['r'],v['Q']/v['Q0'],'o-',label=f'ray {int(ray)}',ms=3)
ax[1].set(xlabel='Compact radius',ylabel='Boosted / unboosted spacelike margin',title='Positive margin and coordinate Jacobian');ax[1].grid(alpha=.25);ax[1].legend(ncol=2,fontsize=8)
fig.suptitle('Local point audit; no evolution or mesh-convergence claim')
for ext in ['png','pdf']:fig.savefig(O/('point-constraints-and-margin.'+ext),dpi=170)
print(json.dumps(result,indent=2))
