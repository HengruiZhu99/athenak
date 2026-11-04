"""Reproduce dense angular bulk-truncation controls; no evolved-grid claims."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
RAW=R/'output/refined-convergence/boosted-angular-survey'
O=D/'boosted-angular-survey';O.mkdir(exist_ok=True)
a=np.genfromtxt(RAW/'constraints.csv',delimiter=',',names=True)
assert len(a)==2160 and all(np.isfinite(a[k]).all() for k in a.dtype.names)
assert (a['Q']>0).all() and (a['Q0']>0).all() and (a['detJ']>0).all()
cases=[];points=None
for nr,na in [(12,12),(24,12),(12,24),(24,24),(48,48),(96,96)]:
 q=a[(a['Nr']==nr)&(a['Na']==na)];assert len(q)==360
 coords=np.array([q[k] for k in ['r','ray','x','y','z']])
 if points is None:points=coords
 else:assert np.array_equal(coords,points)
 c=dict(Nr=nr,Na=na)
 for k in ['H','M','Mr']:
  j=int(np.argmax(abs(q[k])));c[k]=float(abs(q[k][j]));c[k+'_peak']=dict(r=float(q['r'][j]),theta_degrees=float(np.degrees(np.arccos(q['x'][j]/q['r'][j]))),azimuth_degrees=45*int(q['ray'][j]//36))
 cases.append(c)
both=[c for c in cases if c['Nr']==c['Na']]
result=dict(cases=cases,point_count=360,angular_samples_per_meridian=36,azimuths_degrees=[0,45],radii=sorted(set(a['r'])),minimum_Q_over_Q0=float(np.min(a['Q']/a['Q0'])),minimum_boost_map_det=float(a['detJ'].min()),both_orders={k:[float(np.log(both[i][k]/both[i+1][k])/np.log(2)) for i in range(3)] for k in ['H','M','Mr']},radial_2x_ratio={k:cases[1][k]/cases[0][k] for k in ['H','M']},angular_2x_ratio={k:cases[2][k]/cases[0][k] for k in ['H','M']},qualification='Same fixed physical points, exact native stencil values, production cubed-sphere coordinate maps and fourth-order bulk derivatives. Two meridians, not a full sphere or volume norm. Stencils may cross a face boundary analytically; this does not test actual seam interpolation or exchange. No evolution, KO, scri closure, quadrature or altered boost frame. Pointwise initial constraint residual order is not evolved field self-convergence or a uniform spacelikeness proof.')
result['input_sha256']=hashlib.sha256((RAW/'constraints.csv').read_bytes()).hexdigest()
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['constraints.csv','build.json','compile.log','link.log','native.log']:shutil.copy2(RAW/name,O/name)
fig,axs=plt.subplots(2,3,figsize=(12.5,7),sharex=True,constrained_layout=True)
for col,r in enumerate([.8875,.915625,.946875]):
 for row,k in enumerate(['H','M']):
  ax=axs[row,col]
  for n in [12,24,48,96]:
   q=a[(a['Nr']==n)&(a['Na']==n)&(a['r']==r)&(a['ray']<36)]
   ax.semilogy(np.degrees(np.arccos(q['x']/q['r'])),np.abs(q[k]),'.-',label=f'Nᵣ=Nₐ={n}',markersize=3)
  ax.set(title=f'r = {r:g}',ylabel=f'Pointwise |{k}|',xlabel='Angle from +x (degrees)',xticks=[0,45,90,135,180]);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
fig.suptitle('Boosted initial data: angular structure and bulk FD refinement\nExact analytic stencil values on the xy meridian; no halo or evolution')
for ext in ['png','pdf']:fig.savefig(O/('angular-constraint-profiles.'+ext),dpi=160)
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','qualification']},indent=2))
