"""Same-radius scalar scatter in verified snapshots; not a harmonic decomposition."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
coords_path=ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv'
coords=np.genfromtxt(coords_path,delimiter=',',names=True);r=coords['r'];w=coords['volume']
radius,index=np.unique(np.round(r,12),return_inverse=True);W=np.bincount(index,weights=w);valid=W>0
cases=[('IT1','inner-telegrapher-v1','n16-inner-telegrapher-fresh10m-v1-segment01','37665a7936afc1e404c653a1ea8fc1051a437a31f3bc907dc5cadec76c483d9f'),('DO1','direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01','edc58b0df7ef0391567221d8b95990e82ab05aad6264bc87d7e4a49ff2449c5c'),('DG1','damping-ablation-v1','n16-damping-ablation-fresh10m-v1-segment01','b582e457f8a8ba9b143436225bb57318b7e28a241fb495b913eea4787bfa04cf')]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(scope='Verified 10M states; compactified-coordinate volume weighted grid points grouped by radius rounded to12decimals. Not a full spherical-harmonic decomposition or onset localization.',coordinates_sha256=digest(coords_path),states={})
fig,axes=plt.subplots(1,2,figsize=(11,4))
for (label,variant,campaign,sha),color in zip(cases,['#777777','#b55315','#1769aa']):
 cp=ROOT/'output/hyperbolic-core-shell'/variant/'aurora/campaigns'/campaign/'output/puncture.checkpoint';assert digest(cp)==sha
 a=np.memmap(cp,dtype='<f8',mode='r',offset=168,shape=(len(r),31));state=dict(checkpoint_sha256=sha,fields={})
 fields={'chi':a[:,0],'conformal_lapse':a[:,18]+(1+r*r)/2,'Theta':a[:,14],'K_minus_2Theta':a[:,13]-3}
 for name,value in fields.items():
  mean=np.divide(np.bincount(index,weights=w*value),W,out=np.zeros_like(W),where=valid);residual=value-mean[index]
  scatter2=np.divide(np.bincount(index,weights=w*residual**2),W,out=np.zeros_like(W),where=valid)
  regions={}
  for lo,hi in [(.16,.3),(.3,.6)]:
   m=(radius[index]>lo)&(radius[index]<hi)&(w>0);volume=np.sum(w[m]);total=np.sum(w[m]*value[m]**2);coherent=np.sum(w[m]*mean[index[m]]**2);scatter=np.sum(w[m]*residual[m]**2)
   assert abs(total-coherent-scatter)<1e-11*max(total,1e-20),(label,name,lo)
   regions[f'{lo:g}-{hi:g}']=dict(volume=float(volume),rms=float(np.sqrt(total/volume)),same_radius_scatter_rms=float(np.sqrt(scatter/volume)),scatter_squared_fraction=float(scatter/total),coherent_squared_fraction=float(coherent/total))
  state['fields'][name]=regions
  if name=='Theta':
   m=valid&(radius>.13)&(radius<.6)
   axes[0].plot(radius[m],mean[m],label=label,color=color)
   axes[1].plot(radius[m],np.sqrt(scatter2[m]),label=label,color=color)
   state['theta_profile']=dict(radius=radius[m].tolist(),mean=mean[m].tolist(),scatter_rms=np.sqrt(scatter2[m]).tolist())
 assert digest(cp)==sha;result['states'][label]=state
for ax,title in zip(axes,['Same-radius weighted mean of Theta','RMS variation of Theta at the same radius']):
 ax.set(xlabel='Compactified radius',title=title);ax.grid(alpha=.2);ax.axvspan(.16,.18,color='k',alpha=.12);ax.legend()
axes[1].set_yscale('log');fig.suptitle('N16 at10M: radial coherence and angular variation of the constraint field');fig.tight_layout();fig.savefig(HERE/'radial-scatter.png',dpi=150)
(HERE/'radial-scatter.json').write_text(json.dumps(result,indent=2)+'\n')
for label,state in result['states'].items():print(label,state['fields']['Theta']['0.16-0.3'])
