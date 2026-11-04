"""Localize errors in the admitted4.782375M checkpoint; no endpoint extrapolation."""
from pathlib import Path
import json,csv,gzip,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());O=D/'stationary-reference-v2-segment01';prefix=R/'output/refined-convergence/reference-endpoint-reader/segment01'
x=json.loads(prefix.with_suffix('.json').read_text());a=json.loads((O/'analysis-command.json').read_text());assert a['checkpoint_unchanged'] and a['returncode']==0 and max(z for r in a['native_report_norms_match'] for z in r)<1e-9
(O/'spatial-native.json').write_text(json.dumps(x,indent=2)+'\n');bins=np.array(x['radial_bins']);ints=np.array(x['integrals']);regions={}
for lo,hi in ((.3,.6),(.6,.78),(.78,.875),(.875,.9),(.9,.95),(.95,1)):
 # .875 is not a .01-bin boundary: use explicit .87-.90 combined region below.
 if lo==.875 or hi==.875:continue
 mask=(bins[:,0]>=lo-1e-12)&(bins[:,1]<=hi+1e-12)
 regions[f'{lo:g}-{hi:g}']={name:dict(squared_integral=float(bins[mask,col].sum()),fraction_of_r03=float(bins[mask,col].sum()/ints[1,col-2])) for name,col in (('H',3),('M',4),('Mr',5))}
for lo,hi in ((.78,.87),(.87,.9)):
 mask=(bins[:,0]>=lo-1e-12)&(bins[:,1]<=hi+1e-12);regions[f'{lo:g}-{hi:g}']={name:dict(squared_integral=float(bins[mask,col].sum()),fraction_of_r03=float(bins[mask,col].sum()/ints[1,col-2])) for name,col in (('H',3),('M',4),('Mr',5))}
mask=bins[:,0]>=.3-1e-12
peak={name:bins[mask][np.argmax(bins[mask,col])].tolist() for name,col in (('H',3),('M',4))}
parts=[];profiles={}
for p in sorted(prefix.parent.glob(prefix.name+'.rank*.csv')):
 data=p.read_bytes();(O/(p.name+'.gz')).write_bytes(gzip.compress(data,mtime=0));profiles[p.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());parts.append(np.genfromtxt(p,delimiter=',',names=True))
data=np.concatenate(parts);sample=data[(data['r']>.3)&(data['r']<1)];maxima={f:dict(value=float(sample[f][np.argmax(abs(sample[f]))]),r=float(sample['r'][np.argmax(abs(sample[f]))]),ray=int(sample['ray'][np.argmax(abs(sample[f]))])) for f in ('dr_chi','dr_alpha','dr_beta_dev_r','dr_D_r','H','Mnorm')}
r=dict(t_over_M=x['t_over_M'],fixed03_norms=a['fixed03_norms'],regional=regions,peak_bins=peak,sampled_exterior_maxima=maxima,profiles=profiles,qualification='Same native operator and exact admitted checkpoint, explicit known k/gamma backend roundoff calibration; full/fixed history norms reproduced within5.8e-14 scaled. Fractions from native-volume .01 radial bins outside r>.3, beyond every complete horizon. Regions split at .87 rather than .875 to avoid fractional-bin fabrication. Profiles are unsmoothed native-center rays including separately evolved overlap copies; maxima are sampled-ray extrema, not full-domain extrema. One intermediate checkpoint cannot prove convergence or continuum slicing regularity.')
(O/'spatial-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,col,name in zip(axes,(3,4),('Hamiltonian','Momentum')):
 ax.semilogy((bins[mask,0]+bins[mask,1])/2,bins[mask,col],'.-',ms=4)
 for lo,hi,color,label in ((.4,.54,'C1','Omega join'),(.62,.78,'C2','gauge join'),(.9,.95,'C3','refinement-test band')):ax.axvspan(lo,hi,color=color,alpha=.12,label=label)
 ax.set(xlabel='Compact r',ylabel=name+' squared integral per .01 bin');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle(f'Reference-v2 at{x["t_over_M"]:.4f}M: exterior r > .3');fig.savefig(O/'radial-errors.png',dpi=170);fig.savefig(O/'radial-errors.pdf');plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
for ray,label,color,mark in ((0,'shell +x near axis','C0','.'),(1,'shell near face corner','C1','x'),(2,'Cartesian near axis','C2','.'),(3,'shell -x near axis','C3','+')):
 q=data[(data['ray']==ray)&(data['r']>.3)&(data['r']<1)]
 for ax,f in zip(axes.flat,('chi','alpha','metric_rr','dr_beta_dev_r','H','Mnorm')):
  ax.scatter(q['r'],q[f],s=10,color=color,marker=mark,label=label,linewidths=.6);ax.set(xlabel='Compact r',ylabel=f);ax.grid(alpha=.2);ax.axvspan(.9,.95,color='gray',alpha=.08)
for ax in axes.flat:ax.legend(fontsize=7)
fig.suptitle('Unsmoothed field/gradient/constraint profiles at4.7824M; distinct evolved copies retained');fig.savefig(O/'profiles.png',dpi=170);fig.savefig(O/'profiles.pdf');plt.close(fig)
print(json.dumps(dict(regional=regions,peak_bins=peak,sampled_exterior_maxima=maxima),indent=2))
