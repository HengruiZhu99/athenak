"""Separate full native bulk error from an analysis-only exact-shell-halo control."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;O=D/'shell-angular-initial-diagnostic'
a=json.loads((O/'native-initial.json').read_text());b=json.loads((O/'analytic-shell-halos.json').read_text())
def norms(x):return [math.sqrt(x['integrals'][1][i]/x['integrals'][1][0]) for i in [1,2,3]]
result=dict(native_fixed03_norms=norms(a),exact_shell_halo_fixed03_norms=norms(b),relative_norm_change=[y/x-1 for x,y in zip(norms(a),norms(b))],replaced_ghosts=b['replaced_ghosts'],regions={})
for name,x in [('native',a),('exact_shell_halos',b)]:
 result['regions'][name]={}
 for lo,hi in [(.3,.78),(.78,.9),(.9,.95),(.95,1)]:
  q=[row for row in x['radial_bins'] if lo-1e-8<=row[0]<hi-1e-8]
  result['regions'][name][f'{lo:g}-{hi:g}']={k:sum(row[i+2] for row in q)/x['integrals'][1][i] for k,i in [('H2_fraction',1),('M2_fraction',2)]}
result['qualification']='Exact analytic initial reconstruction on the same verified topology; no evolution or active-field override in the halo comparison. M is the native conformal-metric contraction, not the physical-metric norm plotted in the independent fixed-point stencil controls. Exact initial values replace only ghosts with .70<r<.975; scri untouched. Same bulk/radial diagnostic operators, CPU and native integrals agree. Uniform coarse core is held fixed and excluded with r>.3; not an admitted long-time solution or convergence proof.'
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for axis,(label,index) in zip(axs,[('Hamiltonian',3),('Momentum (conformal norm)',4)]):
 for x,name,style in [(a,'Native halos','-'),(b,'Exact shell halos (analysis only)','--')]:
  q=np.array([row for row in x['radial_bins'] if row[0]>=.3-1e-8]);axis.semilogy((q[:,0]+q[:,1])/2,np.where(q[:,2]>0,q[:,index],np.nan),style,label=name)
 axis.axvspan(.78,.9,color='tab:orange',alpha=.12);axis.set(xlim=(.3,1),xlabel='Compact radius r',ylabel='Squared residual integral per 0.01 radial bin',title=label);axis.grid(alpha=.2);axis.legend(fontsize=8)
fig.suptitle('Boosted initial state: residual concentration persists with exact shell halos\nEmpty radial bins are left as gaps')
for ext in ['png','pdf']:fig.savefig(O/('native-vs-exact-halos.'+ext),dpi=170)
print(json.dumps(result,indent=2))
