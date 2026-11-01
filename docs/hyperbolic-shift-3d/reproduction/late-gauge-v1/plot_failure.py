from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
failure=json.loads((HERE/'n16-failed/failure-analysis.json').read_text());assert failure['artifact_hashes_verified'] and not failure['checkpoint_admitted']
paths={'IT1':HERE.parent/'inner-telegrapher-v1/n16-completed/puncture.history.csv','DO1':HERE.parent/'direct-outer-v1/n16-completed/puncture.history.csv','DG1':HERE.parent/'direct-outer-investigation-v1/dg1-completed/puncture.history.csv','LG1':ROOT/'output/hyperbolic-core-shell/late-gauge-v1/aurora/campaigns/n16-late-gauge-fresh10m-v1-segment01/output/puncture.history.csv'}
fig,axes=plt.subplots(2,2,figsize=(11,7))
fields=[('H_L2','Full-domain Hamiltonian RMS'),('H_exterior_L2','Candidate-exterior Hamiltonian RMS'),('M_exterior_L2','Candidate-exterior momentum RMS'),('Theta_exterior_L2','Candidate-exterior Theta RMS')]
for (name,path),color,label in zip(paths.items(),['#777777','#b55315','#1769aa','#228844'],['Earlier gauge (IT1)','Direct outer (DO1)','Restored damping (DG1)','Later transition (LG1)']):
 d=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True));d=d[d['t_over_M']<=3+1e-10]
 for ax,(field,title) in zip(axes.flat,fields):ax.plot(d['t_over_M'],d[field],label=label,color=color,marker='.',markersize=3);ax.set(xlabel='t / M',ylabel=title,yscale='log',xlim=(0,3.02));ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('N16: moving the gauge transition outward causes earlier failure')
fig.tight_layout(rect=(0,.08,1,.96));fig.text(.5,.02,'LG1: last diagnostic 2.85M, last accepted step 2.9715M. No extrapolation to the failure.\nExterior histories use each complete candidate horizon; these are not shared-cutoff norms.',ha='center',fontsize=9)
fig.savefig(HERE/'failure-comparison.png',dpi=150)
