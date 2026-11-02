"""Plot measured operator consistency; no evolution convergence claim."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
x=np.array([16,24,32]);data=json.loads((D/'checks/refinement-summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4))
for name,label in [('radial_p2','Radial regular data'),('angular_p2_with_surface_B','Angular regular data'),('nonconstant_twist_metric','Nonconstant metric')]:
 for ax,key in zip(axes,['surface_geometry','dependent_D']):ax.loglog(x,data[name][key]['rms'],'o-',label=label)
for ax,title in zip(axes,['Geometric surface RHS','Dependent D trace']):
 ax.set(xlabel='N',ylabel='RMS operator error',title=title,xticks=x);ax.set_xticklabels(x);ax.grid(alpha=.25,which='both')
axes[0].legend(fontsize=8);fig.suptitle('Zero-restoration boundary audit; original scri operators retained');fig.tight_layout();fig.savefig(D/'checks/refinement.png',dpi=180)
