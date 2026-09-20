import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).parent
face=json.loads((root/'trumpet-face-results.json').read_text())
refine=json.loads((root/'oblique-refinement.json').read_text())
mode=np.load(root/'actual-oblique-mode.npz')
fig,axs=plt.subplots(1,3,figsize=(11.4,3.3),layout='constrained',facecolor='white')
for deg,label in [(1,'Linear ghosts'),(2,'Quadratic ghosts'),(3,'Cubic ghosts')]:
    rows=[r for r in face if r['sector']=='full' and r['degree']==deg and r['inner']==4 and r['damping']]
    axs[0].plot([r['angle_y']/np.pi for r in rows],[r['gamma'] for r in rows],'-o',ms=4,label=label)
axs[0].set(xlabel=r'$k_y h/\pi$',ylabel=r'$\max\,\mathrm{Re}(\lambda)\ [M^{-1}]$')
axs[0].legend(frameon=False,fontsize=8)
u=mode['mode'].reshape(20,16);q=mode['Q'].reshape(3,16);x=(np.arange(16)+.5)*.25-2
for y,label in [(np.sum(abs(u)**2,axis=0),'State'),(abs(u[7])**2,r'$\Theta$'),(np.sum(abs(q)**2,axis=0),r'$Q_i$')]:
    axs[1].semilogy(x,np.maximum(y/y.max(),1e-15),'-o',ms=3,label=label)
axs[1].set(xlabel=r'Normal coordinate $x/M$',ylabel='Normalized squared amplitude',ylim=(1e-13,2))
axs[1].legend(frameon=False,fontsize=8)
for case,label in [('fixed_grid_k',r'Fixed $k_yh=\pi/2$'),('fixed_physical_k',r'Fixed $k_y=2\pi/M$')]:
    rows=[r for r in refine if r['case']==case and r['mode']=='radiation']
    axs[2].plot([1/r['h'] for r in rows],[r['gamma'] for r in rows],'-o',label=label,ms=4)
axs[2].set(xlabel=r'$M/h$',ylabel=r'$\max\,\mathrm{Re}(\lambda)\ [M^{-1}]$')
axs[2].legend(frameon=False,fontsize=8)
for a in axs:a.grid(alpha=.2)
fig.savefig(root/'shifted-oblique-boundary.png',dpi=180)
fig.savefig(root/'shifted-oblique-boundary.pdf')
