from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
v=np.fromfile(ROOT/'mode0-real.bin').reshape(25,24,24,24)[:,4:20,4:20,4:20]
c=np.load(ROOT/'physical-constraints.npz');x=(np.arange(16)+.5)*.25-2
fig,ax=plt.subplots(1,3,figsize=(11.4,3.5),layout='constrained')
for a,z,title in [(ax[0],7,'Midplane: z = −0.125M'),(ax[1],15,'Outer face: z = 1.875M')]:
 im=a.imshow(abs(v[17,z]),extent=(-2,2,-2,2),origin='lower',cmap='magma',vmin=0,vmax=abs(v[17]).max());a.set_title(title,fontsize=11);a.set_xlabel('x/M');a.set_ylabel('y/M')
fig.colorbar(im,ax=ax[:2],label=r'$|\delta\Theta|$ (arbitrary mode normalization)',shrink=.93)
for name,a in [('Theta',v[17]),('H',c['H']),('M',np.linalg.norm(c['M'],axis=0)),('Q',np.linalg.norm(c['Q'],axis=0))]:
 p=abs(a[:,7,7]);ax[2].semilogy(x,p/p.max(),'.-',label=name)
ax[2].set_xlabel('z/M');ax[2].set_ylabel('Normalized ray amplitude');ax[2].set_title('Ray: x = y = −0.125M',fontsize=11);ax[2].legend(fontsize=9);ax[2].grid(alpha=.2)
fig.savefig(ROOT/'leading-candidate-localization.png',dpi=180);fig.savefig(ROOT/'leading-candidate-localization.pdf')
