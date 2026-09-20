import json
from pathlib import Path
import numpy as np
from scipy.linalg import eig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from strip_model import full_matrix,names
fig,ax=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
records=[];n=64;h=32.;x=(np.arange(n)+.5)*h
for rate in [0.,.02,.1]:
 L,_=full_matrix(n=n,h=h,angle_y=np.pi/8,mode='zero_rate',degree=1,damping=True,shift=1,lapse_damping=.1,sponge_rate=rate,sponge_cells=8.)
 val,vec=eig(L,check_finite=False);j=np.argmax(val.real);v=vec[:,j].reshape(20,n)
 w=v.copy();w[6:16]*=h;energy=np.sum(abs(w)**2,axis=0);energy/=energy.sum()
 theta=abs(v[7]);theta/=max(theta.max(),1e-300)
 ax[0].semilogy(x,energy,label=f'rate {rate:g}, gamma {val[j].real:.4g}')
 ax[1].plot(x,theta,label=f'rate {rate:g}')
 records.append(dict(rate=rate,real=float(val[j].real),imag=float(val[j].imag),peak_index=int(np.argmax(energy)),peak_x=float(x[np.argmax(energy)]),layer_fraction=float(energy[:8].sum()+energy[-8:].sum()),energy=energy.tolist(),theta_normalized=theta.tolist(),eigen_residual=float(np.linalg.norm(L@vec[:,j]-val[j]*vec[:,j])/max(1,np.linalg.norm(L@vec[:,j])))))
for a in ax:
 a.axvline(256,color='gray',lw=.8,ls='--');a.axvline(2048-256,color='gray',lw=.8,ls='--');a.set_xlabel('x from left face (M)');a.legend(fontsize=7)
ax[0].set_ylabel('normalized full-state mode energy');ax[1].set_ylabel('normalized |Theta|')
fig.savefig('sponge-surviving-modes.png',dpi=180)
Path('sponge-surviving-modes.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k not in ['energy','theta_normalized']}for r in records],indent=2))
