"""Plot native algebra/operator evidence; no evolution curves are invented."""
import argparse,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
d=json.loads((a.evidence/'native/audit.json').read_text())['obstruction']
theta=np.linspace(0,np.pi/2,401);g=np.array(d['metric'])
Q=np.cos(theta)**2/g[0,0]+np.sin(theta)**2/g[1,1]
gap=4*d['mu']/3-d['chi']*d['F']-d['eta']*d['chi']*.25/Q
rows=re.findall(r'resolution=(\d+) derivative_error=([\deE+.\-]+)',(a.evidence/'mapped.log').read_text())
n,error=np.array(rows,dtype=float).T
fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
ax[0].plot(theta*180/np.pi,gap);ax[0].plot([90],[0],'rx',ms=9)
ax[0].set(xlabel='Angle from radial axis [degrees]',ylabel='(shift speed² − lapse speed²) / Q',title='Positive radial gap misses a tangential defect')
ax[1].loglog(n,error,'o-',label='Full mapped J/H operator')
ax[1].loglog(n,error[0]*(n/n[0])**-4,'--',label='Fourth-order guide')
ax[1].set(xlabel='Inverse logical spacing',ylabel='Maximum derivative error',title='Constant/linear Cartesian modes and lapse')
ax[1].legend()
for axis in ax:axis.grid(alpha=.2)
fig.suptitle('3D point and mapped-operator checks — not evolution')
fig.savefig(a.evidence/'native-admission.png',dpi=170)
