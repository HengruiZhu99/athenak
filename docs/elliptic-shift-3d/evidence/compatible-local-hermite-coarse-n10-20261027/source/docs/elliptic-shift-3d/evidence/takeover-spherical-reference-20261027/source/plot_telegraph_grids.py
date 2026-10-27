#!/usr/bin/env python3
"""Show resolution along the entire physical radius, not only the first cell."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radial_grid import mapped_grid
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=2049);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained');x=np.linspace(0,1,a.n);records=[]
for strength,kind,label in [(0,'sinh','Uniform'),(6,'inner-c7','Inner C7, s=6'),(14,'inner-c7','Inner C7, s=14'),(16,'inner-exp','Inner exponential, b=16')]:
 r,J=mapped_grid(x,strength,kind);mid=(r[1:]+r[:-1])/2;relative=np.diff(r)/mid
 axes[0].semilogy(x[1:],r[1:],label=label)
 axes[1].loglog(mid,relative,label=label)
 records.append(dict(kind=kind,strength=strength,first_cell=float(r[1]),max_J=float(max(J)),relative_spacing_at_r={str(z):float(np.interp(z,mid,relative)) for z in [1e-7,1e-5,.0005,.004,.07,.7]}))
axes[0].set(xlabel='Uniform computational coordinate x',ylabel='Physical coordinate r',title='Fixed numerical maps',ylim=(1e-10,1))
axes[1].set(xlabel='Physical cell midpoint r',ylabel=r'$\Delta r/r$',title=f'Relative radial spacing, N={a.n}',xlim=(1e-9,1),ylim=(1e-4,2))
for ax in axes:ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(a.output/'grid-resolution.png',dpi=160);fig.savefig(a.output/'grid-resolution.pdf');plt.close(fig)
(a.output/'review.json').write_text(json.dumps(dict(n=a.n,records=records,scope='Grid geometry only; smaller first spacing or relative spacing does not establish PDE convergence.'),indent=2)+'\n')
