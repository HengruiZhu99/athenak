#!/usr/bin/env python3
"""Unsmoothed final constraint and field-gradient localization on a saved grid."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physical_constraints import evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 initial=np.load(a.run/'initial.npz');state=np.load(sorted(a.run.glob('profile-*.npz'))[-1]);r=state['r'];q=state['q'];d=state['d'];om=initial['omega'];cfg=initial['configuration'];phys=evaluate(r,q,d,state['dd'],om);phys[:,:2]/=9;phys[:,2:]/=3
 R=3*r[:-1]/(om[:-1,0]*np.sqrt(initial['q'][:-1,0]));mask=(R>=25)&(R<=100)&(r[:-1]>cfg[3]/6);rows=[]
 for j,name in enumerate(('H','M','Theta','Z')):
  inds=np.flatnonzero(mask);i=inds[np.argmax(abs(phys[mask,j]))];rows.append(dict(field=name,max=float(abs(phys[i,j])),compact_r=float(r[i]),reference_R_M=float(R[i])))
 features=[]
 for j,name in ((8,'alpha'),(9,'beta'),(7,'Lambda'),(11,'D')):
  valid=np.flatnonzero((r>.2)&(r<.999));peak=valid[np.argmax(abs(d[valid,j]))];lo=hi=peak;threshold=abs(d[peak,j])/2
  while lo>valid[0] and abs(d[lo-1,j])>=threshold:lo-=1
  while hi<valid[-1] and abs(d[hi+1,j])>=threshold:hi+=1
  features.append(dict(field=name,peak=float(abs(d[peak,j])),peak_r=float(r[peak]),halfmax_gridpoints=int(hi-lo+1),halfmax_extent=float(r[hi]-r[lo])))
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 for j,name in enumerate(('H','M','Theta','Z')):axes[0,0].semilogy(r[:-1],np.maximum(abs(phys[:,j]),1e-15),label=name)
 for j,name in ((8,'alpha'),(9,'beta')):axes[0,1].plot(r,q[:,j],label=name)
 for j,name in ((7,'Lambda'),(11,'D')):axes[1,0].plot(r,q[:,j],label=name)
 for j,name in ((8,'alpha'),(9,'beta'),(7,'Lambda'),(11,'D')):axes[1,1].semilogy(r,np.maximum(abs(d[:,j]),1e-15),label=name)
 for ax in axes.flat:
  ax.set_xlim(.2,1);ax.set_xlabel('Compact radius');ax.grid(alpha=.25);ax.legend(fontsize=8)
  for x in cfg[[4,5,7,8,12]]:ax.axvline(x,color='grey',lw=.6,alpha=.4)
 axes[0,0].set(ylabel='Dimensionless physical constraint',ylim=(1e-9,1e2),title='Constraint localization')
 axes[0,1].set(ylabel='Total field',title='Lapse and shift')
 axes[1,0].set(ylabel='Total field',title='Independent connection and driver')
 axes[1,1].set(ylabel='Absolute radial derivative',ylim=(1e-8,1e5),title='Unsmoothed derivatives')
 fig.savefig(a.output/'localization.png',dpi=160)
 (a.output/'localization.json').write_text(json.dumps(dict(run=str(a.run),time_M=float(state['t_M']),reference_exterior_maxima=rows,gradient_features=features,join_radii=cfg[[4,5,7,8,12]].tolist()),indent=2)+'\n');print(json.dumps(rows));print(json.dumps(features))
if __name__=='__main__':main()
