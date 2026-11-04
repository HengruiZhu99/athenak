#!/usr/bin/env python3
"""Plot unsmoothed exported Omega/gauge profiles."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('profile_csv',type=Path)
parser.add_argument('output',type=Path)
a=parser.parse_args()
rows=list(csv.DictReader(a.profile_csv.open()))
r=np.array([float(x['compact_r']) for x in rows])
value=lambda key: np.array([float(x[key]) for x in rows])
def weight(begin):
 z=np.clip((.56-r)/(.56-begin),0,1)
 t=np.minimum(z,1-z)
 s=t**5*(126+t*(-420+t*(540+t*(-315+70*t))))
 return np.where(z>.5,1-s,s)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
axes[0,0].plot(r,(1-r*r)/2,label='Legacy',color='#3b617d')
axes[0,0].plot(r,value('omega'),label='Flat interior',color='#bd5b36')
axes[0,0].set(ylabel=r'$\Omega$',ylim=(.31,.51));axes[0,0].legend()
axes[0,1].plot(r,-r,label='Legacy',color='#3b617d')
axes[0,1].plot(r,value('domega'),label='Flat interior',color='#bd5b36')
axes[0,1].set(ylabel=r'$d\Omega/dr$')
axes[1,0].plot(r,np.full_like(r,-1),color='#3b617d')
axes[1,0].plot(r,value('ddomega'),color='#bd5b36')
axes[1,0].set(ylabel=r'$d^2\Omega/dr^2$')
axes[1,1].plot(r,weight(.2),label='Legacy gauge',color='#3b617d')
axes[1,1].plot(r,value('inner_gauge_weight'),label='Adapted gauge',color='#bd5b36')
axes[1,1].set(ylabel='Inner RHS weight');axes[1,1].legend()
for ax in axes.flat:
 ax.set(xlim=(0,.6),xlabel='Compact Cartesian radius r')
 ax.axvspan(.18,.22,color='#bd5b36',alpha=.08)
 ax.axvline(.18,color='0.45',ls=':',lw=.8)
 ax.axvline(.28,color='0.45',ls=':',lw=.8)
 ax.grid(alpha=.15)
fig.suptitle('Flat-Omega experiment: analytic profiles without smoothing')
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=180)
