#!/usr/bin/env python3
"""Plot recorded spherical controls without labeling completion as stability."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def series(path):
    rows=list(csv.DictReader(path.open()));groups={}
    for r in rows:groups.setdefault(float(r['time']),[]).append(r)
    result=[]
    for t,rs in sorted(groups.items()):
        if len(rs)!=2:continue
        vol=sum(float(r['volume']) for r in rs)
        h=np.sqrt(sum(float(r['volume'])*float(r['H_l2'])**2 for r in rs)/vol)
        m=np.sqrt(sum(float(r['volume'])*float(r['M_conformal_l2'])**2 for r in rs)/vol)
        result.append([t,h,m,min(float(r['min_alpha']) for r in rs),
                       min(float(r['min_chi']) for r in rs),max(float(r['H_linf']) for r in rs)])
    return np.asarray(result)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--case',action='append',nargs=2,metavar=('LABEL','CSV'),required=True)
    args=ap.parse_args();fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    titles=['Hamiltonian L2 · combined coordinate volume','Momentum L2 · combined coordinate volume',
            'Minimum lapse','Minimum chi']
    for label,path in args.case:
        s=series(Path(path));assert len(s)
        for ax,col in zip(axs.flat,[1,2,3,4]):ax.plot(s[:,0],s[:,col],label=label,lw=1.5)
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title);ax.set_xlabel('Evolution time');ax.grid(alpha=.2)
        for t,c in [(.79277343749968521,'#777777'),(1.5837303635638276,'#aaaaaa')]:
            ax.axvline(t,color=c,ls=':',lw=1)
    axs[0,0].set_yscale('log');axs[0,1].set_yscale('log')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Fresh spherical-boundary controls · original pulse and κ₁ = 10',fontsize=14)
    fig.supxlabel('Dotted lines: old Cartesian N24/N32 failure times. Grids and discrete boundary operators differ.',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
