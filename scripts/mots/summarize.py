#!/usr/bin/env python3
"""Plot frozen-search evidence; failed surfaces are never labeled horizons."""
import argparse
import json
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--qualification',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted(args.results.glob('*/search/*.cartoon_m0_horizon_0.txt')):
        rows=[line.split() for line in path.read_text().splitlines() if not line.startswith('#')]
        valid=[(i,r) for i,r in enumerate(rows) if float(r[7])>0 and math.isfinite(float(r[13]))]
        if not valid:continue
        i,best=min(valid,key=lambda p:float(p[1][13]))
        state=json.loads((path.parent/'frozen_mots.json').read_text())
        record={'case':path.parents[1].name,'time':float(best[1]),'best_row':i,
            'best_epsilon2':float(best[13]),'best_epsilon_inf':float(best[-5]),
            'failure':best[15],'verified_count':sum(int(r[-6]) for r in rows),
            'candidate_count':len(rows),'area':float(best[7]),'center':float(best[4]),
            'spacing':float(best[-2]),'state':state,'history':str(path)}
        records.append(record)
    (args.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    late=[r for r in records if r['case'].startswith('n512-t50')]
    if late:
        best=min(late,key=lambda r:r['best_epsilon2'])
        path=Path(best['history']).parent/f"mots.mots_surface_{best['state']['cycle']}_{best['best_row']}.csv"
        data=np.genfromtxt(path,delimiter=',',names=True)
        radius=math.sqrt(best['area']/(4*math.pi))
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for sign in [-1,1]:axes[0].plot(sign*data['rho'],data['z'],color='C0')
        axes[0].set(xlabel='Meridional x',ylabel='z',title='Best trial surface — not a verified MOTS')
        axes[0].set_aspect('equal',adjustable='datalim')
        axes[1].semilogy(data['theta'],np.maximum(abs(data['theta_plus'])*radius,1e-16))
        axes[1].axhline(1e-5,color='k',ls='--',label=r'$\epsilon_\infty$ target')
        axes[1].set(xlabel=r'$\theta$',ylabel=r'$R_A |\Theta_+|$',title=best['case'])
        axes[1].legend();fig.savefig(args.output/'late_trial.png',dpi=180);fig.savefig(args.output/'late_trial.pdf');plt.close(fig)
    scan=[r for r in records if r['case'].startswith('n512-') and r['case'].endswith('L16')]
    if scan:
        scan.sort(key=lambda r:r['time'])
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        ax.semilogy([r['time'] for r in scan],[r['best_epsilon2'] for r in scan],'o-',label='Best search residual, L=16')
        ax.axhline(1e-6,color='k',ls='--',label='Verification target')
        ax.set(xlabel='Saved coordinate time',ylabel=r'$\epsilon_2$',title='A = −0.05, N512 frozen-slice searches')
        ax.legend();fig.savefig(args.output/'time_scan.png',dpi=180);fig.savefig(args.output/'time_scan.pdf');plt.close(fig)
    if args.qualification:
        qual=json.loads(args.qualification.read_text())
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for suffix,label in [('', 'Uniform grid'),('-amr','Refinement interface crossing')]:
            points=[(int(r['case'].split('-')[0][1:]),r['residual']) for r in qual if r['case'].endswith('-amr')==bool(suffix)]
            ax.loglog(*zip(*points),'o-',label=label)
        ax.axhline(1e-6,color='k',ls='--',label='Verification target')
        ax.set(xlabel='Root resolution N',ylabel=r'$\epsilon_2$',title='Native VC Schwarzschild sampling and solve')
        ax.legend();fig.savefig(args.output/'qualification.png',dpi=180);fig.savefig(args.output/'qualification.pdf');plt.close(fig)
    print(json.dumps([{k:r[k] for k in ['case','time','best_epsilon2','verified_count']} for r in records],indent=2))


if __name__=='__main__':main()
