#!/usr/bin/env python3
"""Compare staged trial surfaces and their independently sampled null expansion."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    rows=json.loads((args.sequence/'summary.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    colors=['#7e8791', '#dc8c29', '#2778b5', '#1b9675', '#9256ac']
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.facecolor':'white'})
    comparison=[]
    for signed in [False, True]:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.75]},layout='constrained')
        for r,color in zip(rows,colors):
            folder=Path(r['output'])/'search'
            dense_path=folder/f"mots.mots_dense_{r['state']['cycle']}_{r['row']}.json"
            if not dense_path.exists():
                dense_path=folder/f"mots.mots_dense_{r['row']}.json"
            dense=json.loads(dense_path.read_text())
            d=np.genfromtxt(folder/f"mots.mots_surface_{r['state']['cycle']}_{r['row']}.csv",delimiter=',',names=True)
            assert dense['valid'] and np.all(d['valid']==1)
            ra=np.sqrt(dense['area']/(4*np.pi))
            v=ra*d['theta_plus']
            order=np.argsort(d['theta'])
            for sign in [-1,1]:
                axes[0].plot(sign*d['rho'],d['z'],color=color,lw=1.5,
                             label=f"L={r['lmax']}" if sign==1 else None)
            label=rf"$L={r['lmax']}$  ($\epsilon_2={dense['epsilon2']:.2e}$)"
            axes[1].plot(d['theta'][order]/np.pi,v[order] if signed else np.maximum(abs(v[order]),1.e-16),
                         color=color,lw=1.15,label=label)
            if not signed:
                comparison.append({**r,'dense':dense,'dense_profile_max':float(max(abs(v)))})
        axes[0].set(xlabel='Meridional x',ylabel='z',title='Trial surfaces')
        axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(fontsize=9,ncol=2)
        ax=axes[1]
        if signed:
            ax.set_yscale('symlog',linthresh=1.e-5)
            ax.axhline(0,color='black',lw=.7)
            ax.axhspan(-1.e-5,1.e-5,color='black',alpha=.06)
            ax.set_ylabel(r'$R_A\Theta_+$  (signed; symmetric log scale)')
        else:
            ax.set_yscale('log');ax.set_ylim(1.e-7,1)
            ax.axhline(1.e-5,color='black',ls='--',lw=1,label=r'$\epsilon_\infty$ target')
            ax.set_ylabel(r'$R_A|\Theta_+|$')
        ax.set(xlabel=r'$\theta/\pi$',xlim=(0,1),title='Expansion on a common 1,061-point grid')
        ax.set_xticks([0,.25,.5,.75,1]);ax.grid(alpha=.16)
        ax.legend(fontsize=9,loc='lower right' if not signed else 'upper right')
        levels=' → '.join(str(r['lmax']) for r in rows)
        fig.suptitle(f'A = −0.05 · N512 · t = 50\nAngular continuation L = {levels} — no verified MOTS',fontsize=13)
        name='late_trial_angular_signed' if signed else 'late_trial_angular'
        for ext in ['png','pdf']:
            fig.savefig(args.output/f'{name}.{ext}',dpi=190)
        plt.close(fig)
    (args.output/'angular_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['lmax','epsilon2','dense','iterations']} for r in comparison],indent=2))


if __name__=='__main__': main()
