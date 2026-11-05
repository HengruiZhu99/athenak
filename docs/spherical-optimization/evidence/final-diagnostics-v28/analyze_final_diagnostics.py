#!/usr/bin/env python3
"""Unsmoothed finalist fields, physical lapse and accepted horizon history."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from physical_constraints import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');rows=[];inputs={}
    for folder in a.runs:
        def read(p):
            inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text()) if p.suffix=='.json' else np.load(p)
        cfg=read(folder/'manifest.json')['config'];initial=read(folder/'initial.npz');last=read(sorted(folder.glob('profile-*.npz'))[-1]);history=read(folder/'history.json');r=last['r'];q=last['q'];d=last['d'];dd=last['dd'];om=initial['omega'];n=cfg['n'];label=f'N{n}'
        R=3*r[:-1]/(om[:-1,0]*np.sqrt(initial['q'][:-1,0]));mask=(R>=25)&(R<=100)&(r[:-1]>initial['configuration'][3]/6)
        physical_lapse=q[:-1,8]/om[:-1,0];phys=evaluate(r,q,d,dd,om);phys[:,:2]/=9;phys[:,2:]/=3
        axes[0,0].plot(R[mask],q[:-1,8][mask],label=label+' conformal');axes[0,0].plot(R[mask],physical_lapse[mask],ls='--',label=label+' physical')
        for f,name in [(0,'chi'),(1,'gr')]:axes[0,1].plot(r,q[:,f],label=label+' '+name,ls='-' if f==0 else '--')
        for f,name in [(9,'beta'),(11,'D')]:axes[0,2].plot(r,q[:,f],label=label+' '+name,ls='-' if f==9 else '--')
        axes[1,0].semilogy(R[mask],np.maximum(np.max(abs(phys[mask]),axis=1),1e-16),label=label)
        for f,name in [(8,'alpha'),(0,'chi')]:axes[1,1].semilogy(r,np.maximum(abs(dd[:,f]),1e-16),label=label+' '+name,ls='-' if f==8 else '--')
        mass=[h['horizon_mass'] if h['horizon_accepted'] else np.nan for h in history];axes[1,2].plot([h['t_M'] for h in history],mass,'o-',label=label)
        ext=np.flatnonzero(mask);maxima=[]
        for j,name in enumerate(('H','M','Theta','Z')):
            i=ext[np.argmax(abs(phys[mask,j]))];maxima.append(dict(field=name,max=float(abs(phys[i,j])),r=float(r[i]),reference_R_M=float(R[i])))
        rows.append(dict(run=str(folder),n=n,time_M=float(last['t_M']),physical_lapse_exterior_range=[float(min(physical_lapse[mask])),float(max(physical_lapse[mask]))],conformal_lapse_exterior_range=[float(min(q[:-1,8][mask])),float(max(q[:-1,8][mask]))],exterior_maxima=maxima,horizon_history=[{k:h[k] for k in ('t_M','horizon_accepted','horizon_mass','horizon_r')} for h in history],regional_final={k:v for k,v in history[-1].items() if k.startswith(('join_','tail_','scri_collar_'))},floor_calls=read(folder/'result.json')['floor_calls']))
    axes[0,0].set(xlabel='Reference R/M',ylabel='Lapse',title='Conformal and physical lapse')
    axes[0,1].set(xlabel='Compact radius',ylabel='Total field',title='Conformal factor and radial metric',xlim=(.6,1))
    axes[0,2].set(xlabel='Compact radius',ylabel='Total field',title='Shift and auxiliary field',xlim=(.6,1))
    axes[1,0].set(xlabel='Reference R/M',ylabel='Max dimensionless physical constraint',title='Nodal exterior error localization')
    axes[1,1].set(xlabel='Compact radius',ylabel='Absolute unsmoothed second derivative',title='Boundary and join resolution',xlim=(.6,1))
    axes[1,2].axhline(1/3,color='k',ls=':',lw=1);axes[1,2].set(xlabel='t/M',ylabel='Accepted horizon mass',title='Gaps remain missing; reference M=1/3')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(a.output/'final-diagnostics.png',dpi=170);plt.close(fig)
    (a.output/'diagnostics.json').write_text(json.dumps(dict(runs=rows,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Unsmoothed saved total fields/jets; physical lapse excludes scri where Omega=0. Exterior uses initial reference map. Horizon interpolation and accepted roots are diagnostics, not invariant error certification; gaps are retained.'),indent=2)+'\n')
if __name__=='__main__':main()
