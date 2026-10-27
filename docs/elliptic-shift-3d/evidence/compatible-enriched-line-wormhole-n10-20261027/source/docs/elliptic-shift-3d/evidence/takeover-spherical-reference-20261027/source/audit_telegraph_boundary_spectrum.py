#!/usr/bin/env python3
"""Frozen outer-block spectrum: a diagnostic, not full-system stability proof."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegraph import options,TelegraphEngine
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole


def audit(n,width,kind,library,fast_library):
    args=options(n,library);args.grid_map=kind;args.grid_stretch=16.;args.mapped_ko='local';args.ko_order=8;args.width=width
    ev=PunctureChart(args);q,*_=ev.initial_wormhole(IsotropicWormhole(1/3));B=np.zeros(n)
    engine=TelegraphEngine(ev,fast_library,.005,.02,inner_profile='af-regular',width=.24)
    # The fixed lapse clock cannot be perturbed. Other endpoint conditions
    # remain the actual SAT evolution; inner perturbations are held zero.
    indices=np.array([8*i+j for i in np.flatnonzero(ev.r>=.9) for j in [0,2,3,4,5,6] if not (i==n-1 and j==6)])
    measurements=[]
    for epsilon in [1e-7,5e-8]:
        matrix=np.empty((len(indices),len(indices)))
        for column,k in enumerate(indices):
            plus=q.copy();minus=q.copy();plus.flat[k]+=epsilon;minus.flat[k]-=epsilon
            matrix[:,column]=(engine.rhs(plus,B)[0].ravel()[indices]-engine.rhs(minus,B)[0].ravel()[indices])/(2*epsilon)
        eigen,vectors=np.linalg.eig(matrix);j=int(np.argmax(eigen.real));value=eigen[j]
        mode=np.zeros(q.size,dtype=complex);mode[indices]=vectors[:,j];mode/=max(abs(mode))
        # Apply the full linearized RHS to the restricted eigenvector. Real
        # and imaginary parts are differentiated separately if necessary.
        action=np.zeros(q.size,dtype=complex)
        for component,factor in [(mode.real,1.),(mode.imag,1j)]:
            if max(abs(component))==0:continue
            plus=q+epsilon*component.reshape(q.shape);minus=q-epsilon*component.reshape(q.shape)
            action+=factor*(engine.rhs(plus,B)[0].ravel()-engine.rhs(minus,B)[0].ravel())/(2*epsilon)
        residual=float(np.linalg.norm(action-value*mode)/max(np.linalg.norm(action),1e-300))
        dt=.05/(n-1);z=dt*eigen;rk=1+z+z*z/2+z**3/6+z**4/24
        strength=np.sqrt(np.sum(abs(mode.reshape(q.shape))**2,axis=1));strength/=max(strength)
        measurements.append(dict(perturbation=epsilon,largest_real=float(value.real),imaginary=float(value.imag),
            growing_modes=int(sum(eigen.real>1e-5)),RK4_amplification_max=float(max(abs(rk))),
            full_operator_mode_residual=residual,mode_r=ev.r.tolist(),mode_strength=strength.tolist()))
    engine.close()
    return dict(n=n,width=width,grid_map=kind,outer_block_dimension=len(indices),measurements=measurements)


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--fast-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
    for n in [129,257,513]:
        for width,kind in [(7,'inner-exp'),(9,'inner-exp'),(7,'inner-exp-scri')]:
            r=audit(n,width,kind,a.library,a.fast_library);records.append(r)
            print(n,width,kind,[(m['largest_real'],m['full_operator_mode_residual']) for m in r['measurements']],flush=True)
    root=Path(__file__).resolve().parent
    sources={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in list(sys.modules.values()) if getattr(module,'__file__',None) and Path(module.__file__).resolve().parent==root}
    for name in ['fast_integrator.cpp','telegraph_integrator.cpp']:
        sources[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    result=dict(records=records,source_sha256=sources,library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        fast_library_sha256=hashlib.sha256(a.fast_library.read_bytes()).hexdigest(),mass=1/3,cutoff=.9,
        scope='Frozen linearization at analytic wormhole initial data, with perturbations restricted to r>=.9 and the lapse-clock variation fixed to zero. Exact nonlinear RHS and elliptic solve are differentiated; the independent interior perturbations are held zero. Two central-difference amplitudes test sensitivity. Full-operator residual tests how well the restricted eigenvector extends to the complete grid. Nonpositive block eigenvalues and RK4 amplification do not prove nonlinear, nonnormal or full-system stability.')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for width,kind,label in [(7,'inner-exp','width7'),(9,'inner-exp','width9'),(7,'inner-exp-scri','width7, refined scri')]:
        rows=[r for r in records if r['width']==width and r['grid_map']==kind]
        axes[0].plot([r['n']-1 for r in rows],[r['measurements'][-1]['largest_real'] for r in rows],'o-',label=label)
    for rec in records:
        if rec['width']!=9:continue
        m=rec['measurements'][-1];r=np.array(m['mode_r']);y=np.array(m['mode_strength']);mask=(y>0)&(r>.85)
        axes[1].semilogy((1-r[mask])/(r[-1]-r[-2]),y[mask],label=f"N={rec['n']}")
    axes[0].axhline(0,color='0.5',lw=.8);axes[0].set(title='Largest real part of outer-block eigenvalues',xlabel='N - 1',ylabel=r'$\max\mathrm{Re}\,\lambda$')
    axes[1].set(title='Growing width9 mode near scri',xlabel='Distance from scri / last grid spacing',ylabel='Normalized mode magnitude',xlim=(0,20),ylim=(1e-8,1.1))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(a.output/'spectrum.png',dpi=160);fig.savefig(a.output/'spectrum.pdf');plt.close(fig)

if __name__=='__main__':main()
