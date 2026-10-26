#!/usr/bin/env python3
"""Regular constraint-solved CMC Einstein--massless-scalar initial data.

The conformal factor is solved, not used as a constraint projection in evolution.
psi=1+Omega^3 w factors the regular mass branch of Appendix-C Hamiltonian data.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from numerics import weights
from kernel import Kernel


def operators(n,width=7,grid=None):
    r=np.linspace(0,1,n+1) if grid is None else np.asarray(grid);ext=np.r_[-r[1:width][::-1],r]
    mapping=np.r_[np.arange(width-1,0,-1),np.arange(n+1)]
    D1=np.zeros((n+1,n+1));D2=np.zeros_like(D1)
    for i,x in enumerate(r):
        center=width-1+i;start=min(max(center-width//2,0),len(ext)-width)
        idx=np.arange(start,start+width)
        for mat,order in [(D1,1),(D2,2)]:
            w=weights(ext[idx],x,order)
            for j,v in zip(mapping[idx],w):mat[i,j]+=v
    for matrix in [D1,D2]:
        np.fill_diagonal(matrix,0.)
        matrix[np.diag_indices(n+1)]=-matrix.sum(axis=1)
    D1[0]=0. # Exact regular-center even parity.
    return r,D1,D2


def solve_data(n,amplitude=.058,tolerance=1e-11,grid=None,precision='float'):
    start=time.perf_counter();r,D1,D2=operators(n,grid=grid);O=(1-r*r)/2;A=(1+r*r)/2
    phi=amplitude*np.exp(-(r*r-.25**2)**2/(4*.1**4))
    phi_r=-r*(r*r-.25**2)/.1**4*phi
    coef=np.zeros_like(r);coef[1:]=2*O[1:]/r[1:]-5*r[1:]
    L=O[:,None]*D2+coef[:,None]*D1-15*np.eye(n+1)
    L[0]=3*O[0]*D2[0];L[0,0]-=15
    source=np.zeros_like(r);source[:-1]=np.pi*phi_r[:-1]**2/O[:-1]**2
    w=np.zeros_like(r);iterations=[];converged=False
    def derivative(matrix,z):
        return np.einsum('ij,ij->i',matrix,z[None,:]-z[:,None],optimize=False)
    def residual(z):
        dz=derivative(D1,z);ddz=derivative(D2,z)
        linear=O*ddz+coef*dz-15*z;linear[0]=3*O[0]*ddz[0]-15*z[0]
        return (linear+source+np.pi*O*phi_r**2*z-7.5*O**2*z**2
                -7.5*O**5*z**3-3.75*O**8*z**4-.75*O**11*z**5)
    if precision not in ('float','decimal'):raise ValueError('Unknown initial-data arithmetic')
    if precision=='decimal':
        # High-precision residual and iterate; double Jacobian solves supply
        # Newton corrections (iterative refinement), never loosen the tolerance.
        from decimal import Decimal,localcontext
        with localcontext() as context:
            context.prec=50;D=Decimal.from_float
            nodes=[np.flatnonzero((D1[i]!=0)|(D2[i]!=0)) for i in range(n+1)]
            rows=[[(int(j),D(float(D1[i,j])),D(float(D2[i,j]))) for j in nodes[i]] for i in range(n+1)]
            od=[D(float(x)) for x in O];cd=[D(float(x)) for x in coef];sd=[D(float(x)) for x in source]
            matter=[D(float(x)) for x in np.pi*O*phi_r**2]
            def precise_residual(z):
                values=[]
                for i,row in enumerate(rows):
                    dz=sum((a*(z[j]-z[i]) for j,a,b in row),Decimal(0))
                    ddz=sum((b*(z[j]-z[i]) for j,a,b in row),Decimal(0))
                    x=z[i];o=od[i]
                    linear=(3*o*ddz if i==0 else o*ddz+cd[i]*dz)-15*x
                    values.append(linear+sd[i]+matter[i]*x-Decimal('7.5')*o**2*x**2
                        -Decimal('7.5')*o**5*x**3-Decimal('3.75')*o**8*x**4-Decimal('.75')*o**11*x**5)
                return values
            wd=[Decimal(0)]*(n+1)
            for step in range(30):
                E=precise_residual(wd);norm=max(abs(x) for x in E)
                iterations.append({'iteration':step,'max_regular_residual':float(norm),'arithmetic':'50-digit Decimal residual/iterate, double Jacobian correction'})
                if norm<D(float(tolerance)):converged=True;break
                w=np.array([float(x) for x in wd])
                J=L+np.diag(np.pi*O*phi_r**2-15*O**2*w-22.5*O**5*w**2-15*O**8*w**3-3.75*O**11*w**4)
                delta=np.linalg.solve(J,-np.array([float(x) for x in E]));fraction=1.
                while fraction>2**-16:
                    trial=[x+D(float(fraction*y)) for x,y in zip(wd,delta)]
                    if min(1+o**3*x for o,x in zip(od,trial))>0 and max(abs(x) for x in precise_residual(trial))<norm:break
                    fraction/=2
                else:break
                wd=trial;iterations[-1]['accepted_fraction']=fraction
            w=np.array([float(x) for x in wd]);precise_norm=float(max(abs(x) for x in precise_residual(wd)))
    for step in range(30 if precision=='float' else 0):
        E=residual(w);norm=float(max(abs(E)));iterations.append({'iteration':step,'max_regular_residual':norm})
        if norm<tolerance:converged=True;break
        J=L+np.diag(np.pi*O*phi_r**2-15*O**2*w-22.5*O**5*w**2-15*O**8*w**3-3.75*O**11*w**4)
        delta=np.linalg.solve(J,-E);fraction=1.
        while fraction>2**-16:
            trial=w+fraction*delta
            if min(1+O**3*trial)>0 and max(abs(residual(trial)))<norm:break
            fraction/=2
        else:break
        w=trial;iterations[-1]['accepted_fraction']=fraction
    psi=1+O**3*w;chi=psi**-4
    dchi=derivative(D1,chi);ddchi=derivative(D2,chi)
    q=np.zeros((n,8));q[:,0]=chi[1:]-1
    d=np.zeros_like(q);dd=np.zeros_like(q);d[:,0]=dchi[1:];dd[:,0]=ddchi[1:]
    # r=0 omitted only in the tensor kernel's axis chart. Its field values and
    # elliptic equation are included in the BVP and extraction integrals.
    return dict(n=n,r=r,omega=O,alpha_ref=A,w=w,psi=psi,chi=chi,q=q,d=d,dd=dd,
        phi=phi,phi_r=phi_r,D1=D1,D2=D2,iterations=iterations,converged=converged,
        residual=max(abs(residual(w))) if precision=='float' else precise_norm,
        rounded_double_residual=float(max(abs(residual(w)))),precision=precision,
        runtime_seconds=time.perf_counter()-start)


def stationary_chi_lapse(initial):
    """Initial lapse only: chi_t=0 for K=-3,A_rr=0,beta=-r.

    It matches the stationary Schwarzschild exterior lapse to its compatible
    mass jets. The lapse remains live during evolution.
    """
    r=initial['r'];O=initial['omega'];w=initial['w'];psi=initial['psi']
    wp=np.einsum('ij,ij->i',initial['D1'],w[None,:]-w[:,None],optimize=False)
    psip=-3*r*O**2*w+O**3*wp
    return 2*O*r*psip/psi


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--amplitude',type=float,default=.058);p.add_argument('--resolutions',type=int,nargs='+',default=[128,256,512]);p.add_argument('--tolerance',type=float,default=1e-11);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=[];kernel=Kernel(a.library)
    for n in a.resolutions:
        z=solve_data(n,a.amplitude,a.tolerance);r=z['r'];O=z['omega'];psi=z['psi'];chi=z['chi']
        R,S,c=kernel.parts(r[1:],z['q'],z['d'],z['dd'],-r[1:],-np.ones(n),np.zeros(n),total_chi=chi[1:])
        H=c[:,0]-8*np.pi*O[1:]**2*chi[1:]*z['phi_r'][1:]**2
        psir=np.einsum('ij,ij->i',z['D1'],psi[None,:]-psi[:,None],optimize=False)
        derivative_R=np.zeros_like(r);derivative_R[:-1]=(psi[:-1]**2*z['alpha_ref'][:-1]/O[:-1]**2+2*r[:-1]*psi[:-1]*psir[:-1]/O[:-1])
        energy=2*np.pi*r*r*z['phi_r']**2*derivative_R
        mass_flux_integral=float(np.trapezoid(energy,r));bondi_mass=float(8*z['w'][-1])
        row={'n':n,'amplitude':a.amplitude,'converged':z['converged'],'regular_elliptic_residual':float(z['residual']),
          'raw_FD_H_max':float(max(abs(H))),'raw_FD_H_volume_RMS':float(np.sqrt(np.sum(r[1:]**2*H*H)/np.sum(r[1:]**2))),
          'M_max':float(max(abs(c[:,1]))),'Z_max':float(max(abs(c[:,3]))),
          'min_chi':float(min(chi)),'min_psi':float(min(psi)),'Bondi_mass_from_regular_branch':bondi_mass,
          'integrated_scalar_energy':mass_flux_integral,'mass_balance_error':abs(bondi_mass-mass_flux_integral),
          'runtime_seconds':z['runtime_seconds'],'Newton_history':z['iterations']}
        records.append(row)
        np.savez(a.output/f'initial-n{n}.npz',r=r,q=z['q'],d=z['d'],dd=z['dd'],chi=chi,phi=z['phi'],phi_r=z['phi_r'],w=z['w'],H=H)
        print(json.dumps({k:v for k,v in row.items() if k!='Newton_history'}),flush=True)
    result={'kind':'coupled Einstein--massless-scalar constraint-solved initial data only; no matter evolution',
       'formulation':'g=1,A_rr=0,K_phys=-3,Theta=Lambda=0; physical scalar Gaussian; n_bar(phi)=0, Pi=beta phi_r',
       'regularized_equation':'Omega w_rr+(5Omega_r+2Omega/r)w_r-15w-(15/2)Omega^2 w^2-(15/2)Omega^5 w^3-(15/4)Omega^8 w^4-(3/4)Omega^11 w^5+pi(1+Omega^3w)phi_r^2/Omega^2=0',
       'boundary':'even center; at actual scri w_r=-3w; psi=1+Omega^3w; no Omega floor',
       'invariant_mass':'M_B=8w(scri), independently integral 2pi r^2 phi_r^2 R_r dr',
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':records}
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if all(x['converged'] for x in records) else 1
if __name__=='__main__':raise SystemExit(main())
