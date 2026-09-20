"""Summarize the bounded full-state endpoint-closure test, including failures."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.fft import dct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent/'radial-volume'))
import radial_operator as volume
REFERENCE=-.0727906934460

def offgrid(data):
    z=data['near_constraint_only_branch'][0]
    y=np.asarray(z['full_state_real'])+1j*np.asarray(z['full_state_imag'])
    degree=data['degree'];co=dct(y,type=1,axis=1)/degree;co[:,0]*=.5;co[:,-1]*=.5
    inner,outer=data['inner_M'],data['outer_M'];scale=2/(outer-inner)
    # Midpoint samples independent of the Chebyshev collocation grid.
    r=np.linspace(inner,outer,512,endpoint=False)+(outer-inner)/1024
    x=(2*r-inner-outer)/(outer-inner)
    vals=[]
    for derivative in range(3):
        vals.append(np.array([np.polynomial.chebyshev.chebval(x,np.polynomial.chebyshev.chebder(c,derivative)*scale**derivative) for c in co]))
    U,DU,D2U=vals
    A2,A1,A0=volume.coefficients(r,data['kappa1'])
    rhs=np.einsum('rij,jr->ir',A2,D2U)+np.einsum('rij,jr->ir',A1,DU)+np.einsum('rij,jr->ir',A0,U)
    lam=z['real_per_M']+1j*z['imag_per_M'];res=lam*U-rhs
    return {'sample_count':len(r),'relative_L2_full_PDE':float(np.linalg.norm(res)/(abs(lam)*np.linalg.norm(U)+np.linalg.norm(rhs))),'max_abs_full_PDE_Theta_normalized':float(np.max(abs(res))),'max_last_8_chebyshev_coefficients_over_max_coefficient':float(np.max(abs(co[:,-8:]))/np.max(abs(co)))}

def main():
    rows=[]
    for n in [24,32,48,64,80]:
        d=json.loads((ROOT/f'full-areal-n{n}.json').read_text());z=d['near_constraint_only_branch'][0];f=d['fastest_finite_mode']
        rows.append({'degree':n,'collocation_points':n+1,'finite_eigenvalues':d['spectrum_summary']['finite_count'],'infinite_eigenvalues':d['spectrum_summary']['infinite_count'],'positive_real_count_above_1e-8':d['spectrum_summary']['positive_count_1e-8'],'constraint_branch_real_per_M':z['real_per_M'],'constraint_branch_imag_per_M':z['imag_per_M'],'constraint_branch_abs_eigenvalue_error':abs(complex(z['real_per_M'],z['imag_per_M'])-REFERENCE),'constraint_branch_endpoint_PDE_relative':z['full_PDE_relative_residual_replaced_endpoint_rows'],'constraint_branch_dynamic_radiation_relative':z['radiative_constraint_dynamic_residuals_relative'],'constraint_branch_algebraic_BC_max_relative':max(z['algebraic_boundary_relative_residuals']),'constraint_branch_offgrid':offgrid(d),'fastest_real_per_M':f['real_per_M'],'fastest_imag_per_M':f['imag_per_M'],'fastest_real_over_degree_squared':f['real_per_M']/n**2,'fastest_endpoint_PDE_relative':f['full_PDE_relative_residual_replaced_endpoint_rows'],'fastest_dynamic_radiation_relative':f['radiative_constraint_dynamic_residuals_relative'],'fastest_algebraic_BC_max_relative':max(f['algebraic_boundary_relative_residuals'])})
    out={'scope':'Linear spherical full-state radial annulus [0.2,4]M, exact R0=M=1 trumpet, adapted G2 gauge, original kappa1=0.1 and kappa2=0. Five-row endpoint replacement is NOT a validated stable closure.','reference_constraint_only_real_per_M':REFERENCE,'reference':'../annulus-areal-sigma0.1-scaled0-n128.json','conclusion':'Converging damped physical-constraint branch, but resolution-growing positive grid modes and O(1) dynamic boundary defects invalidate this endpoint replacement as a stable full-state discretization.','rows':rows}
    (ROOT/'convergence.json').write_text(json.dumps(out,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,axs=plt.subplots(1,3,figsize=(12,3.4),layout='constrained');N=np.array([r['degree'] for r in rows])
    axs[0].plot(N,[r['constraint_branch_real_per_M'] for r in rows],'o-',label='Full-state branch')
    axs[0].axhline(REFERENCE,color='k',ls='--',label='Constraint-only reference');axs[0].set(ylabel=r'Re $\lambda$ [$M^{-1}$]',xlabel='Chebyshev degree');axs[0].legend(fontsize=8)
    axs[1].semilogy(N,[r['constraint_branch_abs_eigenvalue_error'] for r in rows],'o-',label='Eigenvalue difference')
    axs[1].semilogy(N,[r['constraint_branch_offgrid']['relative_L2_full_PDE'] for r in rows],'s-',label='Off-grid PDE residual')
    axs[1].set(xlabel='Chebyshev degree',ylabel='Absolute / relative error');axs[1].legend(fontsize=8)
    axs[2].plot(N,[r['fastest_real_per_M'] for r in rows],'o-',label='Fastest positive grid mode')
    axs[2].plot(N,rows[-1]['fastest_real_over_degree_squared']*N**2,'k--',label=r'$\propto N^2$');axs[2].set(xlabel='Chebyshev degree',ylabel=r'Max Re $\lambda$ [$M^{-1}$]');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(ROOT/'full-state-closure-convergence.png',dpi=180);plt.close(fig)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
