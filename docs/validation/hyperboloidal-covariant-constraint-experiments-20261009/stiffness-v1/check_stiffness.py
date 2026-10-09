"""Read-only local generator audit; no native/global/scri stability assertion."""
from pathlib import Path
import hashlib, json, math
import numpy as np

p = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
def mm(a,b): return np.einsum('ij,jk->ik',a,b)
def norm(a): return float(np.linalg.svd(a,compute_uv=False)[0])
def matrix(x):
    q=np.asarray(x['L']);return q[:,:,0]+1j*q[:,:,1]
def balanced(x):
    t=np.ones(20);t[12:]=x['Omega']
    return x['Omega']*matrix(x)*t[:,None]/t[None,:],t
def rk(z):
    z2=mm(z,z);return np.eye(20)+z+z2/2+mm(z2,z)/6
def expm(a):
    # Scaling/squaring Taylor, deliberately independent of eigendecomposition.
    # At the scaled infinity norm <=.5, 32 terms put truncation below 1e-44.
    size=float(np.max(np.sum(abs(a),axis=1)))
    s=max(0,math.ceil(math.log2(size/.5))) if size else 0
    b=a/(2**s);e=np.eye(20,dtype=complex);term=e.copy()
    for j in range(1,33):
        term=mm(term,b)/j;e+=term
    for _ in range(s): e=mm(e,e)
    return e
def meta(x):
    return {k:v for k,v in x.items() if k not in ('L','reference_fixedpoint')}
def scalar_rk(z):return 1+z+z*z/2+z*z*z/6

small=json.loads((p/'small-Omega.json').read_text())
fourier=json.loads((p/'Fourier.json').read_text())
assert len(small)==384 and len(fourier)==1920
assert all(np.isfinite(matrix(x)).all() and x['Omega']>0 for x in small+fourier)
small_rows=[]
for x in small:
    b,t=balanced(x);ev=np.linalg.eigvals(b);o=x['Omega']
    z=.03*b;r=rk(z);e=expm(z)
    ehalf=expm(z/2)
    assert norm(e-mm(ehalf,ehalf))/max(1,norm(e))<2e-12
    rawr=r/t[:,None]*t[None,:];rawe=e/t[:,None]*t[None,:]
    small_rows.append(dict(meta(x),
        omega_lambda_real_max=float(ev.real.max()),
        omega_spectral_radius=float(abs(ev).max()),
        omega_L_max=float(o*abs(matrix(x)).max()),
        balanced_L_max=float(abs(b).max()),balanced_L_norm=norm(b),
        lambda_theta_double_coefficient=float((o*o*matrix(x)[17,3]).real),
        dt=.03*o,rk_spectral_radius=float(abs(scalar_rk(.03*ev)).max()),
        raw_rk_norm=norm(rawr),raw_exact_norm=norm(rawe),
        balanced_rk_norm=norm(r),balanced_exact_norm=norm(e),
        balanced_rk_relative_error=norm(r-e)/norm(e)))

# Global native step values. Continuum Fourier rows beyond each grid's actual
# outer radius are excluded; these are NOT discrete stencil or boundary modes.
grids={24:.0142578125,36:.00383680555555556,48:.003251953125}
rk_rows=[];propagators=[];native_summaries=[]
for n,omin in grids.items():
    dt=.03*omin
    for x in fourier:
        if x['Omega']<omin-2e-14:continue
        b,t=balanced(x);ev=np.linalg.eigvals(b)/x['Omega'];z=dt*ev
        damped=ev.real<=1e-9
        rr=abs(scalar_rk(z));exact=abs(np.exp(z))
        rec=dict(meta(x),N=n,dt=dt,
            lambda_real_max=float(ev.real.max()),spectral_radius=float(abs(ev).max()),
            rk_radius=float(rr.max()),exact_radius=float(exact.max()),
            damped_rk_radius=float(rr[damped].max()) if damped.any() else 0,
            damped_rk_excess=float(max(0,rr[damped].max()-1)) if damped.any() else 0,
            max_scalar_relative_error=float((abs(scalar_rk(z)-np.exp(z))/np.maximum(abs(np.exp(z)),1e-300)).max()))
        rk_rows.append(rec)
        if x['kappa']!=10 or not x['norm_gauge'] or abs(x['Omega']-omin)>2e-13:continue
        if x['oblique'] and x['k']==0:continue
        for tau in [.03,.1,.3,1.,3.]:
            e=expm(tau*b);ehalf=expm(tau*b/2)
            err=norm(e-mm(ehalf,ehalf))/max(1,norm(e))
            assert err<3e-12,(meta(x),tau,err)
            raw=e/t[:,None]*t[None,:]
            propagators.append(dict(meta(x),N=n,time=tau*x['Omega'],time_over_Omega=tau,
                raw_exact_norm=norm(raw),balanced_exact_norm=norm(e),
                doubling_relative_error=err))
        r=rk(dt/x['Omega']*b);e=expm(dt/x['Omega']*b)
        native_summaries.append(dict(meta(x),N=n,dt=dt,
            raw_rk_norm=norm(r/t[:,None]*t[None,:]),
            raw_exact_norm=norm(e/t[:,None]*t[None,:]),
            balanced_rk_norm=norm(r),balanced_exact_norm=norm(e),
            balanced_rk_relative_error=norm(r-e)/norm(e)))

assert max(x['damped_rk_excess'] for x in rk_rows)<2e-10
# This weighted estimate is a sampled analytical similarity, NOT a runtime
# transformation/falloff condition. A+Lambda weights are essential.
assert max(x['balanced_L_norm'] for x in small_rows)<200
summary={
    'passed_finite_Omega_local_numerical_gate':True,
    'native_stability_accepted':False,'nonlinear_scri_closure_accepted':False,
    'forms':{'0':'unchanged C0','1':'mechanical Appendix B C1','2':'C1 plus derived covariant connection repair'},
    'input_sha256':{f:sha(p/f) for f in ['full20.cpp','Fourier.json','small-Omega.json']},
    'configuration_order':['alpha','chi','physical_P','physical_Theta','beta_x','beta_y','beta_z','g00','g01','g02','g11','g12','A00','A01','A02','A11','A12','Lambda_x','Lambda_y','Lambda_z'],
    'similarity':'T=diag(1 indices0..11, Omega indices12..19), B=Omega*T*L*T^-1; analysis only',
    'fourier_rows':len(fourier),'small_Omega_rows':len(small),
    'native_RK_rows':len(rk_rows),'propagator_rows':len(propagators),
    'max_damped_RK_excess':max(x['damped_rk_excess'] for x in rk_rows),
    'reference_fixedpoint_max_Fourier':max(x['reference_fixedpoint'] for x in fourier),
    'reference_fixedpoint_max_small_Omega':max(x['reference_fixedpoint'] for x in small),
    'small_Omega_max_balanced_L_norm':max(x['balanced_L_norm'] for x in small_rows),
    'small_Omega_max_scaled_spectral_radius':max(x['omega_spectral_radius'] for x in small_rows),
    'small_Omega':small_rows,'native_RK':rk_rows,
    'native_step_norms':native_summaries,'propagators':propagators,
    'scope':['Actual20 algebraically reduced local continuum Fourier kernel, not native stencil/ghost/global generator',
             'Finite physicalTheta/background perturbations impose no Omega falloff',
             'Small positive primitive frozen roots are retained; no subsidiary eigenmode identification',
             'No uniform unweighted propagator or complete nonlinear scri closure claim',
             'The highest continuum frequencies exceed finite-grid Nyquist and are diagnostic samples',
             'Native final-stage algebraic projection and full22 RK require separate actual one-step verification']}
(p/'check-report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print('PASS finite-Omega local numerical gate; no native acceptance')
print('damped RK excess',summary['max_damped_RK_excess'])
print('max balanced smallOmega norm',summary['small_Omega_max_balanced_L_norm'])
